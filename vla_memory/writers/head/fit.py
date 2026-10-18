"""Fitting the event head: a logistic regression on z-scored features with balanced class weights, its L2 penalty chosen by
cross-validation over whole episodes, and a threshold that keeps the share of false positives at or below a set value.

    fit_logistic(X, y, l2)            the weights and the bias, by Newton's method on the weighted log loss
    auroc(scores, y)                  the area under the ROC curve (ties count half)
    threshold_at(negatives, fpr)      the smallest float32 threshold with at most floor(fpr * n) negatives at or above it
    cross_validate(X, y, groups, l2)  the out-of-fold scores and the AUROC of every fold; a fold holds whole episodes
    fit_head(X, y, groups)            the whole procedure: the L2 with the best mean AUROC, the threshold from its
                                      out-of-fold negatives, the final fit on all rows; returns the head and a report

The loss is sum_i c_i * logloss(x_i . w + b, y_i) + l2 * |w|^2, with c = 0.5 / (number of positives) for a positive and
0.5 / (number of negatives) for a negative, so that both classes weigh the same. Features are z-scored with the mean and
the standard deviation (plus 1e-6) of the rows the fit sees.
"""
from __future__ import annotations

import numpy as np

from .file import EventHead

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
FOLDS = 5
FPR = 0.01
STD_FLOOR = 1e-6
ITERATIONS = 100
TOLERANCE = 1e-9          # Newton stops when no parameter moves by more than this


# ---------------------------------------------------------------------- the logistic regression
def class_weights(y: np.ndarray) -> np.ndarray:
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    return np.where(y == 1, 0.5 / positives, 0.5 / negatives)


def objective(design: np.ndarray, y: np.ndarray, weights: np.ndarray, theta: np.ndarray, penalty: np.ndarray) -> float:
    """The weighted log loss plus the L2 penalty; design: the features with a column of ones for the bias."""
    z = design @ theta
    loss = np.sum(weights * (np.logaddexp(0.0, z) - y * z))
    return float(loss + np.sum(penalty * theta * theta))


def newton_step(design: np.ndarray, y: np.ndarray, weights: np.ndarray, theta: np.ndarray, penalty: np.ndarray):
    """The Newton step: the gradient solved against the Hessian of the objective."""
    p = 0.5 * (1.0 + np.tanh(0.5 * (design @ theta)))          # the sigmoid, without overflow
    gradient = design.T @ (weights * (p - y)) + 2.0 * penalty * theta
    hessian = (design * (weights * p * (1.0 - p))[:, None]).T @ design + np.diag(2.0 * penalty)
    return np.linalg.solve(hessian, gradient)


def fit_logistic(X, y, l2: float) -> tuple[np.ndarray, float]:
    """The weights (d,) and the bias of the balanced, L2-penalised logistic regression of y (0/1) on X (n, d)."""
    y = np.asarray(y, float)
    design = np.hstack([np.asarray(X, float), np.ones((len(y), 1))])
    weights = class_weights(y)
    penalty = np.full(design.shape[1], float(l2))
    penalty[-1] = 0.0                                             # the bias is not penalised
    theta = np.zeros(design.shape[1])
    loss = objective(design, y, weights, theta, penalty)
    for _ in range(ITERATIONS):
        step = newton_step(design, y, weights, theta, penalty)
        scale = 1.0
        while scale > 1e-6 and objective(design, y, weights, theta - scale * step, penalty) > loss:
            scale *= 0.5                                          # halve the step until the loss does not grow
        theta = theta - scale * step
        loss = objective(design, y, weights, theta, penalty)
        if np.max(np.abs(scale * step)) < TOLERANCE:
            break
    return theta[:-1], float(theta[-1])


# ---------------------------------------------------------------------- measures
def auroc(scores, y) -> float:
    """The probability that a positive scores above a negative (Mann-Whitney; ties count half); NaN without both classes."""
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if not positives or not negatives:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]    # tied scores share their mean rank
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def threshold_at(negatives, fpr: float = FPR) -> tuple[float, int]:
    """The smallest float32 threshold that at most floor(fpr * n) of the negatives' scores reach, and that number."""
    ranked = np.sort(np.asarray(negatives, np.float32))[::-1]
    allowed = min(int(np.floor(fpr * len(ranked))), len(ranked) - 1)
    return float(np.nextafter(ranked[allowed], np.float32(np.inf))), allowed


# ---------------------------------------------------------------------- the procedure
def statistics(X) -> tuple[np.ndarray, np.ndarray]:
    """The z-scoring of these rows: the mean and the standard deviation plus the floor, computed in float32."""
    X = np.asarray(X, np.float32)
    return X.mean(0), X.std(0) + np.float32(STD_FLOOR)


def zscore(X, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, np.float32) - mean) / std).astype(np.float32)


def folds_of(groups, folds: int = FOLDS) -> np.ndarray:
    """The fold of every row: the episodes in sorted order are dealt to the folds in turn."""
    position = {group: i for i, group in enumerate(sorted(set(groups)))}
    return np.array([position[group] % folds for group in groups])


def cross_validate(X, y, groups, l2: float, folds: int = FOLDS) -> tuple[np.ndarray, list]:
    """The out-of-fold scores (float32) and the AUROC of every fold."""
    fold = folds_of(groups, folds)
    y = np.asarray(y)
    oof = np.full(len(y), np.nan, np.float32)
    aurocs = []
    for f in range(folds):
        train = fold != f
        mean, std = statistics(X[train])
        w, b = fit_logistic(zscore(X[train], mean, std), y[train], l2)
        oof[~train] = (zscore(X[~train], mean, std).astype(float) @ w + b).astype(np.float32)
        aurocs.append(auroc(oof[~train], y[~train]))
    return oof, aurocs


def choose_l2(X, y, groups, l2s=L2_GRID, folds: int = FOLDS) -> tuple[int, list]:
    """The index of the L2 with the best mean out-of-fold AUROC (an exact tie keeps the earlier one) and the table."""
    table = []
    for l2 in l2s:
        oof, aurocs = cross_validate(X, y, groups, l2, folds)
        rounded = [round(a, 5) for a in aurocs]
        table.append({"l2": l2, "fold_auroc": rounded, "mean_auroc": float(np.mean(aurocs)), "oof": oof})
    best = max(range(len(l2s)), key=lambda i: (table[i]["mean_auroc"], -i))
    return best, table


def fit_head(X, y, groups, l2s=L2_GRID, folds: int = FOLDS, fpr: float = FPR, consecutive: int = 2, layer: int = 12,
             pool: str = "text") -> tuple[EventHead, dict]:
    """Fit the head on the rows X (n, d), labels y and the episode of every row; returns the head and the report."""
    y = np.asarray(y, int)
    best, table = choose_l2(X, y, groups, l2s, folds)
    oof = table[best]["oof"]
    threshold, allowed = threshold_at(oof[y == 0], fpr)
    mean, std = statistics(X)
    w, b = fit_logistic(zscore(X, mean, std), y, l2s[best])
    head = EventHead(w=w.astype(np.float32), b=np.array([b], np.float32), mean=mean, std=std, threshold=threshold,
                     consecutive=consecutive, layer=layer, pool=pool, l2=float(l2s[best]))
    report = {"l2": float(l2s[best]), "threshold": threshold, "N": consecutive, "fpr_max": fpr,
              "inner_cv": [{key: value for key, value in row.items() if key != "oof"} for row in table],
              "threshold_oof": {"negatives": int(np.sum(y == 0)), "at_or_above": int(np.sum(oof[y == 0] >= threshold)),
                                "allowed": allowed, "recall": float(np.mean(oof[y == 1] >= threshold))},
              "training": {"episodes": len(set(groups)), "rows": int(len(y)), "positives": int(np.sum(y == 1)),
                           "negatives": int(np.sum(y == 0))}}
    return head, report
