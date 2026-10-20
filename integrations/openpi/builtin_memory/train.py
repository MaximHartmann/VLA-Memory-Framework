"""The training launcher of the memory on stock openpi.

    python -m builtin_memory.train --exp-name <name> --init-checkpoint <checkpoint dir> \
        --dataset <repo id> <feature index> [--dataset <repo id> <feature index> ...] [--phases 4] [--steps 10000] \
        [--feature-stats <stats.npz>] [--checkpoint-dir <dir>] [--overwrite | --resume] [--no-wandb]

The TrainConfig is built here (config.train_config: the reference recipe). The datasets are concatenated in the given
order, each with its feature cache (sequences.py). The init checkpoint gives the weights and the norm stats; a checkpoint
without the memory (the memoryless policy) needs --feature-stats, the features' mean and std for the memory's z-scoring.

openpi's scripts/train.py builds its data loader from DataConfig.repo_id alone and turns every batch into openpi's
Observation, which drops the call histories, so train() is a copy of its main() with data.create_data_loader in its place.
Everything else is the trainer's own: init_train_state (the weight loader, the frozen parameters in bfloat16), train_step
(one optimizer step on the model's loss), the checkpoints and the logging.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import logging
import pathlib

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import wandb

import openpi
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import sharding

from . import data
from .config import train_config


def openpi_trainer():
    """openpi's scripts/train.py as a module; it is a script beside the package, in the checkout's scripts/ folder."""
    path = pathlib.Path(openpi.__file__).resolve().parents[2] / "scripts" / "train.py"
    if not path.exists():
        raise FileNotFoundError(f"{path}: the training needs an openpi checkout (its scripts/ next to src/)")
    spec = importlib.util.spec_from_file_location("openpi_train_script", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def train(config: _config.TrainConfig) -> None:
    """openpi's scripts/train.py main(), copied with the memory's data loader (module docstring)."""
    trainer = openpi_trainer()
    trainer.init_logging()
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}.")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume)
    trainer.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    data_loader = data.create_data_loader(config, sharding=data_sharding, shuffle=True)
    train_state, train_state_sharding = trainer.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    train_step = jax.jit(functools.partial(trainer.train_step, config),
                         in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                         out_shardings=(train_state_sharding, replicated_sharding), donate_argnums=(1,))
    run_steps(config, train_step, train_rng, train_state, data_loader, checkpoint_manager, mesh)


def run_steps(config, train_step, train_rng, train_state, data_loader, checkpoint_manager, mesh) -> None:
    """The training loop of openpi's main() (copied): a step per batch, the logs every log_interval steps, a checkpoint every
    save_interval steps and at the last step."""
    data_iter = iter(data_loader)
    batch = next(data_iter)
    start_step = int(train_state.step)
    progress = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step, total=config.num_train_steps,
                         dynamic_ncols=True)
    infos = []
    for step in progress:
        with sharding.set_mesh(mesh):
            train_state, info = train_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            progress.write(f"Step {step}: " + ", ".join(f"{key}={value:.4f}" for key, value in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


def build_parser() -> argparse.ArgumentParser:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-name", required=True,
                        help="the run's name: its checkpoints go to <checkpoint-dir>/builtin_memory/<name>/<step>")
    parser.add_argument("--init-checkpoint", required=True, help="the checkpoint to continue (weights and norm stats)")
    parser.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("REPO_ID", "FEATURE_INDEX"),
                        help="a LeRobot dataset and its feature cache's index (repeatable; the first gives the task)")
    parser.add_argument("--phases", type=int, default=4, help="phases of the phase head (0: none; default 4)")
    parser.add_argument("--feature-stats", default=None, help=".npz with the features' mean and std, for a new memory")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-calls", type=int, default=96, help="the history cap (default 96 calls)")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--no-wandb", action="store_true", help="no logging to Weights & Biases")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--overwrite", action="store_true", help="replace an existing run of this name")
    restart.add_argument("--resume", action="store_true", help="continue an existing run of this name")
    return parser


def main(argv=None) -> None:
    """Build the TrainConfig from the command line and train."""
    args = build_parser().parse_args(argv)
    repo_ids, feature_indexes = zip(*args.dataset)
    config = train_config(args.exp_name, init_checkpoint=args.init_checkpoint, repo_ids=repo_ids,
                          feature_indexes=feature_indexes, phases=args.phases, feature_stats=args.feature_stats,
                          steps=args.steps, batch_size=args.batch_size, max_calls=args.max_calls,
                          checkpoint_base_dir=args.checkpoint_dir, save_interval=args.save_interval,
                          log_interval=args.log_interval, wandb_enabled=not args.no_wandb, overwrite=args.overwrite,
                          resume=args.resume)
    train(config)


if __name__ == "__main__":
    main()
