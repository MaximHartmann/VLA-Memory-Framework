# Training

How the policies the memory methods run against were trained: the scripts that turn recorded demonstrations into training
data, and our changes to openpi (commit 215abfb). Not needed to run the proxy.

## Files

| file | role |
|---|---|
| `convert_to_lerobot.py` | recorded episodes (`.npz` with images, joint state, actions, waypoint phase) to a LeRobot dataset in openpi's LIBERO-like schema (`image`, `wrist_image`, `state`, `actions`, `prompt`) |
| `relabel_subgoals.py` | the sub-goal policy's data: each frame's prompt becomes the active skill ("pick up the red block", ...), by waypoint phase |
| `relabel_history.py` | the history policy's data: each frame's prompt becomes the full instruction plus "History: <events so far>" |
| `relabel_memory.py` | the memory policy's data: the memory store built from the training datasets, and each frame's retrieved memory (the image column `memory_image`, the prompt "<skill>. memory: <sentence>") |
| `relabel_keyframe.py` | the keyframe policy's data: each frame with the wrist picture of the episode's last key moment (the image column `memory_image`) |
| `lerobot_tools.py` | the helpers the relabelling scripts share: the metadata files of a dataset, its parquet episode files, a new dataset with a `memory_image` column, the per-frame manifests |
| `fit_event_head.py` | the event head of the history policy (`docs/writers/head/README.md`): training rows from recorded rollouts of the policy, their features from the frozen policy, the fit on the CPU |
| `openpi/openpi_changes.patch` | our changes to openpi: the input and output transforms (`src/openpi/policies/red_blue_policy.py`), the data and training configs (`src/openpi/training/config.py`), a weight guard and optional gradient accumulation in `scripts/train.py`, and the inference-time centre crop (`serve_policy.py --policy.center_crop`) |

## The datasets

The scripts write LeRobot datasets (v2.1) to a local folder, about 4 GB each. They are not part of this repository and not on
the Hugging Face Hub: the names below are the repo ids LeRobot files them under, in the folder given with `--root`. The
recordings they start from are not in the repository either: one `.npz` file per demonstration episode (the camera frames,
the joint state, the actions and the waypoint phase of every control step), written by the simulator bench that recorded
them; the `--raw-dir` defaults of `relabel_subgoals.py` and `relabel_history.py` name the recording folders of the reference
setup. Each dataset is produced from the recordings or from the datasets before it:

| dataset | produced by | from |
|---|---|---|
| `vla_memory/red_blue_v2_abs_train` (and `_val`) | `convert_to_lerobot.py --action-space absolute` (`--no-videos` when it feeds `relabel_memory.py` or `relabel_keyframe.py`, which read the frames as images) | the recordings; the plain instruction is every frame's task |
| `vla_memory/red_blue_v2_abs_subgoal_train` (and `_val`) | `relabel_subgoals.py` | the converted dataset and the recordings' waypoint phases |
| `vla_memory/red_blue_v2_abs_history_train` (and `_val`) | `relabel_history.py` | the converted dataset and the recordings' waypoint phases |
| `vla_memory/red_blue_v2_abs_memory_train` (and `_val`) | `relabel_memory.py`, which also writes the memory store | the sub-goal datasets: the demonstrations and the expert corrections, each given with its source label |
| `vla_memory/red_blue_v2_abs_keyframe_train` (and `_val`) | `relabel_keyframe.py` | the converted dataset and the history dataset of the same frames |
| `vla_memory/red_blue_v2_recap_subgoal_train` | `convert_to_lerobot.py`, then `relabel_subgoals.py` | the kept rollouts of the sub-goal policy, mixed with demonstrations (`docs/methods/recap/README.md`) |

## How to train

1. Apply our changes to an openpi checkout at commit 215abfb:

   ```
   git -C <openpi checkout> apply <this repository>/training/openpi/openpi_changes.patch
   ```

2. Write the training dataset with the scripts above, under the repo id its config names (the table below), and compute
   its normalisation statistics once, in the openpi checkout:

   ```
   uv run scripts/compute_norm_stats.py --config-name pi05_red_blue_v2_subgoal
   ```

3. Train with openpi's own command:

   ```
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_red_blue_v2_subgoal --exp-name=<name> --overwrite
   ```

The dataset names are the `repo_id`s of the configs in our openpi patch, under the namespace `vla_memory/` (the default of
`convert_to_lerobot.py --repo-id`); a dataset produced under another name needs the config's `repo_id` changed to match.

| config | policy | dataset (local, see above) |
|---|---|---|
| `pi05_red_blue_v2_subgoal` | the sub-goal method's | `vla_memory/red_blue_v2_abs_subgoal_train` |
| `pi05_red_blue_v2_monolith` | the memoryless policy (the full instruction) | `vla_memory/red_blue_v2_abs_train` |
| `pi05_red_blue_v2_history` | the history method's | `vla_memory/red_blue_v2_abs_history_train` |
| `pi05_red_blue_v2_subgoal_recap` | RECAP-lite | `vla_memory/red_blue_v2_recap_subgoal_train` |
| `pi05_red_blue_v2_memory`, `pi05_red_blue_v2_memory_control` | the memory policy and its control | `vla_memory/red_blue_v2_abs_memory_train` |
| `pi05_red_blue_v2_keyframe` | the keyframe method's | `vla_memory/red_blue_v2_abs_keyframe_train` |

The RECAP configs continue from the sub-goal policy's checkpoint at
`$RUNS/checkpoints/pi05_red_blue_v2_subgoal/v2_1gpu/20000/params`, the folder openpi's training wrote for that run
(`checkpoints/<config>/<experiment name>/<step>/params`): set `RUNS` to the folder whose `checkpoints/` holds it. No
checkpoint is distributed with this repository; the policies of the measured results come from these configs and this
data. A checkpoint is served with openpi's
`uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint folder>` (with
`--policy.center_crop 0.95` for the centre crop), and the proxy connects to that server's port.

## LoRA settings

Every config fine-tunes pi0.5 from `pi05_base` with openpi's LoRA variants and default ranks: rank 16 (alpha 16) on the 2B
vision-language backbone (`gemma_2b_lora`), rank 32 (alpha 32) on the 300M action expert (`gemma_300m_lora`), on the
attention and feed-forward weights; only the LoRA matrices are trained, the base weights stay frozen. A 30,000-step run at
batch 16 takes about 38 hours on one RTX 3090.
