# Training

How the policies the memory methods run against were trained: the scripts that turn recorded demonstrations into training
data. Not needed to run the proxy. The policy itself is trained with its own code base: any VLA that trains on LeRobot
datasets works (the reference policies are pi0.5 fine-tuned with openpi), and what each method's policy must read is listed
in the README's "What the policy must accept".

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

## The datasets

The scripts write LeRobot datasets (v2.1) to a local folder, about 4 GB each. They are not part of this repository and not on
the Hugging Face Hub: the names below are the repo ids LeRobot files them under, in the folder given with `--root`. The
recordings they start from are not in the repository either: one `.npz` file per demonstration episode (the camera frames,
the joint state, the actions and the waypoint phase of every control step), written by the simulator bench that recorded
them; the `--raw-dir` defaults of `relabel_subgoals.py` and `relabel_history.py` name the recording folders of the reference
setup. Each dataset is produced from the recordings or from the datasets before it, and trains the policy of one method:

| dataset | trains the policy of | produced by | from |
|---|---|---|---|
| `vla_memory/red_blue_v2_abs_train` (and `_val`) | the memoryless policy (the full instruction) | `convert_to_lerobot.py --action-space absolute` (`--no-videos` when it feeds `relabel_memory.py` or `relabel_keyframe.py`, which read the frames as images) | the recordings; the plain instruction is every frame's task |
| `vla_memory/red_blue_v2_abs_subgoal_train` (and `_val`) | the sub-goal method | `relabel_subgoals.py` | the converted dataset and the recordings' waypoint phases |
| `vla_memory/red_blue_v2_abs_history_train` (and `_val`) | the history method | `relabel_history.py` | the converted dataset and the recordings' waypoint phases |
| `vla_memory/red_blue_v2_abs_memory_train` (and `_val`) | the memory policy; its control trains on the same frames without the memory | `relabel_memory.py`, which also writes the memory store | the sub-goal datasets: the demonstrations and the expert corrections, each given with its source label |
| `vla_memory/red_blue_v2_abs_keyframe_train` (and `_val`) | the keyframe method | `relabel_keyframe.py` | the converted dataset and the history dataset of the same frames |
| `vla_memory/red_blue_v2_recap_subgoal_train` | RECAP-lite, which continues the sub-goal policy | `convert_to_lerobot.py`, then `relabel_subgoals.py` | the kept rollouts of the sub-goal policy, mixed with demonstrations (`docs/methods/recap/README.md`) |

## How to train

Train the policy with the VLA's own training code on the dataset of its method. With openpi, which the reference policies
used: a training config whose `repo_id` is the dataset, `scripts/compute_norm_stats.py` once for that config, then
`scripts/train.py`. A checkpoint is served with `scripts/serve_policy.py`, and the proxy connects to that server's port.
RECAP-lite continues from the sub-goal policy's checkpoint instead of starting from `pi05_base`
(`docs/methods/recap/README.md`). No checkpoint is distributed with this repository; the policies of the measured results
were trained on these datasets.

## The reference policies

The reference policies are pi0.5 fine-tuned from `pi05_base` (RECAP-lite through its parent) with openpi's LoRA variants and
default ranks: rank 16 (alpha 16) on the 2B vision-language backbone (`gemma_2b_lora`), rank 32 (alpha 32) on the 300M action
expert (`gemma_300m_lora`), on the attention and feed-forward weights; only the LoRA matrices are trained, the base weights
stay frozen. A 30,000-step run at batch 16 takes about 38 hours on one RTX 3090. At inference the exterior image is
centre-cropped to 95%, the mean of openpi's 95% random-crop training augmentation.
