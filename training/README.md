# Training

This folder holds the scripts that turn recorded demonstrations into training data for the policies the memory methods run
against, and the script that fits the head writer's event head. You do not need it to run the proxy. The policy itself is
trained with its own code base: any VLA that trains on LeRobot datasets works (the reference policies are π0.5 fine-tuned
with openpi). What each method's policy must read is listed in the main README, "What the policy must accept".

## Files

| file | role |
|---|---|
| `convert_to_lerobot.py` | recorded episodes (`.npz` with images, joint state, actions, waypoint phase) to a LeRobot dataset in openpi's LIBERO-like schema (`image`, `wrist_image`, `state`, `actions`, `prompt`) |
| `relabel_subgoals.py` | the sub-goal policy's data: each frame's prompt becomes the active skill ("pick up the red block", ...), by waypoint phase (the task file's `demo_phases`) |
| `relabel_history.py` | the history policy's data: each frame's prompt becomes the full instruction plus "History: <events so far>", the text the task file's `history` section renders |
| `relabel_memory.py` | the memory policy's data: the memory store built from the training datasets, and each frame's retrieved memory (the image column `memory_image`, the prompt "<skill>. memory: <sentence>") |
| `relabel_keyframe.py` | the keyframe policy's data: each frame with the wrist picture of the episode's last key moment (the image column `memory_image`) |
| `lerobot_tools.py` | the helpers the relabelling scripts share: the metadata files of a dataset, its parquet episode files, a new dataset with a `memory_image` column, the per-frame manifests |
| `fit_event_head.py` | the event head of a policy (`docs/writers/head/README.md`): fitted on the CPU on the proxy's recordings of the policy's own runs (`--record-head`), or, for openpi's π0.5, on recorded pictures whose features the frozen policy computes |

`convert_to_lerobot.py` needs LeRobot (openpi's environment has it); the relabelling scripts need numpy and pyarrow. The
scripts find the framework by themselves, also when it is not installed. Each script's `--help` lists its options.

## The datasets

The scripts write LeRobot datasets (format v2.1), about 4 GB each, each into a folder of its own:
`convert_to_lerobot.py` into the folder `--root`, `relabel_subgoals.py` and `relabel_history.py` from the folder `--src`
into a new folder `--dst`, and `relabel_memory.py` and `relabel_keyframe.py` into `<root>/<repo-id>_train` and `_val`.
The datasets are not part of this repository and not on the Hugging Face Hub; the names below are their repository ids
(LeRobot's names for them). The recordings they start from are not in the repository either: one `.npz` file per
demonstration episode (the camera frames, the joint state, the actions and the waypoint phase of every control step),
written by the simulator bench that recorded them. `convert_to_lerobot.py`, `relabel_subgoals.py` and `relabel_history.py`
need their folder, given with `--raw-dir`.

Each dataset is made from the recordings or from the datasets before it, and trains the policy of one method:

| dataset | trains the policy of | produced by | from |
|---|---|---|---|
| `vla_memory/red_blue_v2_abs_train` (and `_val`) | the memoryless policy (the full instruction) | `convert_to_lerobot.py --action-space absolute` (`--no-videos` when it feeds `relabel_memory.py` or `relabel_keyframe.py`, which read the frames as images) | the recordings; the plain instruction is every frame's task |
| `vla_memory/red_blue_v2_abs_subgoal_train` (and `_val`) | the sub-goal method | `relabel_subgoals.py` | the converted dataset and the recordings' waypoint phases |
| `vla_memory/red_blue_v2_abs_history_train` (and `_val`) | the history method | `relabel_history.py` | the converted dataset and the recordings' waypoint phases |
| `vla_memory/red_blue_v2_abs_memory_train` (and `_val`) | the memory policy; its control trains on the same frames without the memory | `relabel_memory.py`, which also writes the memory store | the sub-goal datasets: the demonstrations and the expert corrections, each given with its source label |
| `vla_memory/red_blue_v2_abs_keyframe_train` (and `_val`) | the keyframe method | `relabel_keyframe.py` | the converted dataset and the history dataset of the same frames |
| `vla_memory/red_blue_v2_recap_subgoal_train` | RECAP-lite, which continues the sub-goal policy | `convert_to_lerobot.py`, then `relabel_subgoals.py` | the kept rollouts of the sub-goal policy, mixed with demonstrations (`docs/methods/recap/README.md`) |

For example, the sub-goal policy's training data:

```
python training/convert_to_lerobot.py --raw-dir <recordings> --root <datasets>/red_blue_v2_abs_train \
    --repo-id vla_memory/red_blue_v2_abs_train --action-space absolute
python training/relabel_subgoals.py --src <datasets>/red_blue_v2_abs_train \
    --dst <datasets>/red_blue_v2_abs_subgoal_train --raw-dir <recordings>
```

`--start` and `--limit` (convert) and `--start` (relabel) cut the recordings into a training and a validation split.

## A task whose plan varies

The skills, the phase table and the history phrases come from the task file (`--task`, by default the reference task).
Some tasks have a plan that differs from episode to episode, because their objects vary or their order is free; a fixed
table from phases to skills cannot work for them. Each of their recordings then holds, besides the camera frames, the state
and the actions:

- `plan`: the episode's skills as strings, all of them skills of the task;
- `skill_index`: the position in that plan of the skill active at every frame, instead of `phase`.

`relabel_subgoals.py` and `relabel_history.py` read these: a frame's prompt is the skill at its position, or the
instruction with the skills before it (at most `history.max_lines`), and `tasks.jsonl` lists all skills of the task. The
event head's picture episodes (`fit_event_head.py --episodes`) may likewise hold `plan` or `objects`. `relabel_keyframe.py`
matches the history prompts against the task order, so it covers only tasks whose plan is the task order.

## How to train

Train the policy with the VLA's own training code on the dataset of its method. With openpi, which the reference policies
used: write a training config whose `repo_id` is the dataset, run openpi's `scripts/compute_norm_stats.py` once for that
config, then openpi's `scripts/train.py`. Serve a checkpoint with openpi's `scripts/serve_policy.py` (or with
`integrations/openpi/openpi_server.py`, which also hands out the policy's state for the head writer), and give that
server's port to the proxy (`--upstream-port`). RECAP-lite continues from the sub-goal policy's checkpoint instead of
starting from `pi05_base` (`docs/methods/recap/README.md`). No checkpoint comes with this repository; the policies of our
measured results were trained on these datasets.

The event head is fitted after the policy is trained, on the proxy's recordings of the policy's own runs
([the head writer](../docs/writers/head/README.md)):

```
python training/fit_event_head.py --recordings 'runs/*.npz' --out head/
```

## The reference policies

The reference policies are π0.5 fine-tuned from `pi05_base` (RECAP-lite through its parent) with openpi's LoRA variants
and default ranks: rank 16 (alpha 16) on the 2B vision-language backbone (`gemma_2b_lora`), rank 32 (alpha 32) on the 300M
action expert (`gemma_300m_lora`), on the attention and feed-forward weights. Only the LoRA matrices (small extra weight
matrices) are trained; the base weights stay frozen. A 30,000-step run at batch 16 takes about 38 hours on one RTX 3090. At
inference the exterior image is centre-cropped to 95 %, the mean of openpi's 95 % random-crop training augmentation.
