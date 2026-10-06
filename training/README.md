# Training

The scripts that produced the policies the memory methods run against: the data pipeline from recorded demonstrations to a
training set, our additions to openpi, and the launcher that ran the fine-tuning. Kept as used; not needed to run the proxies.

## Files

| file | role |
|---|---|
| `convert_to_lerobot.py` | recorded episodes (`.npz` with images, joint state, actions, waypoint phase) to a LeRobot dataset in openpi's LIBERO-like schema (`image`, `wrist_image`, `state`, `actions`, `prompt`) |
| `relabel_subgoals.py` | rewrites the dataset's per-frame prompt to the active skill ("pick up the red block", ...), by waypoint phase, with an integrity check against the recordings; the data of the sub-goal policy |
| `relabel_history.py` | the same for the history prompt: the full instruction plus "History: <events so far>", at the same phase boundaries; the data of the history policy |
| `train_avalon.sh` | the launcher: runs openpi's `scripts/train.py` for one config inside the container, pins the GPU and its NUMA node, checks the norm statistics, logs VRAM, and reports the exit (`finished OK`, `WEIGHT GUARD ABORT`, `FAILED`) |
| `train_v2_avalon.sh` | runs the sub-goal and monolith arms as a pair (one GPU each, or FSDP on eight) |
| `openpi/red_blue_policy.py` | our input/output transforms for openpi: the 8-D joint-space state and action (7 joints plus the gripper), the two cameras, the prompt |
| `openpi/config_red_blue_additions.py` | the lines we added to openpi's `src/openpi/training/config.py`: the data config (`LeRobotRedBlueDataConfig`), the config factory `_red_blue_abs_ablation` and the four configs `pi05_red_blue_v2_{subgoal,monolith,history,subgoal_recap}`; an extract for reading, not a module |
| `openpi/openpi_changes.patch` | the complete diff of our changes to openpi (commit 215abfb, Apache 2.0): the config additions, the policy transforms, a weight guard in `train.py` that aborts on non-finite or exploding parameters (exit code 3), optional gradient accumulation, a switch for buffer donation, and the inference-time centre crop (`transforms.py`, `serve_policy.py --policy.center_crop`) |

The demonstration generator (`generate_demos.py`, Isaac Sim) and the host paths in the launcher (`/data/mhartmann/...`, the
Singularity image) belong to the research code.

## LoRA settings

The four configs fine-tune pi0.5 with openpi's LoRA variants of its two transformers, with openpi's default ranks: the
2B vision-language backbone (`paligemma_variant="gemma_2b_lora"`) gets rank 16, alpha 16, on the attention and feed-forward
weights; the 300M action expert (`action_expert_variant="gemma_300m_lora"`) rank 32, alpha 32. Only the LoRA matrices are
trained (`freeze_filter` of the same config); the base weights of `pi05_base` stay frozen. The ranks are set in openpi's
`src/openpi/models/gemma.py` (`get_config`), not in our files; `_red_blue_abs_ablation` in `config_red_blue_additions.py` only
selects the variants.

## Running a fine-tuning

One policy is one run of the launcher:

```
GPU=0 CONFIG=pi05_red_blue_v2_subgoal EXP=v2_1gpu setsid nohup training/train_avalon.sh >/dev/null 2>&1 &
```

The three variables are required: `GPU` (one index, or a comma-separated list for several GPUs), `CONFIG` (one of the four
openpi configs above; it names the model settings and the dataset), `EXP` (a name for this run; the checkpoints and the log are
filed under it). `setsid nohup ... &` detaches the run so it survives a logout; a 30,000-step run takes about 38 hours on one
RTX 3090.

What the launcher does, in order:

1. Opens the log `$RUNS/logs/<CONFIG>__<EXP>.log` (`RUNS` defaults to `/data/mhartmann/vla-runs`) and writes everything there.
2. Looks up the dataset of the config and checks that its norm statistics exist at `openpi/assets/<CONFIG>/<repo>/norm_stats.json`.
   Without them openpi would train on unnormalised actions, so the run stops here if they are missing. They are computed once per
   dataset with openpi's `scripts/compute_norm_stats.py <CONFIG>`.
3. Records the code version (openpi commit, md5 of `train.py` and `config.py`) and refuses to start if the GPU already has a
   process on it.
4. Pins the process to the GPU's NUMA node and starts a background logger that writes the GPU's memory use every 5 seconds.
5. Starts openpi's `scripts/train.py <CONFIG> --exp-name <EXP>` inside the Singularity container, with the environment the
   runs used: the openpi source and the LeRobot data folder mounted, 90 % of the GPU memory preallocated, buffer donation and
   the weight guard on (`OPENPI_DONATE`, `OPENPI_WEIGHT_GUARD`), wandb in offline mode.
6. When training ends, prints the peak GPU memory and the final line: `finished OK`, `WEIGHT GUARD ABORT` (the guard found a
   non-finite or exploding parameter; exit code 3) or `train.py FAILED`.

Optional variables: `RESUME=1` continues a run from its last checkpoint instead of starting over (`--resume` instead of
`--overwrite`); `EXTRA_ARGS` passes more flags to `train.py`, e.g. `--num-train-steps 5000 --save-interval 1000 --keep-period 1000`
(the RECAP-lite schedule) or `--seed 7`; `FSDP=8` shards the model over eight GPUs; `GRAD_ACCUM=2` splits each batch into two
micro-batches for GPUs with less memory. `train_v2_avalon.sh` wraps all this for the sub-goal and monolith pair.

The result is a folder per checkpoint, `$RUNS/checkpoints/<CONFIG>/<EXP>/<step>/`, every `--save-interval` steps. A checkpoint
is served to the proxies with openpi's `scripts/serve_policy.py policy:checkpoint --policy.config=<CONFIG> --policy.dir=<that folder>`
(with `--policy.center_crop 0.95` for the centre crop); the proxies then connect to that server's port.
