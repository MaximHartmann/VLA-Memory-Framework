# RECAP-lite in detail

RECAP-lite is a training recipe, not a part of the proxy. This page explains the recipe step by step: which parts the
framework provides, what each script reads and writes, and how the new policy is used behind the proxy. What RECAP-lite is
for and what we found are on the short page, [docs/methods/recap](../README.md).

![RECAP-lite: the policy's own successful runs become its training data](recap_loop.svg)

## The pieces

| step | done by | what it does |
|---|---|---|
| 1. roll out | the proxy (`--writer oracle --method subgoal`) and your environment | runs the parent with the ground-truth skill as its prompt; the environment records every step |
| 2. filter and cut | your environment | keeps the runs that completed the task, each cut 16 steps after the final placement |
| 3. convert | `training/convert_to_lerobot.py` | writes the kept runs as a LeRobot dataset |
| 4. relabel | `training/relabel_subgoals.py` | sets every frame's prompt to the skill that was active |
| 5. mix | your own script | adds a share of the original demonstrations |
| 6. fine-tune | the policy's own training code (openpi for the reference policy) | continues training from the parent |
| 7. serve | the policy server and the proxy | the new policy replaces the parent; nothing else changes |

![From recorded runs to the training dataset](recap_data.svg)

## 1. The rollouts

```
python -m vla_memory.proxy --writer oracle --method subgoal --task vla_memory/tasks/red_blue_blocks.yaml \
    --listen-port 8001 --upstream-port 8000
```

`--task` names the task file; for the reference task it can be left out. The policy server on port 8000 serves the parent.
This is the oracle writer with the sub-goal method: at every call the simulator reports its current skill
(`memory/oracle_prompt`), the oracle points at it (`OracleWriter.before_call`), and the method makes it the policy's prompt
(`SubgoalMethod.prompt`). The code path of the call is in the [oracle walkthrough](../../../writers/oracle/detailed/README.md)
and on the [sub-goal page](../../subgoal/detailed/README.md); RECAP-lite adds nothing to it.

The environment, not the proxy, records every step: one `.npz` file per episode, named `episode_*.npz` (both scripts below
read files of this name, in sorted order). The scripts read these arrays from it:

| array | what it holds | read by |
|---|---|---|
| `exterior`, `wrist` | both camera pictures of every control step (240 x 320 in the reference setup) | `convert_to_lerobot.py` |
| `state` | the robot's state of every step: the 7 arm joint angles and the gripper | both scripts |
| `actions` | the actions the policy executed: 7 absolute joint targets and the gripper command | `convert_to_lerobot.py` |
| `instruction` | the episode's instruction | `convert_to_lerobot.py` |
| `fps` | the frame rate (control steps per second) | `convert_to_lerobot.py --action-space absolute` |
| `phase`, or `plan` and `skill_index` | the active skill of every step | `relabel_subgoals.py` |

A demonstration stores the scripted expert's waypoint phase. A rollout has no waypoints, so it stores one representative
phase per skill, taken from the prompt the policy was given: 0 for "pick up the red block", 4 for "place the red block", 7
for "pick up the blue block", 12 for "place the blue block". Instead of phases, a recording may store the plan (`plan`: the
skills in order) and the plan position of every step (`skill_index`); [training/README.md](../../../../training/README.md)
describes this format.

## 2. Filter and cut

In simulation, whether a run worked is known for free: the environment scores the recorded movements. Only runs that
completed the whole task are kept; a **full task** is red picked up and put back, then blue picked up and put back. A run that
fails is thrown away whole. A kept run is cut 16 steps after the final placement, because the demonstrations also end soon
after the last release and retreat. The framework has no script for this step; it belongs to the environment.

## 3. Convert

```
python training/convert_to_lerobot.py --raw-dir <kept runs> --root <dataset root> --repo-id <user>/<name> \
    --action-space velocity --no-videos
```

The script (`main`, `add_episode`) reads every `episode_*.npz` of `--raw-dir` and writes a LeRobot dataset (the data format
the training code reads) at `--root`: per frame the two pictures (`image`, `wrist_image`), `state`, `actions`, and the
instruction as the frame's task. `--no-videos` stores the pictures as PNG files rather than videos. `--start N` and
`--limit N` convert only a part of the folder.

**The action space.** `--action-space velocity` stores the actions exactly as recorded. The name comes from the
demonstrations, which record joint velocities; for them `--action-space absolute` converts back to absolute joint targets
(`to_absolute`: target = measured position + velocity x time step). A rollout's actions are the policy's own absolute
targets already, so no conversion is wanted. The gripper command is never changed in either mode.

## 4. Relabel

```
python training/relabel_subgoals.py --src <converted dataset> --dst <relabelled dataset> --raw-dir <kept runs> --start 0 \
    [--task task.yaml]
```

Every frame's prompt becomes the skill that was active: the stored phase maps to a skill through the task file's
`demo_phases.phase_to_skill` (0 to 3 "pick up the red block", 4 to 6 "place the red block", 7 to 11 "pick up the blue block",
12 to 14 "place the blue block"). Before it touches an episode, the script checks that the dataset's state equals the
recording's exactly and that the episode goes through its skills in order; at the end it checks the metadata it wrote. Only
the prompts change; pictures, state and actions are copied. `--start 0` is right when the dataset's episodes are in the same
order as the recordings. The [sub-goal page](../../subgoal/detailed/README.md#2-how-the-policy-is-trained-for-it) explains the
script in more detail.

## 5. Mix with demonstrations

The relabelled demonstrations of some episodes are added to the kept runs, so that the policy keeps what it learned from them
(the reference round: 40 demonstrations, about 18 % of the episodes). The scripts in `training/` convert one folder in one
action mode, and the framework has no script that merges two datasets: the reference round converted the rollouts and the
demonstrations separately and merged them with a script of its own. For a single conversion instead, store the rollouts' arm
actions in the demonstrations' convention (the target minus the measured position, divided by the time step) and convert
everything with `--action-space absolute`; the conversion back is exact. We did not run it that way.

## 6. Fine-tune

The policy's own training code runs, starting from the parent's weights, with:

- the same LoRA settings as the parent (LoRA trains small added matrices instead of all weights);
- the parent's normalisation statistics (the means and spreads that scale the numbers going into the network), so that the
  loaded weights stay consistent;
- a short schedule and a low learning rate (the size of each training step). The reference round: 5,000 steps at batch size
  16, the learning rate rising over 200 warm-up steps to 1e-5 and falling to 1e-6 on a cosine curve (the parent's own recipe
  was 2.5e-5 to 2.5e-6 over 30,000 steps).

## 7. Choose a checkpoint and serve it

Try the saved checkpoints on development seeds with the ground-truth skills, and fix the rule for choosing before you look.
Ours: the best mean score of the task's milestones; scores within 0.25 of the best count as a tie, broken first by the number
of full tasks and then by the loss on held-out demonstrations (how far the policy's predicted actions are from those of
demonstrations it was not trained on).

Serve the chosen checkpoint and point the proxy at it:

```
python -m vla_memory.proxy --writer oracle --method subgoal --task task.yaml --listen-port <port> --upstream-port <new policy port>
```

Nothing else changes. The new policy is trained on the same skill prompts as its parent, so every writer, the sub-goal method,
the proxy's log and the extensions work as before.

## 8. Using it with your own policy and task

1. **A parent policy** that already succeeds often enough to produce good runs, and a way to run it with its prompts (the
   proxy, with any writer and method).
2. **Record every step** of the rollouts in the format of section 1: both pictures, state, executed actions, instruction, and
   the active skill (a phase per step and the task file's phase table, or `plan` and `skill_index`).
3. **Define success** for your task, keep only the runs that meet it, and cut them shortly after the final success.
4. **Convert and relabel** with `training/convert_to_lerobot.py` and the relabel script of your parent's method
   (`relabel_subgoals.py` for the sub-goal policy; `relabel_history.py` and `relabel_keyframe.py` exist for the other two
   methods, but only the sub-goal policy was done). The actions must mean the same in the demonstrations and in the rollouts.
5. **Mix in demonstrations** (about a fifth of the episodes) and fine-tune from the parent with a short schedule and a low
   learning rate; keep the parent's normalisation statistics.
6. **Choose** a checkpoint on development seeds, then run fresh seeds and compare with the parent seed by seed. The new
   policy is served behind the same proxy.

## Limits to keep in mind

- **Successes only.** The policy gets no example of what to do in the states that led it astray.
- **Little to gain from a strong parent.** The method can only copy what the parent already does well; a policy that fails
  more often has more to gain, but must still succeed often enough to produce good runs.
- **It inherits its parent's weaknesses.** Rollouts recorded with one camera position cannot teach the policy another one.
- **Forgetting.** The demonstrations in the mix are meant to prevent it; skills outside the data may still be forgotten.
- **A noisy choice of checkpoint.** On a few development seeds several checkpoints score almost the same, so the tie-break
  decides.
- **One round, one policy, one task, simulation.** More rounds, other tasks, other policies and a real robot were not tried.

## Where the idea comes from

- **pi*0.6 with RECAP** (Physical Intelligence, 2025, arXiv 2511.14759) is the source. The robot's own runs and expert
  corrections train a value function, every step gets a label that says whether it went better than expected, the policy is
  fine-tuned on all the data and, at run time, asked for the "good" behaviour. The paper reports more than twice the
  throughput and about half the failure rate on the hardest tasks. RECAP-lite keeps only the first of its ingredients:

  | ingredient of RECAP | RECAP-lite |
  |---|---|
  | the robot's own runs | kept: the parent's rollouts with ground-truth skills |
  | corrections by an expert | dropped |
  | a value function trained on the runs | dropped |
  | a label for every step, as text in the prompt | dropped: the runs are kept or thrown away whole |
  | several rounds | dropped: one round |
  | training on diverse data first | RECAP's first stage is plain behaviour cloning on demonstrations: that is the parent |

- **Self-Demonstrated Generative Control** (Garg et al., 2026, arXiv 2608.19490) uses zero-shot rollouts of pi0.5 as extra
  training data, a close relative of this recipe.
- **Reinforcement learning on VLAs** learns from a reward, not only from the successes: SimpleVLA-RL (Li et al., 2025, arXiv
  2509.09674), pi-RL (Chen et al., 2025, arXiv 2510.25889: an open recipe for pi0 and pi0.5), ConRFT (arXiv 2502.05450),
  iRe-VLA (arXiv 2501.16664), VLA-RL (arXiv 2505.18719). We did not try them.
- **A human as the filter:** HG-DAgger (arXiv 1810.02890), Sirius (arXiv 2211.08416) and RoboCat (arXiv 2306.11706) run the
  same loop with a human who corrects. In simulation a scripted expert can play that part
  ([learning from interventions](../../learning_from_interventions/README.md)).
- **Forgetting and replay.** Fine-tuning on new data can erase what a policy knew. Zhu et al. (2026, arXiv 2605.26820)
  fine-tuned pi0.5 on real arms one task after another: plain fine-tuning forgot much of the earlier tasks, while replaying a
  buffer of 20 % of the old data kept them. Liu et al. (2026, arXiv 2603.03818) found pi0 more resistant to forgetting than
  expected, and a 20 % replay buffer removed its forgetting. Our mix has about 18 % demonstrations.
