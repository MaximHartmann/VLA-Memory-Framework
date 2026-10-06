import openpi.policies.red_blue_policy as red_blue_policy
@dataclasses.dataclass(frozen=True)
class LeRobotRedBlueDataConfig(DataConfigFactory):
    """Sequential red-then-blue pick-and-place, generated in Isaac Sim.

    Shape matches LIBERO (8-D state, base + wrist camera), so the repack and the
    overall structure follow LeRobotLiberoDataConfig. The one substantive
    departure is the delta transform below, and it is not optional.
    """

    # For the ABSOLUTE-target datasets (*_abs) only: train on joint targets
    # relative to the current state, converted back at inference, over the 7 arm
    # joints; the gripper (index 7) stays absolute. This is openpi's own recipe
    # for absolute Franka joint positions (RLDSDroidDataConfig, JOINT_POSITION;
    # pi05_full_droid_finetune). NEVER set it for the velocity datasets: those
    # are already relative to the state, see create().
    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Key names here are the ones convert_to_lerobot.py writes, mapped onto
        # the keys the sim's policy client sends at inference.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[red_blue_policy.RedBlueInputs(model_type=model_config.model_type)],
            outputs=[red_blue_policy.RedBlueOutputs()],
        )

        # NO DELTA TRANSFORM. generate_demos.py:301 records
        #     act_arm = (q_cmd_arm - q_meas[arm_idx]) / dt
        # so `actions` are already joint VELOCITIES -- a difference from the
        # measured state, scaled by 1/dt -- not absolute joint targets. Applying
        # _transforms.DeltaActions here subtracts the state a SECOND time.
        #
        # That mistake is invisible in any shape check and does not raise: it
        # shows up only in the statistics, where the action mean comes out as
        # almost exactly -1 x the state mean (mean |action| 1.05 instead of
        # 0.05), because subtracting state from something already centred on
        # zero just yields -state. Caught by reading norm_stats.json before the
        # first training run.
        #
        # This matches LIBERO after all, whose actions are likewise already
        # deltas and whose config therefore sets extra_delta_transform=False.
        # The flag below is for ABSOLUTE targets, where the state has not been
        # subtracted yet -- see the field comment.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


    # Gradient accumulation: each step splits the batch into this many micro-batches, runs them one
    # after another and applies ONE optimizer update with the mean gradient. The effective batch,
    # step count and LR schedule are unchanged; only one micro-batch's activations are in memory.
    grad_accum_steps: int = 1
        if self.grad_accum_steps < 1 or self.batch_size % self.grad_accum_steps:
            raise ValueError(
                f"grad_accum_steps ({self.grad_accum_steps}) must be >= 1 and divide batch_size ({self.batch_size})."
            )
def _red_blue_abs_ablation(name: str, repo_id: str, *, state: bool = True, delta: bool = True) -> TrainConfig:
    """One arm of the subgoal-vs-monolith ablation (2026-09-21).

    Both arms train on ABSOLUTE joint targets with the two fixes the
    conditioning diagnostics called for, and differ ONLY in the dataset's
    prompts: per-frame subgoals (red_blue_blocks_abs_subgoal_train, written by
    scripts/relabel_subgoals.py) or the static full instruction
    (red_blue_blocks_abs_train). One function builds both, so they cannot drift.

      discrete_state_input=True   pi05 reads the joint state ONLY as prompt
          tokens. The earlier red_blue configs copied pi05_libero's False, so the
          model never saw its own joints: at red_blue_abs step 4000 its first
          target missed the arm by 0.014 rad median, 0.31 rad max -- a step
          command at every chunk boundary.
      extra_delta_transform=True  targets relative to the current state
          (openpi's recipe for absolute Franka joint positions).

    Everything else is pi05_red_blue_lora_abs, whose run trained healthily (loss
    0.109 -> 0.011 by step 4000, conditioning intact), unlike the velocity run,
    which lost all conditioning between steps 20 and 30. Norm stats are identical
    for both arms (prompts are not part of them): compute once, copy.

    `state`/`delta` switch the two fixes off one at a time. Both arms of the
    ablation collapsed at step 20-30 exactly like the velocity run, while
    pi05_red_blue_lora_abs (neither fix) did not, so the *_stateonly and
    *_deltaonly configs isolate which of the two triggers it.
    """
    lora = {"paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora"}
    return TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=state, **lora),
        data=LeRobotRedBlueDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=delta,
        ),
        batch_size=16,
        num_workers=0,  # spawn + cv2 kills data-loader workers in this container (no libGL)
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(pi05=True, **lora).get_freeze_filter(),
        ema_decay=None,
        fsdp_devices=4,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    )


    # v2 data (2026-09-25): physical grasp with fer_drive2s.usd, camera preset v2, close command 0.60. Same model
    # settings as the v1 ablation pair (absolute targets, no state tokens, no DeltaActions); new names so the asset
    # dirs get their own norm stats.
    _red_blue_abs_ablation("pi05_red_blue_v2_subgoal", "hartmann/red_blue_v2_abs_subgoal_train", state=False, delta=False),
    _red_blue_abs_ablation("pi05_red_blue_v2_monolith", "hartmann/red_blue_v2_abs_train", state=False, delta=False),
    # v2 history arm (2026-09-27): monolith instruction + " History: <events so far>" per frame (scripts/relabel_history.py),
    # the UniMem-style textual event memory inside the VLA prompt; same images/state/actions as the monolith data.
    _red_blue_abs_ablation("pi05_red_blue_v2_history", "hartmann/red_blue_v2_abs_history_train", state=False, delta=False),
    # RECAP-lite (2026-09-28): continue the SELECTED subgoal checkpoint (20000) on the agent's own successful closed-loop
    # rollouts (seeds 2001-2200, oracle prompts) mixed with 40 original demos; same LoRA config and norm stats as the
    # parent so the loaded weights stay consistent; short schedule with a lower peak LR.
    dataclasses.replace(
        _red_blue_abs_ablation("pi05_red_blue_v2_subgoal_recap", "hartmann/red_blue_v2_recap_subgoal_train", state=False, delta=False),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/mhartmann/vla-runs/checkpoints/pi05_red_blue_v2_subgoal/v2_1gpu/20000/params"),
        num_train_steps=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=1.0e-5, decay_steps=5_000, decay_lr=1.0e-6),
    ),
    # Full RECAP round 1 (2026-10-04): the RECAP-lite config above on the round-1 data -- the 200 rollouts of 20000 INCLUDING
    # the failures, 200 rollouts with scripted-expert hand-over, 100 nudged rollouts with hand-over, and the same 40 demos --
    # whose prompts carry the advantage indicator ("<sub-goal>. Advantage: good|bad" from the value function; expert and
    # demo frames good; 25 % of the chunks keep the plain sub-goal). Same LoRA, norm stats, parent weights (20000) and
    # schedule (5000 steps, cosine 1e-5 -> 1e-6, 200 warm-up) as RECAP-lite, so the comparison isolates what RECAP adds.
    dataclasses.replace(
        _red_blue_abs_ablation("pi05_red_blue_v2_subgoal_recapfull", "hartmann/red_blue_v2_recapfull_r1_train", state=False, delta=False),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/mhartmann/vla-runs/checkpoints/pi05_red_blue_v2_subgoal/v2_1gpu/20000/params"),
        num_train_steps=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=1.0e-5, decay_steps=5_000, decay_lr=1.0e-6),
    ),
