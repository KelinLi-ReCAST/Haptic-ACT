"""Config of the F/T Haptic-ACT policy (`--policy.type=acthaptic`).

It inherits every field of LeRobot's ACTConfig. The force input is whatever the dataset provides under
`observation.environment_state` (here a 30-dim window of the last 5 TCP wrenches), which stock ACT already turns
into a dedicated encoder token. Added on top: haptic dropout, a learned re-planning head and an optional
contact-phase head."""
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("acthaptic")
@dataclass
class ACTHapticConfig(ACTConfig):
    # Probability (per sample, training only) of replacing the force token input with zeros (= the normalized
    # mean), so the policy cannot rely on it being present and the visual path keeps learning the task. At test
    # time the full signal is always given.
    haptic_dropout: float = 0.3
    # Weight of an auxiliary cross-entropy that predicts the demonstration phase (the dataset's `phase` column)
    # from the force token's encoder output. 0 disables the head; it was off in every reported run.
    aux_phase_weight: float = 0.0
    n_phases: int = 7
    # Learned re-planning head: from the force token's encoder output, predict the dataset's per-frame `replan`
    # label. At rollout it decides when the policy abandons its current action chunk (eval_ft.py).
    replan_weight: float = 1.0
    replan_pos_weight: float = 20.0  # positives are ~2% of the frames
