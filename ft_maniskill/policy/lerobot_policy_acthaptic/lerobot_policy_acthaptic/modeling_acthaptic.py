"""F/T Haptic-ACT policy: LeRobot's ACT whose `observation.environment_state` token carries the F/T window, plus
(1) haptic dropout on that token's input during training, (2) a learned re-planning head on the token's encoder
output (it predicts the dataset's `replan` label and is used at rollout to abandon the current action chunk) and
(3) an optional phase head.

The token order in ACT's encoder is [latent, robot_state, env_state, image patches...], so with a robot state
present the force token is index 2 of the encoder output. The encoder output is captured with a forward hook
instead of re-implementing ACT.forward, so the network is exactly LeRobot's ACT."""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES

from .configuration_acthaptic import ACTHapticConfig

# The labels reach forward() under the observation namespace (train_acthaptic.py re-exposes them there, because
# the policy processor pipeline drops every other dataset column). The bare column names are accepted as a
# fallback for callers that bypass the pipeline.
PHASE_KEY = "observation.phase"
REPLAN_KEY = "observation.replan"
_PHASE_FALLBACK = "phase"
_REPLAN_FALLBACK = "replan"


def _label(batch: dict, key: str, fallback: str) -> Tensor | None:
    return batch.get(key, batch.get(fallback))


class ACTHapticPolicy(ACTPolicy):
    config_class = ACTHapticConfig
    name = "acthaptic"

    def __init__(self, config: ACTHapticConfig, **kwargs):
        super().__init__(config, **kwargs)
        if not config.env_state_feature:
            raise ValueError("F/T Haptic-ACT needs `observation.environment_state` (the F/T window) among the input features")
        self.aux_phase_head = nn.Linear(config.dim_model, config.n_phases) if config.aux_phase_weight > 0 else None
        self.replan_head = nn.Linear(config.dim_model, 1) if config.replan_weight > 0 else None
        self._encoder_out: Tensor | None = None
        self.model.encoder.register_forward_hook(lambda _m, _i, out: setattr(self, "_encoder_out", out))
        self._haptic_token_index = 2 if config.robot_state_feature else 1

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = dict(batch)
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        if self.training and self.config.haptic_dropout > 0:
            keep = torch.rand(batch[OBS_ENV_STATE].shape[0], 1, device=batch[OBS_ENV_STATE].device) >= self.config.haptic_dropout
            batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE] * keep.to(batch[OBS_ENV_STATE].dtype)

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # L1 over the non-padded action steps only (mean over valid entries), plus ACT's KL term.
        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss
        if self.config.use_vae and log_sigma_x2_hat is not None:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight

        replan_target = _label(batch, REPLAN_KEY, _REPLAN_FALLBACK)
        if self.replan_head is not None and self.training and replan_target is None:
            raise KeyError(
                f"replan_weight={self.config.replan_weight} but neither {REPLAN_KEY!r} nor {_REPLAN_FALLBACK!r} "
                f"is in the batch (keys: {sorted(batch)}): the re-planning head would train on nothing"
            )
        if self.replan_head is not None and replan_target is not None:
            logit = self.replan_head(self._encoder_out[self._haptic_token_index]).reshape(-1)
            target = replan_target.reshape(-1).float()
            pos_weight = torch.tensor(self.config.replan_pos_weight, device=logit.device)
            replan_loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
            loss_dict["replan_loss"] = replan_loss.item()
            with torch.no_grad():
                pred = logit > 0
                loss_dict["replan_recall"] = pred[target > 0.5].float().mean().item() if (target > 0.5).any() else float("nan")
                loss_dict["replan_fpr"] = pred[target < 0.5].float().mean().item() if (target < 0.5).any() else float("nan")
            loss = loss + self.config.replan_weight * replan_loss

        phase_target = _label(batch, PHASE_KEY, _PHASE_FALLBACK)
        if self.aux_phase_head is not None and self.training and phase_target is None:
            raise KeyError(f"aux_phase_weight > 0 but no {PHASE_KEY!r} / {_PHASE_FALLBACK!r} label in the batch")
        if self.aux_phase_head is not None and phase_target is not None:
            logits = self.aux_phase_head(self._encoder_out[self._haptic_token_index])
            target = phase_target.reshape(-1).long()
            aux_loss = F.cross_entropy(logits, target)
            loss_dict["aux_phase_loss"] = aux_loss.item()
            loss_dict["aux_phase_acc"] = (logits.argmax(-1) == target).float().mean().item()
            loss = loss + self.config.aux_phase_weight * aux_loss
        return loss, loss_dict

    @torch.no_grad()
    def replan_probability(self) -> Tensor:
        """P(re-plan now) for the batch of the most recent forward / predict_action_chunk call, from the force
        token's encoder output. Call it after predict_action_chunk on the current observation."""
        if self.replan_head is None or self._encoder_out is None:
            raise RuntimeError("no re-planning head, or no forward pass yet")
        return torch.sigmoid(self.replan_head(self._encoder_out[self._haptic_token_index]).reshape(-1))
