"""lerobot-train for F/T Haptic-ACT (`--policy.type=acthaptic`), with the re-planning label passed to the policy.

LeRobot's policy processor pipeline keeps only `observation.*`, `action`, `*_is_pad` and a fixed set of metadata keys,
so every other dataset column, including the `replan` (and `phase`) supervision labels, is dropped before
policy.forward() sees the batch. This entry point re-exposes the labels under the observation namespace, which the
pipeline passes through untouched (the normalizer ignores keys that are not policy features). Every lerobot-train
flag works as usual."""
import lerobot_policy_acthaptic  # noqa: F401  (registers the acthaptic policy type)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_train import main

LABEL_KEYS = {"replan": "observation.replan", "phase": "observation.phase"}
_getitem = LeRobotDataset.__getitem__


def _getitem_with_labels(self, idx):
    item = _getitem(self, idx)
    for src, dst in LABEL_KEYS.items():
        if src in item and dst not in item:
            item[dst] = item[src]
    return item


LeRobotDataset.__getitem__ = _getitem_with_labels

if __name__ == "__main__":
    main()
