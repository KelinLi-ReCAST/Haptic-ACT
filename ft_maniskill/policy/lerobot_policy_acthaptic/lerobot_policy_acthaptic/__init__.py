"""LeRobot third-party policy plugin: importing it registers the `acthaptic` policy type
(lerobot-train and lerobot-eval import every installed `lerobot_policy_*` package)."""
from .configuration_acthaptic import ACTHapticConfig  # noqa: F401  (registers "acthaptic")
