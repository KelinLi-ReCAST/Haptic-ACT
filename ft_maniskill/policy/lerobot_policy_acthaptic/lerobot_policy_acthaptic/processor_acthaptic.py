"""Same pre/post-processors as ACT (normalization, batching, device)."""
from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_acthaptic_pre_post_processors(*args, **kwargs):
    return make_act_pre_post_processors(*args, **kwargs)
