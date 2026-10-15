"""Write dataset-level stats (mean / std / min / max / count) into meta/stats.json for every numeric `observation.*`
feature that lacks them. Columns added with add_features have no per-episode stats, so LeRobot's aggregation never
picks them up; without stats the normalizer would leave the F/T window in raw N and N m.
Usage: python write_stats.py <dataset>"""
import json
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(sys.argv[1])
ds = LeRobotDataset(repo_id="local/stats", root=root)
hf = ds.hf_dataset
path = root / "meta/stats.json"
stats = json.load(open(path))
added = []
for key, info in ds.meta.features.items():
    if not key.startswith("observation.") or info["dtype"] in ("video", "image", "string") or key in stats:
        continue
    x = np.asarray(hf[key], dtype=np.float64).reshape(len(hf), -1)
    stats[key] = {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist(), "count": [len(x)]}
    added.append(key)
json.dump(stats, open(path, "w"), indent=2)
print(root, "added stats for:", added)
