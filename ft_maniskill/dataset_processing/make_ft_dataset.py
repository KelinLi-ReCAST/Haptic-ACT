"""Replace observation.environment_state (the joint-torque window of augment_dataset.py) by the F/T window, the input
of F/T Haptic-ACT: the last 5 TCP wrenches [fx fy fz tx ty tz], oldest first, padded at the episode start with the
episode's first reading (30 dims). Everything else is unchanged.
Usage: python make_ft_dataset.py --root <dataset with observation.wrench> --out <new dataset>"""
import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import add_features, remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

HIST = 5
ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
tmp = args.out.with_name(args.out.name + "_tmp")

ds = LeRobotDataset(repo_id="local/src", root=args.root)
hf = ds.hf_dataset
w = np.asarray(hf["observation.wrench"], dtype=np.float32)
ep = np.asarray(hf["episode_index"]).reshape(-1)
n = len(ep)
assert w.shape == (n, 6), w.shape
hist = np.zeros((n, HIST * 6), np.float32)
starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
ends = np.r_[starts[1:], n]
for a, b in zip(starts, ends):
    for k in range(HIST):
        idx = np.clip(np.arange(b - a) - (HIST - 1 - k), 0, None)
        hist[a:b, k * 6:(k + 1) * 6] = w[a:b][idx]
print("frames", n, "episodes", len(starts), "|F| p50/p99 N:", np.round(np.percentile(np.linalg.norm(w[:, :3], axis=1), [50, 99]), 2))
mid = remove_feature(ds, "observation.environment_state", output_dir=tmp, repo_id="local/ft-tmp")
names = [f"t-{HIST - 1 - k}_{c}" for k in range(HIST) for c in ("fx", "fy", "fz", "tx", "ty", "tz")]
feature = {"observation.environment_state": (lambda row, e, fi: hist[int(row["index"])], {"dtype": "float32", "shape": [HIST * 6], "names": names})}
out = add_features(mid, feature, output_dir=args.out, repo_id=f"local/{args.out.name}")
print("wrote", args.out, sorted(out.meta.features))
