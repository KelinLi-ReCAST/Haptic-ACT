"""Add the per-frame `replan` label (int64; 1 = the object is being pushed against the obstacle during the deliberate
jam: phase == jam (3) and contact force > 2 N) to a dataset, as a copy at --out. It is the training target of the
learned re-planning head, not an observation, and it is never normalized. In the demonstrations of PickCube,
StackCube and PushCube no jam happens, so the label is all zero.
Usage: python add_replan_label.py --root <dataset> --out <new dataset> --force-key observation.peg_box_force
       (observation.contact_force for PickCube, StackCube and PushCube)"""
import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JAM_PHASE = 3
FORCE_THRESHOLD = 2.0  # N

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--force-key", default="observation.peg_box_force")
args = ap.parse_args()
ds = LeRobotDataset(repo_id="local/src", root=args.root)
hf = ds.hf_dataset
phase = np.asarray(hf["phase"]).reshape(-1)
force = np.asarray(hf[args.force_key]).reshape(-1)
label = ((phase == JAM_PHASE) & (force > FORCE_THRESHOLD)).astype(np.int64)
print(f"frames={len(label)} positives={label.sum()} ({label.mean() * 100:.2f}%)")
features = {"replan": (lambda row, ep, fi: np.array([label[int(row["index"])]], dtype=np.int64), {"dtype": "int64", "shape": [1], "names": ["replan"]})}
out = add_features(ds, features, output_dir=args.out, repo_id=f"local/{args.out.name}")
print("wrote", args.out, "episodes", out.meta.total_episodes)
