"""Derive the force-free dataset that ACT, Diffusion Policy and BC train on from a released lerobot_ft_replan
dataset, by dropping every force-derived column.

A separate copy rather than a training flag, because stock ACT and Diffusion Policy turn any
observation.environment_state column into an extra input: a baseline trained on lerobot_ft_replan would see the F/T
window. observation.torque and observation.contact_force stay in the files; ACT and Diffusion Policy read only
observation.state, the images and observation.environment_state, so they are never inputs.
Usage: python make_baseline_dataset.py <lerobot_ft_replan dataset> --out <new dataset>"""
import argparse
from pathlib import Path

from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FORCE_COLUMNS = ["observation.environment_state", "observation.wrench", "observation.torque_hist", "observation.torque_do", "replan"]

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
src = LeRobotDataset(repo_id="local/ft", root=args.root)
missing = [c for c in FORCE_COLUMNS if c not in src.meta.features]
if missing:
    raise KeyError(f"{args.root} lacks the expected columns {missing}; is it a lerobot_ft_replan dataset?")
ds = remove_feature(src, FORCE_COLUMNS, output_dir=args.out, repo_id=f"local/{args.out.name}")
print("wrote", args.out, "features:", sorted(ds.meta.features))
