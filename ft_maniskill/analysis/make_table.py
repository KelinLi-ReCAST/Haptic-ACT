"""Results table (methods x tasks, "mean +- std %", plus a Mean column) from the evaluations in ../results.

results/<task>/<method>/<step>.json holds one evaluated checkpoint: 10 seeds x 25 trials, with the per-seed success
rates. A cell is the success rate after training has stabilised: for each seed, the mean over the last N evaluated
checkpoints (default 10); then mean +- std over the 10 seeds. Mean column: for each seed, the success averaged over
the tasks, then mean +- std of those 10 numbers (seed k uses environment seeds k..k+24 in every task). Std is the
population std. The best mean of each column is bold. Writes table.md and table.json to --out.
Usage: python make_table.py [--last 10] [--out ../results]"""
import argparse
import json
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
METHODS = [("haptic-act", "Haptic-ACT (F/T)"), ("act", "ACT"), ("dp", "Diffusion Policy"), ("bc", "BC")]
TASKS = [("peg", "PegInsertionSide"), ("pickcube", "PickCube"), ("stackcube", "StackCube"), ("pushcube", "PushCube")]


def per_seed(task: str, method: str, last: int) -> tuple[dict[int, float], str]:
    runs = sorted((int(p.stem), json.loads(p.read_text())) for p in (RESULTS / task / method).glob("*.json"))
    chosen = runs[-last:]
    seeds = sorted(int(k) for k in chosen[0][1]["per_seed_success"])
    rates = {k: st.fmean(r["per_seed_success"][str(k)] for _, r in chosen) for k in seeds}
    return rates, f"{chosen[0][0] // 1000}k-{chosen[-1][0] // 1000}k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    rows = []
    for method, name in METHODS:
        row = {"method": name}
        cells = {}
        for task, _ in TASKS:
            rates, steps = per_seed(task, method, args.last)
            cells[task] = rates
            row[task] = {"mean": st.fmean(rates.values()), "std": st.pstdev(rates.values()), "checkpoints": steps, "per_seed": rates}
        seeds = sorted(set.intersection(*[set(c) for c in cells.values()]))
        averaged = [st.fmean(cells[t][k] for t, _ in TASKS) for k in seeds]
        row["mean"] = {"mean": st.fmean(averaged), "std": st.pstdev(averaged)}
        rows.append(row)
    columns = [t for t, _ in TASKS] + ["mean"]
    best = {c: max(r[c]["mean"] for r in rows) for c in columns}

    def fmt(row, c):
        text = f"{row[c]['mean']:.1f} ± {row[c]['std']:.1f}"
        return f"**{text}**" if row[c]["mean"] == best[c] else text

    header = ["Method"] + [name for _, name in TASKS] + ["Mean"]
    md = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    md += ["| " + " | ".join([r["method"]] + [fmt(r, c) for c in columns]) + " |" for r in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table.md").write_text("\n".join(md) + "\n")
    (args.out / "table.json").write_text(json.dumps({"last_checkpoints": args.last, "std": "population", "rows": rows}, indent=2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
