#!/usr/bin/env python
import argparse
import csv
import json
import re
from pathlib import Path


PHASES = ["free", "onset", "persistent", "release", "post_contact"]


def parse_success_rate(path: Path):
    if not path.exists():
        return None
    text = path.read_text()
    match = re.search(r"success_rate': ([0-9.]+)", text)
    if not match:
        return None
    return float(match.group(1))


def summarize_policy(policy: str, planning_dir: Path, outputs_dir: Path, num_eval: int, seed: int):
    row = {
        "policy": policy,
        "seed": seed,
    }

    event_path = outputs_dir / f"pusht_event_prediction_{policy}_val.json"
    if event_path.exists():
        data = json.loads(event_path.read_text())
        row["overall_mse"] = data["overall"]["mean_mse"]
        for phase in PHASES:
            stats = data["phases"].get(phase, {})
            row[f"{phase}_mse"] = stats.get("mean_mse", "")
            row[f"{phase}_p95"] = stats.get("p95", "")
            row[f"{phase}_count"] = stats.get("count", "")
    else:
        row["overall_mse"] = ""
        for phase in PHASES:
            row[f"{phase}_mse"] = ""
            row[f"{phase}_p95"] = ""
            row[f"{phase}_count"] = ""

    planning_path = planning_dir / f"planning_{policy}_val{num_eval}_seed{seed}.txt"
    row["planning_success_rate"] = parse_success_rate(planning_path)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--planning-dir", default="src/plan")
    parser.add_argument("--output-csv", default="outputs/pusht_masking_ablation_summary.csv")
    args = parser.parse_args()

    rows = []
    for policy_template in args.policies:
        for seed in args.seeds:
            policy = policy_template.format(seed=seed)
            rows.append(
                summarize_policy(
                    policy=policy,
                    planning_dir=Path(args.planning_dir),
                    outputs_dir=Path(args.outputs_dir),
                    num_eval=args.num_eval,
                    seed=seed,
                )
            )

    fieldnames = ["policy", "seed", "planning_success_rate", "overall_mse"]
    for phase in PHASES:
        fieldnames.extend([f"{phase}_mse", f"{phase}_p95", f"{phase}_count"])

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
