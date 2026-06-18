#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


PHASES = ["free", "onset", "persistent", "release", "post_contact"]


def summarize_policy(policy: str, outputs_dir: Path, seed: int) -> list[dict]:
    path = outputs_dir / f"pusht_multihorizon_event_prediction_{policy}_val.json"
    rows = []
    if not path.exists():
        return [
            {
                "policy": policy,
                "seed": seed,
                "horizon": "",
                "phase": "",
                "mean_mse": "",
                "p95": "",
                "count": "",
                "missing": str(path),
            }
        ]

    data = json.loads(path.read_text())
    for horizon, stats in data["horizons"].items():
        rows.append(
            {
                "policy": policy,
                "seed": seed,
                "horizon": horizon,
                "phase": "overall",
                "mean_mse": stats.get("mean_mse", ""),
                "p95": "",
                "count": stats.get("count", ""),
                "missing": "",
            }
        )
        for phase in PHASES:
            phase_stats = stats["phases"].get(phase, {})
            rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "horizon": horizon,
                    "phase": phase,
                    "mean_mse": phase_stats.get("mean_mse", ""),
                    "p95": phase_stats.get("p95", ""),
                    "count": phase_stats.get("count", ""),
                    "missing": "",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-csv", default="outputs/pusht_multihorizon_event_prediction_summary.csv")
    args = parser.parse_args()

    rows = []
    for template in args.policies:
        for seed in args.seeds:
            policy = template.format(seed=seed)
            rows.extend(summarize_policy(policy, Path(args.outputs_dir), seed))

    fieldnames = ["policy", "seed", "horizon", "phase", "mean_mse", "p95", "count", "missing"]
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
