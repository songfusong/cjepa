import argparse
import csv
import json
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.custom_codes.custom_dataset import PushTSlotDataset
from src.custom_codes.hungarian import reorder_slots_to_match_AP


MODE_ID_TO_NAME = {
    0: "free",
    1: "onset",
    2: "persistent",
    3: "release",
    4: "post_contact",
}


def load_slot_data(path: str, split: str) -> dict:
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data[split]


def load_model(path: str, device: torch.device):
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def encode_ap_tokens(module, pixels_embed: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    proprio = torch.nan_to_num(proprio.float(), 0.0)
    action = torch.nan_to_num(action.float(), 0.0)
    proprio_embed = module.model.proprio_encoder(proprio).unsqueeze(2)
    action_embed = module.model.action_encoder(action).unsqueeze(2)
    return torch.cat([pixels_embed.float(), proprio_embed, action_embed], dim=2)


def parse_horizons(raw: str) -> list[int]:
    horizons = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not horizons or horizons[0] < 1:
        raise ValueError("--horizons must contain positive integers")
    return horizons


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }


@torch.no_grad()
def rollout_one_step_model(
    module,
    embedding: torch.Tensor,
    history_size: int,
    horizon: int,
    slot_num: int,
    use_hungarian: bool,
) -> torch.Tensor:
    """Autoregressively roll a one-step PushT AP world model over latent slots.

    The model was trained with num_preds=1. For H-step evaluation, predicted
    object slots are fed back into the rolling history while proprio/action AP
    tokens are taken from the recorded future sequence. This evaluates latent
    dynamics under the dataset action/proprio context, not closed-loop control.
    """
    history = embedding[:, :history_size].clone()
    preds = []
    for step in range(horizon):
        next_pred = module.model.predict(history, use_inference_function=True)[:, :1]
        if use_hungarian:
            reference = history[:, -1, :slot_num, :]
            next_pred = reorder_slots_to_match_AP(
                next_pred,
                reference=reference,
                cost_type="mse",
                slot_dim=slot_num,
            )

        next_frame = next_pred[:, 0].clone()
        future_index = history_size + step
        next_frame[:, slot_num:, :] = embedding[:, future_index, slot_num:, :]
        preds.append(next_frame[:, :slot_num, :])
        history = torch.cat([history[:, 1:], next_frame.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


@torch.no_grad()
def evaluate(args) -> dict:
    horizons = parse_horizons(args.horizons)
    max_horizon = max(horizons)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.checkpoint, device)
    history_size = int(model.model.history_size)
    slot_data = load_slot_data(args.embedding_dir, args.split)

    dataset = PushTSlotDataset(
        slot_data=slot_data,
        split=args.split,
        history_size=history_size,
        num_preds=max_horizon,
        action_dir=args.action_dir,
        proprio_dir=args.proprio_dir,
        state_dir=args.state_dir,
        contact_event_dir=args.contact_event_dir,
        contact_event_fields="full",
        frameskip=args.frameskip,
        seed=args.seed,
    )
    if args.max_batches is not None:
        print(f"Using max_batches={args.max_batches}; results are a smoke estimate.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    overall_sum = 0.0
    overall_count = 0
    horizon_sums = defaultdict(float)
    horizon_counts = defaultdict(int)
    phase_sums = {h: defaultdict(float) for h in horizons}
    phase_counts = {h: defaultdict(int) for h in horizons}
    phase_values = {h: defaultdict(list) for h in horizons}

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        pixels_embed = batch["pixels_embed"].to(device)
        proprio = batch["proprio"].to(device)
        action = batch["action"].to(device)
        event_mode = batch["event_mode"].to(device)

        embedding = encode_ap_tokens(model, pixels_embed, proprio, action)
        pred = rollout_one_step_model(
            module=model,
            embedding=embedding,
            history_size=history_size,
            horizon=max_horizon,
            slot_num=pixels_embed.shape[2],
            use_hungarian=args.use_hungarian,
        )
        target = embedding[:, history_size : history_size + max_horizon, : pixels_embed.shape[2]]
        future_modes = event_mode[:, history_size : history_size + max_horizon]

        per_slot_mse = (pred - target).pow(2).mean(dim=-1)
        per_step_mse = per_slot_mse.mean(dim=-1)

        overall_sum += float(per_step_mse.sum().item())
        overall_count += int(per_step_mse.numel())

        for horizon in horizons:
            values = per_step_mse[:, horizon - 1]
            modes = future_modes[:, horizon - 1]
            horizon_sums[horizon] += float(values.sum().item())
            horizon_counts[horizon] += int(values.numel())
            for mode_id, mode_name in MODE_ID_TO_NAME.items():
                mask = modes == mode_id
                if mask.sum().item() == 0:
                    continue
                selected = values[mask]
                phase_sums[horizon][mode_name] += float(selected.sum().item())
                phase_counts[horizon][mode_name] += int(selected.numel())
                phase_values[horizon][mode_name].extend(float(v) for v in selected.detach().cpu().tolist())

    horizon_results = {}
    for horizon in horizons:
        phase_results = {}
        for mode_name in MODE_ID_TO_NAME.values():
            count = phase_counts[horizon][mode_name]
            phase_results[mode_name] = {
                "mean_mse": phase_sums[horizon][mode_name] / count if count else float("nan"),
                "count": count,
                **quantiles(phase_values[horizon][mode_name]),
            }
        horizon_results[str(horizon)] = {
            "mean_mse": horizon_sums[horizon] / horizon_counts[horizon] if horizon_counts[horizon] else float("nan"),
            "count": horizon_counts[horizon],
            "phases": phase_results,
        }

    return {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "history_size": history_size,
        "model_num_preds": int(model.model.num_pred),
        "frameskip": args.frameskip,
        "horizons": horizon_results,
        "overall": {
            "mean_mse": overall_sum / overall_count if overall_count else float("nan"),
            "count": overall_count,
        },
        "rollout": {
            "autoregressive": True,
            "uses_recorded_future_ap_tokens": True,
            "use_hungarian": args.use_hungarian,
        },
        "samples": len(dataset),
        "evaluated_batches": min(len(loader), args.max_batches) if args.max_batches is not None else len(loader),
    }


def write_csv(result: dict, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["horizon", "phase", "mean_mse", "count", "p50", "p90", "p95"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for horizon, stats in result["horizons"].items():
            writer.writerow(
                {
                    "horizon": horizon,
                    "phase": "overall",
                    "mean_mse": stats["mean_mse"],
                    "count": stats["count"],
                    "p50": "",
                    "p90": "",
                    "p95": "",
                }
            )
            for phase, phase_stats in stats["phases"].items():
                writer.writerow({"horizon": horizon, "phase": phase, **phase_stats})


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoregressive multi-horizon PushT event-centric latent eval.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_videosaur_slots.pkl")
    parser.add_argument("--action-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_action_meta.pkl")
    parser.add_argument("--proprio-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_proprio_meta.pkl")
    parser.add_argument("--state-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_state_meta.pkl")
    parser.add_argument("--contact-event-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_contact_event_meta.pkl")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--horizons", default="1,5,10,20")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--use-hungarian", action="store_true")
    parser.add_argument("--output-json", default="outputs/pusht_multihorizon_event_prediction.json")
    parser.add_argument("--output-csv", default="outputs/pusht_multihorizon_event_prediction.csv")
    args = parser.parse_args()

    result = evaluate(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    write_csv(result, Path(args.output_csv))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved JSON to {output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
