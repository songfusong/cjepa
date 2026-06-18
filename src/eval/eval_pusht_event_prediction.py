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


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    return values[mask].mean().item()


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
def evaluate(args) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.checkpoint, device)
    history_size = int(model.model.history_size)
    num_preds = int(model.model.num_pred)
    slot_data = load_slot_data(args.embedding_dir, args.split)

    dataset = PushTSlotDataset(
        slot_data=slot_data,
        split=args.split,
        history_size=history_size,
        num_preds=num_preds,
        action_dir=args.action_dir,
        proprio_dir=args.proprio_dir,
        state_dir=args.state_dir,
        contact_event_dir=args.contact_event_dir,
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

    phase_sums = defaultdict(float)
    phase_counts = defaultdict(int)
    phase_window_values = defaultdict(list)
    horizon_sums = defaultdict(float)
    horizon_counts = defaultdict(int)
    overall_sum = 0.0
    overall_count = 0

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        pixels_embed = batch["pixels_embed"].to(device)
        proprio = batch["proprio"].to(device)
        action = batch["action"].to(device)
        event_mode = batch["event_mode"].to(device)

        embedding = encode_ap_tokens(model, pixels_embed, proprio, action)
        history = embedding[:, :history_size]
        target = embedding[:, history_size : history_size + num_preds, : pixels_embed.shape[2]]
        pred = model.model.predict(history, use_inference_function=True)[:, :num_preds, : pixels_embed.shape[2]]

        per_slot_mse = (pred - target).pow(2).mean(dim=-1)
        per_step_mse = per_slot_mse.mean(dim=-1)
        future_modes = event_mode[:, history_size : history_size + num_preds]

        overall_sum += float(per_step_mse.sum().item())
        overall_count += int(per_step_mse.numel())

        for horizon_idx in range(num_preds):
            values = per_step_mse[:, horizon_idx]
            horizon_sums[horizon_idx + 1] += float(values.sum().item())
            horizon_counts[horizon_idx + 1] += int(values.numel())

        for mode_id, mode_name in MODE_ID_TO_NAME.items():
            mask = future_modes == mode_id
            if mask.sum().item() == 0:
                continue
            phase_sums[mode_name] += float(per_step_mse[mask].sum().item())
            phase_counts[mode_name] += int(mask.sum().item())

        window_mode = future_modes[:, -1]
        final_error = per_step_mse[:, -1]
        for mode_id, mode_name in MODE_ID_TO_NAME.items():
            values = final_error[window_mode == mode_id].detach().cpu().tolist()
            phase_window_values[mode_name].extend(float(v) for v in values)

    phases = {}
    for mode_name in MODE_ID_TO_NAME.values():
        count = phase_counts[mode_name]
        phases[mode_name] = {
            "mean_mse": phase_sums[mode_name] / count if count else float("nan"),
            "count": count,
            **quantiles(phase_window_values[mode_name]),
        }

    horizons = {
        str(horizon): {
            "mean_mse": horizon_sums[horizon] / horizon_counts[horizon],
            "count": horizon_counts[horizon],
        }
        for horizon in sorted(horizon_counts)
    }

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "history_size": history_size,
        "num_preds": num_preds,
        "frameskip": args.frameskip,
        "samples": len(dataset),
        "evaluated_batches": min(len(loader), args.max_batches) if args.max_batches is not None else len(loader),
        "overall": {
            "mean_mse": overall_sum / overall_count if overall_count else float("nan"),
            "count": overall_count,
        },
        "horizons": horizons,
        "phases": phases,
    }
    return result


def write_csv(result: dict, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "mean_mse", "count", "p50", "p90", "p95"])
        writer.writeheader()
        for phase, stats in result["phases"].items():
            writer.writerow({"phase": phase, **stats})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PushT C-JEPA prediction error by contact phase.")
    parser.add_argument("--checkpoint", default="/home/jiaoyihang/.stable_worldmodel/pusht_cjepa_random_mask2_e5_bs2048_object.ckpt")
    parser.add_argument("--embedding-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_videosaur_slots.pkl")
    parser.add_argument("--action-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_action_meta.pkl")
    parser.add_argument("--proprio-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_proprio_meta.pkl")
    parser.add_argument("--state-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_state_meta.pkl")
    parser.add_argument("--contact-event-dir", default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_contact_event_meta.pkl")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--frameskip", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-json", default="outputs/pusht_event_prediction.json")
    parser.add_argument("--output-csv", default="outputs/pusht_event_prediction_phases.csv")
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
