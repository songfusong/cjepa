from __future__ import annotations

import math
import os
import pickle as pkl
from dataclasses import dataclass

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm


INPUT_STATE_META = "/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_state_meta.pkl"
OUTPUT_META = "/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_contact_event_meta.pkl"
TRAIN_DATASET = "/home/jiaoyihang/.stable_worldmodel/pusht_expert_train"
VAL_DATASET = "/home/jiaoyihang/.stable_worldmodel/pusht_expert_val"

AGENT_RADIUS = 0.375 * 40.0
BLOCK_SCALE = 40.0
BLOCK_HALF = 20.0
BLOCK_LENGTH = 4.0 * 30.0
BLOCK_HALF_LONG = BLOCK_LENGTH / 2.0
BLOCK_THICK = 30.0
BLOCK_HALF_THICK = BLOCK_THICK / 2.0


@dataclass
class EventConfig:
    contact_threshold: float = AGENT_RADIUS + max(BLOCK_HALF_LONG, BLOCK_HALF_THICK)
    event_window: int = 3


def _block_corners(center_x: float, center_y: float, theta: float) -> np.ndarray:
    # Approximate the PushT T as the union of a horizontal and vertical rectangle.
    rects = []
    local_rects = [
        np.array(
            [
                [-BLOCK_HALF_LONG, -BLOCK_HALF_THICK],
                [BLOCK_HALF_LONG, -BLOCK_HALF_THICK],
                [BLOCK_HALF_LONG, BLOCK_HALF_THICK],
                [-BLOCK_HALF_LONG, BLOCK_HALF_THICK],
            ]
        ),
        np.array(
            [
                [-BLOCK_HALF_THICK, -BLOCK_HALF_LONG],
                [BLOCK_HALF_THICK, -BLOCK_HALF_LONG],
                [BLOCK_HALF_THICK, BLOCK_HALF_LONG],
                [-BLOCK_HALF_THICK, BLOCK_HALF_LONG],
            ]
        ),
    ]
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    for rect in local_rects:
        rects.append(rect @ rot.T + np.array([center_x, center_y], dtype=np.float64))
    return np.concatenate(rects, axis=0)


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _point_poly_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    edges = zip(polygon, np.roll(polygon, -1, axis=0))
    return min(_point_segment_distance(point, a, b) for a, b in edges)


def _label_events(state: np.ndarray, cfg: EventConfig) -> dict[str, np.ndarray]:
    agent = state[:, :2]
    block_x = state[:, 2]
    block_y = state[:, 3]
    block_theta = state[:, 4]

    distance = np.zeros(len(state), dtype=np.float32)
    for i in range(len(state)):
        block_poly = _block_corners(block_x[i], block_y[i], block_theta[i])
        distance[i] = _point_poly_distance(agent[i], block_poly)

    contact = distance <= cfg.contact_threshold
    onset = contact & np.concatenate(([False], ~contact[:-1]))
    release = (~contact) & np.concatenate(([False], contact[:-1]))
    persistent = contact & ~onset
    post_contact = np.zeros_like(contact)
    seen_release = False
    for i in range(len(contact)):
        if release[i]:
            seen_release = True
        post_contact[i] = seen_release and not contact[i]

    event_label = np.full(len(contact), "free", dtype=object)
    event_label[contact] = "contact"
    event_label[onset] = "onset"
    event_label[release] = "release"
    event_label[persistent] = "persistent"
    event_label[post_contact] = "post_contact"

    event_window = np.zeros(len(contact), dtype=np.int64)
    for i in range(len(contact)):
        if onset[i] or release[i]:
            start = max(0, i - cfg.event_window)
            stop = min(len(contact), i + cfg.event_window + 1)
            event_window[start:stop] = 1

    return {
        "contact": contact.astype(np.float32),
        "onset": onset.astype(np.float32),
        "release": release.astype(np.float32),
        "persistent": persistent.astype(np.float32),
        "post_contact": post_contact.astype(np.float32),
        "event_window": event_window.astype(np.float32),
        "distance": distance.astype(np.float32),
        "event_label": event_label,
    }


def main() -> None:
    with open(INPUT_STATE_META, "rb") as f:
        state_meta = pkl.load(f)
    train_meta = load_from_disk(TRAIN_DATASET)
    val_meta = load_from_disk(VAL_DATASET)

    out = {"train": {}, "val": {}}
    cfg = EventConfig()

    for split, dataset in [("train", train_meta), ("val", val_meta)]:
        print(f"Labeling {split}: {len(state_meta[split])} episodes")
        for key, state in tqdm(state_meta[split].items()):
            labels = _label_events(state, cfg)
            out[split][key] = labels
        print(
            split,
            "contact_rate=",
            float(np.mean([v["contact"].mean() for v in out[split].values()])),
            "onset_rate=",
            float(np.mean([v["onset"].mean() for v in out[split].values()])),
            "release_rate=",
            float(np.mean([v["release"].mean() for v in out[split].values()])),
        )

    os.makedirs(os.path.dirname(OUTPUT_META), exist_ok=True)
    with open(OUTPUT_META, "wb") as f:
        pkl.dump(out, f)
    print(f"Saved {OUTPUT_META}")


if __name__ == "__main__":
    main()
