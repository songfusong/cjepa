import argparse
import json
import os
import pickle as pkl
from pathlib import Path

import numpy as np


MODE_TO_ID = {
    "free": 0,
    "onset": 1,
    "persistent": 2,
    "release": 3,
    "post_contact": 4,
}


def rotation_matrix(theta: np.ndarray) -> np.ndarray:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.stack(
        [
            np.stack([cos_t, -sin_t], axis=-1),
            np.stack([sin_t, cos_t], axis=-1),
        ],
        axis=-2,
    )


def transform_vertices(local_vertices: np.ndarray, position: np.ndarray, theta: np.ndarray) -> np.ndarray:
    rot = rotation_matrix(theta)
    return local_vertices[None, :, :] @ np.swapaxes(rot, -1, -2) + position[:, None, :]


def point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    seg = end - start
    seg_len_sq = np.sum(seg * seg, axis=-1)
    rel = points - start
    t = np.sum(rel * seg, axis=-1) / np.maximum(seg_len_sq, 1e-8)
    t = np.clip(t, 0.0, 1.0)
    closest = start + t[:, None] * seg
    return np.linalg.norm(points - closest, axis=-1)


def points_in_convex_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return whether each point lies inside a consistently ordered convex polygon."""
    edges = np.roll(polygon, -1, axis=1) - polygon
    rel = points[:, None, :] - polygon
    cross = edges[:, :, 0] * rel[:, :, 1] - edges[:, :, 1] * rel[:, :, 0]
    return np.all(cross >= -1e-6, axis=1) | np.all(cross <= 1e-6, axis=1)


def point_polygon_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = points_in_convex_polygon(points, polygon)
    distances = []
    for edge_idx in range(polygon.shape[1]):
        start = polygon[:, edge_idx, :]
        end = polygon[:, (edge_idx + 1) % polygon.shape[1], :]
        distances.append(point_segment_distance(points, start, end))
    edge_distance = np.min(np.stack(distances, axis=1), axis=1)
    return np.where(inside, 0.0, edge_distance)


def pusht_t_block_distance(states: np.ndarray, scale: float = 30.0) -> np.ndarray:
    """Approximate distance from agent center to the PushT T-block polygon union."""
    agent_pos = states[:, :2].astype(np.float64)
    block_pos = states[:, 2:4].astype(np.float64)
    block_theta = states[:, 4].astype(np.float64)

    length = 4.0
    horizontal = np.array(
        [
            [-length * scale / 2.0, scale],
            [length * scale / 2.0, scale],
            [length * scale / 2.0, 0.0],
            [-length * scale / 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    vertical = np.array(
        [
            [-scale / 2.0, scale],
            [-scale / 2.0, length * scale],
            [scale / 2.0, length * scale],
            [scale / 2.0, scale],
        ],
        dtype=np.float64,
    )

    horizontal_world = transform_vertices(horizontal, block_pos, block_theta)
    vertical_world = transform_vertices(vertical, block_pos, block_theta)
    distance_h = point_polygon_distance(agent_pos, horizontal_world)
    distance_v = point_polygon_distance(agent_pos, vertical_world)
    return np.minimum(distance_h, distance_v).astype(np.float32)


def mark_window(mask: np.ndarray, event_indices: np.ndarray, k_pre: int, k_post: int) -> None:
    for idx in event_indices.tolist():
        start = max(0, idx - k_pre)
        stop = min(mask.shape[0], idx + k_post + 1)
        mask[start:stop] = True


def build_episode_events(
    states: np.ndarray,
    agent_radius: float,
    contact_margin: float,
    k_pre: int,
    k_post: int,
) -> dict:
    distance = pusht_t_block_distance(states)
    contact = distance <= (agent_radius + contact_margin)
    prev_contact = np.concatenate([[False], contact[:-1]])
    next_contact = np.concatenate([contact[1:], [False]])

    onset = contact & ~prev_contact
    release = (~contact) & prev_contact
    persistent = contact & ~onset

    onset_indices = np.flatnonzero(onset)
    release_indices = np.flatnonzero(release)
    post_contact = np.zeros_like(contact, dtype=bool)
    for idx in np.concatenate([onset_indices, release_indices]).tolist():
        start = min(contact.shape[0], idx + 1)
        stop = min(contact.shape[0], idx + k_post + 1)
        post_contact[start:stop] = True

    event_window = np.zeros_like(contact, dtype=bool)
    mark_window(event_window, onset_indices, k_pre, k_post)
    mark_window(event_window, release_indices, k_pre, k_post)

    mode = np.full(contact.shape[0], MODE_TO_ID["free"], dtype=np.int64)
    mode[persistent] = MODE_TO_ID["persistent"]
    mode[post_contact] = MODE_TO_ID["post_contact"]
    mode[onset] = MODE_TO_ID["onset"]
    mode[release] = MODE_TO_ID["release"]

    distance_delta = np.diff(distance, prepend=distance[:1])

    return {
        "distance": distance.astype(np.float32),
        "distance_delta": distance_delta.astype(np.float32),
        "contact": contact.astype(np.bool_),
        "onset": onset.astype(np.bool_),
        "release": release.astype(np.bool_),
        "persistent": persistent.astype(np.bool_),
        "post_contact": post_contact.astype(np.bool_),
        "event_window": event_window.astype(np.bool_),
        "mode": mode,
    }


def summarize_split(split_events: dict) -> dict:
    num_frames = 0
    counts = {
        "contact": 0,
        "onset": 0,
        "release": 0,
        "persistent": 0,
        "post_contact": 0,
        "event_window": 0,
    }
    for episode in split_events.values():
        num_frames += int(episode["contact"].shape[0])
        for key in counts:
            counts[key] += int(np.asarray(episode[key]).sum())
    summary = {"episodes": len(split_events), "frames": num_frames}
    for key, value in counts.items():
        summary[f"{key}_count"] = value
        summary[f"{key}_rate"] = float(value / max(num_frames, 1))
    return summary


def maybe_save_timelines(events: dict, timeline_dir: Path, max_plots: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional diagnostic path
        print(f"Skipping timeline plots because matplotlib is unavailable: {exc}")
        return

    timeline_dir.mkdir(parents=True, exist_ok=True)
    for split, split_events in events.items():
        for plot_idx, (video_id, episode) in enumerate(split_events.items()):
            if plot_idx >= max_plots:
                break
            t = np.arange(episode["distance"].shape[0])
            fig, ax1 = plt.subplots(figsize=(10, 3))
            ax1.plot(t, episode["distance"], label="surface_distance", color="tab:blue")
            ax1.set_xlabel("timestep")
            ax1.set_ylabel("distance (px)")
            ax2 = ax1.twinx()
            ax2.step(t, episode["contact"].astype(np.float32), where="post", label="contact", color="tab:orange")
            ax2.step(t, episode["event_window"].astype(np.float32), where="post", label="event_window", color="tab:green", alpha=0.6)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel("binary label")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper right")
            fig.tight_layout()
            safe_video_id = video_id.replace("/", "_").replace(".", "_")
            fig.savefig(timeline_dir / f"{split}_{safe_video_id}.png", dpi=140)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PushT contact/event labels from state metadata.")
    parser.add_argument(
        "--state-meta",
        default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_state_meta.pkl",
        help="Path to pusht_expert_state_meta.pkl.",
    )
    parser.add_argument(
        "--output",
        default="/home/jiaoyihang/.stable_worldmodel/artifacts/pusht/pusht_expert_contact_event_meta.pkl",
        help="Output pickle path.",
    )
    parser.add_argument("--agent-radius", type=float, default=15.0)
    parser.add_argument("--contact-margin", type=float, default=3.0)
    parser.add_argument("--k-pre", type=int, default=3)
    parser.add_argument("--k-post", type=int, default=5)
    parser.add_argument("--timeline-dir", default=None)
    parser.add_argument("--max-plots", type=int, default=5)
    args = parser.parse_args()

    with open(args.state_meta, "rb") as f:
        state_meta = pkl.load(f)

    events = {}
    summary = {}
    for split, split_states in state_meta.items():
        events[split] = {}
        for video_id, states in split_states.items():
            events[split][video_id] = build_episode_events(
                np.asarray(states),
                agent_radius=args.agent_radius,
                contact_margin=args.contact_margin,
                k_pre=args.k_pre,
                k_post=args.k_post,
            )
        summary[split] = summarize_split(events[split])

    payload = {
        "meta": {
            "source_state_meta": args.state_meta,
            "agent_radius": args.agent_radius,
            "contact_margin": args.contact_margin,
            "k_pre": args.k_pre,
            "k_post": args.k_post,
            "mode_to_id": MODE_TO_ID,
        },
        "summary": summary,
        "events": events,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump(payload, f)

    if args.timeline_dir:
        maybe_save_timelines(events, Path(args.timeline_dir), args.max_plots)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved contact/event metadata to {output_path}")


if __name__ == "__main__":
    main()
