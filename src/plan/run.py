import time
from pathlib import Path

import datasets
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
import wandb


class HFPushTVideoDataset:
    """Adapter from the downloaded HuggingFace PushT dataset to World.evaluate."""

    column_names = ["pixels", "action", "state", "proprio"]

    def __init__(self, path):
        self.path = Path(path)
        self.dataset = datasets.load_from_disk(path).with_format(None)
        meta_columns = ["episode_idx", "step_idx", "action", "state", "proprio", "pixels"]
        meta = self.dataset.remove_columns(
            [col for col in self.dataset.column_names if col not in meta_columns]
        ).with_format(None)

        self.episode_idx = np.asarray(meta["episode_idx"])
        self.step_idx = np.asarray(meta["step_idx"])
        self.action = np.asarray(meta["action"], dtype=np.float32)
        self.state = np.asarray(meta["state"], dtype=np.float32)
        self.proprio = np.asarray(meta["proprio"], dtype=np.float32)
        self.pixel_paths = np.asarray(meta["pixels"])

        self.episode_rows = {}
        self.episode_video_paths = {}
        for ep_id in np.unique(self.episode_idx):
            rows = np.nonzero(self.episode_idx == ep_id)[0]
            order = np.argsort(self.step_idx[rows])
            rows = rows[order]
            self.episode_rows[int(ep_id)] = rows
            self.episode_video_paths[int(ep_id)] = self.pixel_paths[rows[0]]

        self._video_readers = {}

    def _reader(self, episode_idx):
        episode_idx = int(episode_idx)
        if episode_idx not in self._video_readers:
            from decord import VideoReader, bridge

            bridge.set_bridge("torch")
            video_path = self.path / self.episode_video_paths[episode_idx]
            self._video_readers[episode_idx] = VideoReader(str(video_path), num_threads=1)
        return self._video_readers[episode_idx]

    def load_chunk(self, episodes_idx, start, end):
        chunks = []
        for ep_id, s, e in zip(episodes_idx, start, end):
            ep_id = int(ep_id)
            s = int(s)
            e = int(e)
            rows = self.episode_rows[ep_id][s:e]
            frame_indices = self.step_idx[rows].astype(int).tolist()
            frames = self._reader(ep_id).get_batch(frame_indices).permute(0, 3, 1, 2)
            chunks.append(
                {
                    "pixels": frames,
                    "action": self.action[rows],
                    "state": self.state[rows],
                    "proprio": self.proprio[rows],
                }
            )
        return chunks


def img_transform():
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=196),
            transforms.CenterCrop(size=196),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    episode_idx = np.asarray(dataset["episode_idx"])
    step_idx = np.asarray(dataset["step_idx"])
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, (
        "Planning horizon must be smaller than or equal to eval_budget"
    )
    if cfg.wandb.use_wandb:
        # Initialize wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=dict(cfg))

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224), render_mode="rgb_array")
    cache_dir = cfg.cache_dir or swm.data.utils.get_cache_dir()

    # create the transform
    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    dataset_path = Path(cache_dir, cfg.eval.dataset_name)
    dataset = datasets.load_from_disk(dataset_path)
    meta_columns = ["episode_idx", "step_idx", "action", "proprio"]
    meta_dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in meta_columns]
    ).with_format(None)
    episode_idx = np.asarray(meta_dataset["episode_idx"])
    step_idx = np.asarray(meta_dataset["step_idx"])
    ep_indices, _ = np.unique(episode_idx, return_index=True)

    # create the processing
    action_process = preprocessing.StandardScaler()
    action_process.fit(np.asarray(meta_dataset["action"]))

    proprio_process = preprocessing.StandardScaler()
    proprio_process.fit(np.asarray(meta_dataset["proprio"]))

    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }

    # -- run evaluation
    model = swm.policy.AutoCostModel(cfg.policy, cache_dir)
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)

    # model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(meta_dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in episode_idx])

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = step_idx <= max_start_per_row
    dataset_start = meta_dataset.select(np.nonzero(valid_mask)[0])

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(dataset_start) - 1, size=cfg.eval.num_eval, replace=False)
    eval_episodes = dataset_start[random_episode_indices]["episode_idx"]
    eval_start_idx = dataset_start[random_episode_indices]["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    if cfg.eval.data_format == "hf_video":
        dataset = HFPushTVideoDataset(dataset_path)
    elif cfg.eval.data_format == "frame":
        dataset = swm.data.FrameDataset(cfg.eval.dataset_name, cache_dir=cache_dir)
    elif cfg.eval.data_format == "video":
        dataset = swm.data.VideoDataset(cfg.eval.dataset_name, cache_dir=cache_dir)
    else:
        raise NotImplementedError(f"Data format '{cfg.eval.data_format}' not supported.")

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=list(eval_start_idx),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=list(eval_episodes),
        callables=[
            {
                "method": "_set_state",
                "args": {"state": {"value": "state", "in_dataset": True}},
            },
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_state", "in_dataset": True}},
            },
        ],
    )
    end_time = time.time()

    if cfg.wandb.use_wandb:
        # Log metrics to wandb
        wandb.log(metrics)
        # Finish wandb run
        wandb.finish()

    # dump results
    print(metrics)
    # ---- dump results to a txt file ----
    results_path = Path(__file__).parent / cfg.output.filename
    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs
        f.write(f"policy: {cfg.policy}\n")
        f.write(f"dataset_name: {cfg.eval.dataset_name}\n")
        f.write(f"goal_offset_steps: {cfg.eval.goal_offset_steps}\n")
        f.write(f"eval_budget: {cfg.eval.eval_budget}\n")
        f.write(f"horizon: {cfg.plan_config.horizon}\n")
        f.write(f"receding_horizon: {cfg.plan_config.receding_horizon}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
