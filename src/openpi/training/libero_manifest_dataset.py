"""Query-only LIBERO-100 dataset backed by the RICL split corpus metadata."""

from __future__ import annotations

import bisect
from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import json
from os import PathLike
import pathlib
from typing import SupportsIndex

import h5py
import numpy as np


def flip_libero_image(image: np.ndarray) -> np.ndarray:
    """Rotate a raw LIBERO image by 180 degrees to match online evaluation."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC image, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def libero_action_chunk(actions: np.ndarray, step_idx: int, action_horizon: int) -> np.ndarray:
    """Build a fixed-size future action chunk using LIBERO's episode-end padding."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected LIBERO actions with shape [steps, 7], got {actions.shape}")
    if not 0 <= step_idx < len(actions):
        raise IndexError(f"{step_idx=} is outside an episode with {len(actions)} actions")

    chunk = actions[step_idx : step_idx + action_horizon]
    if len(chunk) == action_horizon:
        return chunk

    padding = np.zeros((action_horizon - len(chunk), 7), dtype=np.float32)
    padding[:, -1] = actions[-1, -1]
    return np.concatenate((chunk, padding), axis=0)


@dataclass(frozen=True)
class EpisodeRef:
    task_id: int
    demo_id: str
    source_hdf5: str
    prompt: str
    length: int


class LiberoManifestDataset:
    """Expose only the 40 query demos of each of the 70 training tasks.

    The metadata file is shared with RICL, but this dataset deliberately never
    opens retrieval embeddings, neighbour files, or context demonstrations.
    """

    def __init__(self, corpus_dir: str | PathLike[str], *, action_horizon: int) -> None:
        self.root = pathlib.Path(corpus_dir).expanduser().resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"LIBERO corpus metadata was not found at {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") not in (1, 2, 3):
            raise ValueError(f"Unsupported LIBERO corpus format: {self.metadata.get('format_version')!r}")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        corpus_horizon = int(self.metadata.get("action_horizon", action_horizon))
        if action_horizon != corpus_horizon:
            raise ValueError(f"Model action horizon is {action_horizon}, corpus was built with {corpus_horizon}")

        self.action_horizon = action_horizon
        self._episodes: list[EpisodeRef] = []
        self._episode_ends: list[int] = []
        self._h5_files: dict[str, h5py.File] = {}
        self._actions: dict[tuple[str, str], np.ndarray] = {}
        total_frames = 0

        for task in self.metadata["tasks"]:
            if not task["is_train"]:
                continue
            source_hdf5 = str(pathlib.Path(task["source_hdf5"]).expanduser().resolve())
            with h5py.File(source_hdf5, "r") as h5_file:
                for raw_demo_id in task["query_demo_ids"]:
                    demo_id = str(raw_demo_id)
                    prefix = f"data/{demo_id}"
                    if prefix not in h5_file:
                        raise KeyError(f"{demo_id!r} is missing from {source_hdf5}")
                    length = int(h5_file[f"{prefix}/actions"].shape[0])
                    if length <= 0:
                        raise ValueError(f"Empty demonstration: task={task['task_id']} demo={demo_id}")
                    obs = h5_file[f"{prefix}/obs"]
                    for key in ("agentview_rgb", "eye_in_hand_rgb", "ee_states", "gripper_states"):
                        if int(obs[key].shape[0]) != length:
                            raise ValueError(
                                f"Length mismatch for task={task['task_id']} demo={demo_id} key={key}: "
                                f"{obs[key].shape[0]} != {length}"
                            )
                    self._episodes.append(
                        EpisodeRef(
                            task_id=int(task["task_id"]),
                            demo_id=demo_id,
                            source_hdf5=source_hdf5,
                            prompt=str(task["prompt"]),
                            length=length,
                        )
                    )
                    total_frames += length
                    self._episode_ends.append(total_frames)

        if not self._episodes:
            raise ValueError("The LIBERO corpus has no training query demonstrations")

    def __len__(self) -> int:
        return self._episode_ends[-1]

    def __getitem__(self, index: SupportsIndex) -> dict[str, object]:
        sample_index = index.__index__()
        if sample_index < 0:
            sample_index += len(self)
        if not 0 <= sample_index < len(self):
            raise IndexError(sample_index)

        episode_index = bisect.bisect_right(self._episode_ends, sample_index)
        episode_start = 0 if episode_index == 0 else self._episode_ends[episode_index - 1]
        step_idx = sample_index - episode_start
        episode = self._episodes[episode_index]
        h5_file = self._h5(episode.source_hdf5)
        prefix = f"data/{episode.demo_id}"
        obs = h5_file[f"{prefix}/obs"]
        action_key = (episode.source_hdf5, episode.demo_id)
        if action_key not in self._actions:
            self._actions[action_key] = np.asarray(h5_file[f"{prefix}/actions"][:], dtype=np.float32)
        actions = self._actions[action_key]

        return {
            "image": flip_libero_image(obs["agentview_rgb"][step_idx]),
            "wrist_image": flip_libero_image(obs["eye_in_hand_rgb"][step_idx]),
            "state": np.concatenate((obs["ee_states"][step_idx], obs["gripper_states"][step_idx])).astype(np.float32),
            "actions": libero_action_chunk(actions, step_idx, self.action_horizon),
            "prompt": episode.prompt,
        }

    def close(self) -> None:
        for h5_file in self._h5_files.values():
            h5_file.close()
        self._h5_files.clear()
        self._actions.clear()

    def _h5(self, source_hdf5: str) -> h5py.File:
        if source_hdf5 not in self._h5_files:
            self._h5_files[source_hdf5] = h5py.File(source_hdf5, "r")
        return self._h5_files[source_hdf5]

    def __getstate__(self) -> dict:
        # h5py handles cannot be pickled into spawned data-loader workers.
        state = dict(self.__dict__)
        state["_h5_files"] = {}
        state["_actions"] = {}
        return state

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class LiberoVictrDataset(LiberoManifestDataset):
    """LIBERO queries augmented with precomputed VICTR retrieval context."""

    def __init__(
        self,
        corpus_dir: str | PathLike[str],
        *,
        action_horizon: int,
        num_context_chunks: int,
        context_chunk_size: int,
        context_frames_per_chunk: int,
        retrieval_metric: str,
    ) -> None:
        super().__init__(corpus_dir, action_horizon=action_horizon)
        corpus_backend = str(self.metadata.get("retrieval_backend", "dino"))
        expected_backend = {
            "vision": "dino",
            "progress": "progress",
            "vision_progress": "dino_progress",
        }.get(retrieval_metric)
        if corpus_backend != expected_backend:
            raise ValueError(
                f"Model retrieval metric {retrieval_metric!r} requires a {expected_backend!r} corpus, "
                f"found {corpus_backend!r}"
            )
        corpus_k = int(self.metadata["num_retrieved"])
        if num_context_chunks != corpus_k:
            raise ValueError(f"Model requests {num_context_chunks} chunks, corpus contains {corpus_k}")
        self.num_context_chunks = num_context_chunks
        self.context_chunk_size = context_chunk_size
        self.context_frames_per_chunk = context_frames_per_chunk
        self._tasks = {int(task["task_id"]): task for task in self.metadata["tasks"]}
        self._neighbors: dict[str, np.ndarray] = {}
        self._refs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __getitem__(self, index: SupportsIndex) -> dict[str, object]:
        sample_index = index.__index__()
        if sample_index < 0:
            sample_index += len(self)
        if not 0 <= sample_index < len(self):
            raise IndexError(sample_index)
        episode_index = bisect.bisect_right(self._episode_ends, sample_index)
        episode_start = 0 if episode_index == 0 else self._episode_ends[episode_index - 1]
        step_idx = sample_index - episode_start
        episode = self._episodes[episode_index]
        task = self._tasks[episode.task_id]
        neighbors_dir = task.get("neighbors_dir")
        if neighbors_dir is None:
            raise ValueError(f"Training task {episode.task_id} has no precomputed VICTR neighbors")
        neighbors_path = str(self.root / neighbors_dir / f"{episode.demo_id}.npz")
        if neighbors_path not in self._neighbors:
            with np.load(neighbors_path, allow_pickle=False) as payload:
                self._neighbors[neighbors_path] = np.asarray(payload["retrieved_bank_indices"], dtype=np.int32)
        bank_indices = self._neighbors[neighbors_path][step_idx]
        if bank_indices.shape != (self.num_context_chunks,):
            raise ValueError(f"Unexpected VICTR neighbor shape: {bank_indices.shape}")

        if episode.task_id not in self._refs:
            with np.load(self.root / task["context_refs_path"], allow_pickle=False) as payload:
                self._refs[episode.task_id] = (
                    np.asarray(payload["demo_indices"], dtype=np.int32),
                    np.asarray(payload["step_indices"], dtype=np.int32),
                )
        demo_indices, context_steps = self._refs[episode.task_id]
        selected_demos = demo_indices[bank_indices]
        if len(np.unique(selected_demos)) != len(selected_demos):
            raise ValueError("VICTR retrieval must select at most one chunk per context demonstration")

        h5_file = self._h5(episode.source_hdf5)
        context_images, context_states, context_actions = [], [], []
        # Retrieval files are nearest-to-farthest. VICTR encodes context in the
        # reverse order so the query attends causally through increasingly relevant chunks.
        for bank_index in reversed(bank_indices.tolist()):
            demo_id = str(task["context_demo_ids"][int(demo_indices[bank_index])])
            context_step = int(context_steps[bank_index])
            prefix = f"data/{demo_id}"
            obs = h5_file[f"{prefix}/obs"]
            demo_actions = np.asarray(h5_file[f"{prefix}/actions"][:], dtype=np.float32)
            chunk_end = min(context_step + self.context_chunk_size, len(demo_actions))
            frame_indices = (
                np.linspace(
                    context_step,
                    max(context_step, chunk_end - 1),
                    self.context_frames_per_chunk,
                )
                .round()
                .astype(np.int64)
            )
            context_images.append(np.stack([flip_libero_image(obs["agentview_rgb"][frame]) for frame in frame_indices]))
            context_states.append(
                np.concatenate((obs["ee_states"][context_step], obs["gripper_states"][context_step])).astype(np.float32)
            )
            context_actions.append(libero_action_chunk(demo_actions, context_step, self.context_chunk_size))

        result = super().__getitem__(sample_index)
        result.update(
            {
                "retrieved_context_images": np.stack(context_images),
                "retrieved_context_states": np.stack(context_states),
                "retrieved_context_actions": np.stack(context_actions),
            }
        )
        return result


def iter_training_trajectories(
    corpus_dir: str | PathLike[str], *, include_context: bool
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield state/action trajectories in the same order as RICL norm-stat computation."""
    root = pathlib.Path(corpus_dir).expanduser().resolve()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    h5_files: dict[str, h5py.File] = {}
    try:
        for task in metadata["tasks"]:
            if not task["is_train"]:
                continue
            demo_ids = list(task["query_demo_ids"])
            if include_context:
                # RICL computes stats in context-then-query order.
                demo_ids = list(task["context_demo_ids"]) + demo_ids
            source_hdf5 = str(pathlib.Path(task["source_hdf5"]).expanduser().resolve())
            if source_hdf5 not in h5_files:
                h5_files[source_hdf5] = h5py.File(source_hdf5, "r")
            h5_file = h5_files[source_hdf5]
            for demo_id in demo_ids:
                prefix = f"data/{demo_id}"
                obs = h5_file[f"{prefix}/obs"]
                states = np.concatenate((obs["ee_states"][:], obs["gripper_states"][:]), axis=1).astype(np.float32)
                actions = np.asarray(h5_file[f"{prefix}/actions"][:], dtype=np.float32)
                if len(states) != len(actions):
                    raise ValueError(f"State/action length mismatch for task={task['task_id']} demo={demo_id}")
                yield states, actions
    finally:
        for h5_file in h5_files.values():
            h5_file.close()
