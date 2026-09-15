"""VICTR retrieval math for LIBERO.

The progress coordinate is deliberately the dataset-independent linear value
``t / (T - 1)`` in ``[0, 1]``. VFE predicts that same coordinate at rollout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

RetrievalMetric = Literal["vision", "progress", "vision_progress"]
FUSION_EPSILON = 1e-8
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dinov2() -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad=False)
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def embed_dino(
    images: np.ndarray,
    model: torch.nn.Module,
    *,
    batch_size: int = 256,
    flip_libero: bool = False,
) -> np.ndarray:
    """Embed HWC uint8 images with the exact DINOv2 preprocessing used by VIKTR."""
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    device = next(model.parameters()).device
    mean = _IMAGENET_MEAN.to(device)
    std = _IMAGENET_STD.to(device)
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        if flip_libero:
            batch = batch[:, ::-1, ::-1]
        batch = np.ascontiguousarray(batch)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device) / 255.0
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        tensor = (tensor - mean) / std
        outputs.append(model.forward_features(tensor)["x_norm_clstoken"].float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def linear_progress(length: int) -> np.ndarray:
    """Per-demonstration linear progress, including exact endpoints."""
    if length <= 0:
        raise ValueError("A demonstration must contain at least one frame")
    if length == 1:
        return np.ones(1, dtype=np.float32)
    return np.linspace(0.0, 1.0, length, dtype=np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + FUSION_EPSILON)


def retrieve_indices(
    *,
    metric: RetrievalMetric,
    context_demo_indices: np.ndarray,
    k: int,
    context_embeddings: np.ndarray | None = None,
    query_embedding: np.ndarray | None = None,
    context_progress: np.ndarray | None = None,
    query_progress: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-to-farthest bank indices and their retrieval scores.

    ``vision_progress`` matches VIKTR's fixed fusion: per-query z-score the raw
    DINO L2 distances and absolute progress distances, then sum them. At most
    one frame is returned from each context demonstration for every metric.
    """
    demo_indices = np.asarray(context_demo_indices, dtype=np.int32).reshape(-1)
    if metric not in {"vision", "progress", "vision_progress"}:
        raise ValueError(f"Unsupported retrieval metric: {metric!r}")
    if not 0 < k <= len(np.unique(demo_indices)):
        raise ValueError(f"k={k} exceeds the number of context demonstrations")

    vision_distances = None
    if metric in {"vision", "vision_progress"}:
        if context_embeddings is None or query_embedding is None:
            raise ValueError(f"{metric} retrieval requires DINO embeddings")
        embeddings = np.asarray(context_embeddings, dtype=np.float32)
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if embeddings.ndim != 2 or embeddings.shape != (len(demo_indices), len(query)):
            raise ValueError("Context/query DINO embedding shapes do not align")
        vision_distances = np.linalg.norm(embeddings - query[None], axis=1)

    progress_distances = None
    if metric in {"progress", "vision_progress"}:
        if context_progress is None or query_progress is None:
            raise ValueError(f"{metric} retrieval requires progress values")
        progress = np.asarray(context_progress, dtype=np.float32).reshape(-1)
        if progress.shape != demo_indices.shape or not np.isfinite(progress).all():
            raise ValueError("Context progress does not align with the retrieval bank")
        if not 0.0 <= float(query_progress) <= 1.0:
            raise ValueError("Query progress must lie in [0, 1]")
        progress_distances = np.abs(progress - float(query_progress))

    if metric == "vision":
        scores = vision_distances
    elif metric == "progress":
        scores = progress_distances
    else:
        scores = _zscore(vision_distances) + _zscore(progress_distances)

    order = np.argsort(scores, kind="stable")
    selected: list[int] = []
    seen_demos: set[int] = set()
    for raw_index in order:
        index = int(raw_index)
        demo_index = int(demo_indices[index])
        if demo_index in seen_demos:
            continue
        selected.append(index)
        seen_demos.add(demo_index)
        if len(selected) == k:
            break
    result = np.asarray(selected, dtype=np.int32)
    return result, np.asarray(scores[result], dtype=np.float32)


class LiberoVictrBank:
    """Lazy, read-only LIBERO retrieval bank shared by rollout policies."""

    def __init__(self, corpus_dir: str | Path):
        self.root = Path(corpus_dir).expanduser().resolve()
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.backend = str(self.metadata.get("retrieval_backend", "dino"))
        self.metric: RetrievalMetric = {
            "dino": "vision",
            "progress": "progress",
            "dino_progress": "vision_progress",
        }[self.backend]
        if (
            self.metric in {"progress", "vision_progress"}
            and self.metadata.get("progress_semantics") != "per_demo_relative_progress_v1"
        ):
            raise ValueError("LIBERO VICTR requires linear per-demo progress")
        self.tasks = {int(task["task_id"]): task for task in self.metadata["tasks"]}
        self._banks: dict[int, dict[str, np.ndarray]] = {}
        self._h5_files: dict[str, h5py.File] = {}
        self._max_distances: dict[int, float] = {}

    def _task_bank(self, task_id: int) -> dict[str, np.ndarray]:
        if task_id not in self._banks:
            task = self.tasks[task_id]
            with np.load(self.root / task["context_refs_path"], allow_pickle=False) as payload:
                bank = {
                    "demo_indices": np.asarray(payload["demo_indices"], dtype=np.int32),
                    "step_indices": np.asarray(payload["step_indices"], dtype=np.int32),
                }
            if self.metric in {"vision", "vision_progress"}:
                bank["embeddings"] = np.asarray(
                    np.load(self.root / task["context_embeddings_path"], mmap_mode="r"), dtype=np.float32
                )
            if self.metric in {"progress", "vision_progress"}:
                bank["progress"] = np.asarray(
                    np.load(self.root / task["context_progress_path"], mmap_mode="r"), dtype=np.float32
                )
            self._banks[task_id] = bank
        return self._banks[task_id]

    def retrieve(
        self,
        task_id: int,
        *,
        k: int,
        query_embedding: np.ndarray | None,
        query_progress: float | None,
    ) -> np.ndarray:
        bank = self._task_bank(int(task_id))
        indices, _ = retrieve_indices(
            metric=self.metric,
            context_demo_indices=bank["demo_indices"],
            k=k,
            context_embeddings=bank.get("embeddings"),
            query_embedding=query_embedding,
            context_progress=bank.get("progress"),
            query_progress=query_progress,
        )
        return indices

    def context(
        self,
        task_id: int,
        bank_indices: np.ndarray,
        *,
        chunk_size: int,
        frames_per_chunk: int,
        camera_keys: tuple[str, ...] = ("agentview_rgb", "eye_in_hand_rgb"),
    ) -> dict[str, np.ndarray]:
        if frames_per_chunk != len(camera_keys):
            raise ValueError("VICTR context slots must match camera count")
        task = self.tasks[int(task_id)]
        bank = self._task_bank(int(task_id))
        source = str(Path(task["source_hdf5"]).expanduser().resolve())
        if source not in self._h5_files:
            self._h5_files[source] = h5py.File(source, "r")
        h5_file = self._h5_files[source]
        images, states, actions = [], [], []
        for raw_index in reversed(np.asarray(bank_indices).tolist()):
            demo_id = str(task["context_demo_ids"][int(bank["demo_indices"][raw_index])])
            step = int(bank["step_indices"][raw_index])
            prefix = f"data/{demo_id}"
            obs = h5_file[f"{prefix}/obs"]
            demo_actions = np.asarray(h5_file[f"{prefix}/actions"][:], dtype=np.float32)
            end = min(step + chunk_size, len(demo_actions))
            images.append(np.stack([np.ascontiguousarray(obs[camera][step][::-1, ::-1]) for camera in camera_keys]))
            states.append(np.concatenate((obs["ee_states"][step], obs["gripper_states"][step])).astype(np.float32))
            chunk = demo_actions[step:end]
            if len(chunk) < chunk_size:
                padding = np.zeros((chunk_size - len(chunk), 7), dtype=np.float32)
                padding[:, -1] = demo_actions[-1, -1]
                chunk = np.concatenate((chunk, padding))
            actions.append(chunk)
        return {
            "retrieved_context_images": np.stack(images),
            "retrieved_context_states": np.stack(states),
            "retrieved_context_actions": np.stack(actions),
        }

    def interpolation_weight(self, task_id: int, top1_distance: float, lamda: float) -> np.ndarray:
        """Upstream RICL: exp(-lamda * clip(d_top1 / max_pairwise_DINO, 0, 1))."""
        if self.metric != "vision":
            raise ValueError("Continuous RICL interpolation requires DINO-only retrieval")
        if not np.isfinite(top1_distance) or top1_distance < 0 or not np.isfinite(lamda) or lamda < 0:
            raise ValueError("RICL distance and lamda must be finite and non-negative")
        if task_id not in self._max_distances:
            scale = self.tasks[task_id].get("max_pairwise_dino_distance")
            if scale is None:
                scale = max_pairwise_distance(self._task_bank(task_id)["embeddings"])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid precomputed DINO distance normalization")
            self._max_distances[task_id] = float(scale)
        normalized = np.clip(top1_distance / self._max_distances[task_id], 0.0, 1.0)
        return np.asarray(np.exp(-lamda * normalized), dtype=np.float32)

    def online_interpolation_weight(self, task_id: int, query_embedding: np.ndarray, lamda: float) -> np.ndarray:
        distances = np.linalg.norm(self._task_bank(task_id)["embeddings"] - query_embedding[None], axis=1)
        return self.interpolation_weight(task_id, float(distances.min()), lamda)

    def close(self) -> None:
        for h5_file in self._h5_files.values():
            h5_file.close()
        self._h5_files.clear()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_h5_files"] = {}
        return state


def max_pairwise_distance(embeddings: np.ndarray, block_size: int = 256) -> float:
    """Exact upstream normalization with bounded memory instead of an N x N x D temporary."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) == 0 or not np.isfinite(embeddings).all():
        raise ValueError("Expected a nonempty finite DINO embedding bank")
    norms = np.sum(embeddings**2, axis=1)
    maximum = 0.0
    for start in range(0, len(embeddings), block_size):
        end = min(start + block_size, len(embeddings))
        squared = norms[start:end, None] + norms[None, :] - 2 * embeddings[start:end] @ embeddings.T
        maximum = max(maximum, float(squared.max()))
    return max(float(np.sqrt(maximum)), 1e-6)
