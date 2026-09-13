"""Build LIBERO VICTR neighbors for DINO, progress, or their fusion.

This consumes the existing DINO RICL corpus and does not duplicate its HDF5
demonstrations or embedding banks. LIBERO progress is always t/(T-1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openpi.policies.libero_victr_retrieval import embed_dino
from openpi.policies.libero_victr_retrieval import linear_progress
from openpi.policies.libero_victr_retrieval import load_dinov2
from openpi.policies.libero_victr_retrieval import retrieve_indices


def _context_progress(
    h5_file: h5py.File,
    context_demo_ids: list[str],
    demo_indices: np.ndarray,
    step_indices: np.ndarray,
) -> np.ndarray:
    lengths = {
        demo_index: len(h5_file[f"data/{demo_id}/actions"]) for demo_index, demo_id in enumerate(context_demo_ids)
    }
    return np.asarray(
        [
            1.0 if lengths[int(demo)] == 1 else int(step) / (lengths[int(demo)] - 1)
            for demo, step in zip(demo_indices, step_indices, strict=True)
        ],
        dtype=np.float32,
    )


def build(
    dino_corpus_dir: Path,
    output_dir: Path,
    embedding_batch_size: int,
    retrieval_metric: str = "vision_progress",
    query_embeddings_dir: Path | None = None,
) -> None:
    dino_corpus_dir = dino_corpus_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    query_embeddings_dir = query_embeddings_dir.expanduser().resolve() if query_embeddings_dir is not None else None
    source = json.loads((dino_corpus_dir / "metadata.json").read_text(encoding="utf-8"))
    if str(source.get("retrieval_backend", "dino")) != "dino":
        raise ValueError("--dino-corpus-dir must point to a DINO corpus")
    if source.get("embedding_type") != "CLS":
        raise ValueError("VICTR uses DINOv2 CLS embeddings; the source corpus must use embedding_type='CLS'")
    k = int(source["num_retrieved"])
    if retrieval_metric not in {"vision", "progress", "vision_progress"}:
        raise ValueError(f"Unsupported retrieval metric: {retrieval_metric}")
    backend = {
        "vision": "dino",
        "progress": "progress",
        "vision_progress": "dino_progress",
    }[retrieval_metric]
    output_dir.mkdir(parents=True, exist_ok=True)
    neighbors_root = output_dir / "neighbors"
    progress_root = output_dir / "progress"
    refs_root = output_dir / "refs"
    neighbors_root.mkdir(exist_ok=True)
    progress_root.mkdir(exist_ok=True)
    refs_root.mkdir(exist_ok=True)
    if retrieval_metric in {"vision", "vision_progress"}:
        if query_embeddings_dir is None:
            raise ValueError("DINO retrieval requires --query-embeddings-dir")
        query_embeddings_dir.mkdir(parents=True, exist_ok=True)
    model = None

    output_tasks = []
    for position, source_task in enumerate(source["tasks"], start=1):
        task = dict(source_task)
        task_id = int(task["task_id"])
        embeddings_path = (dino_corpus_dir / task["context_embeddings_path"]).resolve()
        context_embeddings = (
            np.asarray(np.load(embeddings_path, mmap_mode="r"), dtype=np.float32)
            if retrieval_metric in {"vision", "vision_progress"}
            else None
        )
        with np.load(dino_corpus_dir / task["context_refs_path"], allow_pickle=False) as payload:
            demo_indices = np.asarray(payload["demo_indices"], dtype=np.int32)
            step_indices = np.asarray(payload["step_indices"], dtype=np.int32)

        task_refs_dir = refs_root / f"task_{task_id:03d}"
        task_progress_dir = progress_root / f"task_{task_id:03d}"
        task_refs_dir.mkdir(exist_ok=True)
        task_progress_dir.mkdir(exist_ok=True)
        refs_path = task_refs_dir / "context_refs.npz"
        progress_path = task_progress_dir / "context_progress.npy"
        np.savez_compressed(refs_path, demo_indices=demo_indices, step_indices=step_indices)

        source_hdf5 = Path(task["source_hdf5"]).expanduser().resolve()
        neighbors_rel = None
        with h5py.File(source_hdf5, "r") as h5_file:
            progress = _context_progress(
                h5_file,
                [str(value) for value in task["context_demo_ids"]],
                demo_indices,
                step_indices,
            )
            np.save(progress_path, progress)
            if task["is_train"]:
                task_neighbors = neighbors_root / f"task_{task_id:03d}"
                task_neighbors.mkdir(exist_ok=True)
                task_query_embeddings = (
                    query_embeddings_dir / f"task_{task_id:03d}" if query_embeddings_dir is not None else None
                )
                if task_query_embeddings is not None:
                    task_query_embeddings.mkdir(exist_ok=True)
                for demo_id in task["query_demo_ids"]:
                    images = h5_file[f"data/{demo_id}/obs/agentview_rgb"][:]
                    query_embeddings = None
                    if task_query_embeddings is not None:
                        cache_path = task_query_embeddings / f"{demo_id}.npy"
                        if cache_path.exists():
                            query_embeddings = np.asarray(np.load(cache_path, mmap_mode="r"), dtype=np.float32)
                            if query_embeddings.shape != (len(images), int(source["embedding_dim"])):
                                raise ValueError(
                                    f"Unexpected cached query embedding shape at {cache_path}: "
                                    f"{query_embeddings.shape}"
                                )
                        else:
                            if model is None:
                                model = load_dinov2()
                            query_embeddings = embed_dino(
                                images,
                                model,
                                batch_size=embedding_batch_size,
                                flip_libero=True,
                            )
                            np.save(cache_path, query_embeddings)
                    query_progress = linear_progress(len(images))
                    selected = np.empty((len(images), k), dtype=np.int32)
                    scores = np.empty((len(images), k), dtype=np.float32)
                    for frame in range(len(images)):
                        selected[frame], scores[frame] = retrieve_indices(
                            metric=retrieval_metric,
                            context_demo_indices=demo_indices,
                            k=k,
                            context_embeddings=context_embeddings,
                            query_embedding=query_embeddings[frame] if query_embeddings is not None else None,
                            context_progress=progress,
                            query_progress=float(query_progress[frame]),
                        )
                    np.savez_compressed(
                        task_neighbors / f"{demo_id}.npz",
                        retrieved_bank_indices=selected,
                        retrieval_scores=scores,
                        query_progress=query_progress,
                    )
                neighbors_rel = str(task_neighbors.relative_to(output_dir))

        task.update(
            {
                "context_embeddings_path": str(embeddings_path),
                "context_progress_path": str(progress_path.relative_to(output_dir)),
                "context_refs_path": str(refs_path.relative_to(output_dir)),
                "neighbors_dir": neighbors_rel,
            }
        )
        output_tasks.append(task)
        print(f"[{position}/{len(source['tasks'])}] task={task_id} backend={backend}")

    metadata = {
        **{key: value for key, value in source.items() if key != "tasks"},
        "format_version": 3,
        "retrieval_backend": backend,
        "progress_semantics": "per_demo_relative_progress_v1",
        "progress_range": [0.0, 1.0],
        "tasks": output_tasks,
    }
    if retrieval_metric == "vision_progress":
        metadata["fusion"] = {
            "method": "per_query_zscore_sum_v1",
            "epsilon": 1e-8,
            "vision_weight": 1.0,
            "progress_weight": 1.0,
        }
    if query_embeddings_dir is not None:
        metadata["query_embeddings_dir"] = str(query_embeddings_dir)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--query-embeddings-dir",
        type=Path,
        help="Persistent per-query DINO cache shared by the vision and fused corpus builds.",
    )
    parser.add_argument(
        "--retrieval-metric",
        choices=("vision", "progress", "vision_progress"),
        default="vision_progress",
    )
    args = parser.parse_args()
    build(
        args.dino_corpus_dir,
        args.output_dir,
        args.embedding_batch_size,
        args.retrieval_metric,
        args.query_embeddings_dir,
    )


if __name__ == "__main__":
    main()
