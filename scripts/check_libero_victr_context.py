"""CPU-only FAST budget preflight using the actual shared train/rollout transforms.

Run from third_party/openpi. samples-per-task=0 checks every bank entry;
the default samples all tasks (including unseen) but is not an exhaustive guarantee.
Runtime encoding always rejects overflow, even after a sampled preflight passes.
"""

import argparse
import json
import os
from pathlib import Path

# Set before importing JAX/Torch. This command must never allocate GPU memory.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from openpi import transforms
from openpi.policies.libero_victr_retrieval import LiberoVictrBank
from openpi.training import config


def check(config_name: str, corpus_dir: str | None, samples_per_task: int) -> dict:
    if samples_per_task < 0:
        raise ValueError("samples-per-task must be non-negative")
    cfg = config.get_config(config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data_cfg.norm_stats is None:
        raise ValueError("Missing training normalization statistics")
    bank = LiberoVictrBank(corpus_dir or data_cfg.libero_corpus_dir)
    if bank.metric != cfg.model.retrieval_metric:
        raise ValueError("Corpus retrieval metric does not match the model config")
    encode = transforms.compose(
        [
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ]
    )
    maxima = {}
    count = 0
    try:
        for task_id, task in bank.tasks.items():
            with np.load(bank.root / task["context_refs_path"]) as refs:
                size = len(refs["step_indices"])
                if len(np.unique(refs["demo_indices"])) < cfg.model.num_context_chunks:
                    raise ValueError(f"Task {task_id} has too few context demonstrations")
            if cfg.model.use_action_interpolation and task["is_train"]:
                path = bank.root / task["neighbors_dir"] / f"{task['query_demo_ids'][0]}.npz"
                with np.load(path) as neighbors:
                    if "retrieval_scores" not in neighbors:
                        raise ValueError(f"Rebuild raw DINO retrieval_scores: {path}")
            indices = (
                np.arange(size)
                if samples_per_task == 0
                else np.unique(np.linspace(0, size - 1, min(samples_per_task, size), dtype=int))
            )
            maximum = 0
            for index in indices:
                context = bank.context(
                    task_id,
                    np.array([index]),
                    chunk_size=cfg.model.context_chunk_size,
                    frames_per_chunk=cfg.model.context_frames_per_chunk,
                    camera_keys=cfg.model.context_camera_keys,
                )
                sample = {
                    "observation/image": context["retrieved_context_images"][0, 0],
                    "observation/wrist_image": context["retrieved_context_images"][0, 1],
                    "observation/state": context["retrieved_context_states"][0],
                    "prompt": task["prompt"],
                    **context,
                }
                if cfg.model.use_action_interpolation:
                    sample["exp_lamda_distance"] = np.float32(0.5)
                encoded = encode(sample)
                maximum = max(maximum, int(encoded["context_tokens_mask"].sum()))
                count += 1
            maxima[str(task_id)] = maximum
            print(f"task={task_id} checked={len(indices)} max_tokens={maximum}", flush=True)
    finally:
        bank.close()
    return {
        "config": config_name,
        "corpus": str(bank.root),
        "encoding": "fast_v2",
        "exhaustive": samples_per_task == 0,
        "contexts_checked": count,
        "max_tokens": max(maxima.values()),
        "budget": cfg.model.context_text_max_length,
        "per_task_max": maxima,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_victr_libero100_dino_progress")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.config, args.corpus_dir, args.samples_per_task)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
