from __future__ import annotations

import json

import h5py
import numpy as np

from openpi.training.libero_manifest_dataset import LiberoManifestDataset
from openpi.training.libero_manifest_dataset import libero_action_chunk


def _write_demo(h5_file: h5py.File, demo_id: str, *, length: int, offset: int) -> None:
    demo = h5_file.create_group(f"data/{demo_id}")
    obs = demo.create_group("obs")
    image = np.arange(length * 2 * 3 * 3, dtype=np.uint8).reshape(length, 2, 3, 3) + offset
    obs.create_dataset("agentview_rgb", data=image)
    obs.create_dataset("eye_in_hand_rgb", data=image + 1)
    obs.create_dataset("ee_states", data=np.full((length, 6), offset, dtype=np.float32))
    obs.create_dataset("gripper_states", data=np.full((length, 2), offset + 1, dtype=np.float32))
    actions = np.arange(length * 7, dtype=np.float32).reshape(length, 7) + offset
    demo.create_dataset("actions", data=actions)


def test_dataset_uses_only_training_query_demos(tmp_path) -> None:
    hdf5_path = tmp_path / "task.hdf5"
    with h5py.File(hdf5_path, "w") as h5_file:
        _write_demo(h5_file, "demo_context", length=2, offset=10)
        _write_demo(h5_file, "demo_query", length=3, offset=20)
        _write_demo(h5_file, "demo_unseen", length=4, offset=30)

    metadata = {
        "format_version": 1,
        "action_horizon": 2,
        "tasks": [
            {
                "task_id": 0,
                "is_train": True,
                "prompt": "training task",
                "source_hdf5": str(hdf5_path),
                "context_demo_ids": ["demo_context"],
                "query_demo_ids": ["demo_query"],
            },
            {
                "task_id": 1,
                "is_train": False,
                "prompt": "unseen task",
                "source_hdf5": str(hdf5_path),
                "context_demo_ids": [],
                "query_demo_ids": ["demo_unseen"],
            },
        ],
    }
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    dataset = LiberoManifestDataset(corpus_dir, action_horizon=2)

    assert len(dataset) == 3
    first = dataset[0]
    assert first["prompt"] == "training task"
    np.testing.assert_array_equal(first["state"], np.array([20] * 6 + [21] * 2, dtype=np.float32))
    assert np.asarray(first["actions"]).shape == (2, 7)
    raw_first_image = np.arange(3 * 2 * 3 * 3, dtype=np.uint8).reshape(3, 2, 3, 3)[0] + 20
    np.testing.assert_array_equal(first["image"], raw_first_image[::-1, ::-1])


def test_action_chunk_padding_preserves_gripper() -> None:
    actions = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
    chunk = libero_action_chunk(actions, step_idx=2, action_horizon=3)

    np.testing.assert_array_equal(chunk[0], actions[2])
    np.testing.assert_array_equal(chunk[1:, :-1], np.zeros((2, 6), dtype=np.float32))
    np.testing.assert_array_equal(chunk[1:, -1], np.full(2, actions[-1, -1], dtype=np.float32))
