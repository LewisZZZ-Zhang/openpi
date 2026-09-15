from __future__ import annotations

import json
import pickle

import h5py
import numpy as np
import pytest

from openpi.policies.libero_victr_retrieval import LiberoVictrBank
from openpi.policies.libero_victr_retrieval import max_pairwise_distance
from openpi.training.libero_manifest_dataset import LiberoManifestDataset
from openpi.training.libero_manifest_dataset import LiberoVictrDataset
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


@pytest.mark.parametrize("k", [1, 2])
def test_victr_train_rollout_context_and_ricl_weights_match(tmp_path, k):
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as h5:
        for demo_id, offset in [("near", 10), ("far", 30), ("query", 50)]:
            _write_demo(h5, demo_id, length=3, offset=offset)
    np.savez(tmp_path / "refs.npz", demo_indices=[0, 1], step_indices=[2, 0])
    np.save(tmp_path / "embeddings.npy", np.array([[0.0], [2.0]], dtype=np.float32))
    (tmp_path / "neighbors").mkdir()
    np.savez(tmp_path / "neighbors/query.npz", retrieved_bank_indices=[[0, 1]] * 3, retrieval_scores=[[0.5, 1.5]] * 3)
    task = {
        "task_id": 0,
        "is_train": True,
        "prompt": "pick object",
        "source_hdf5": str(path),
        "context_demo_ids": ["near", "far"],
        "query_demo_ids": ["query"],
        "context_refs_path": "refs.npz",
        "context_embeddings_path": "embeddings.npy",
        "neighbors_dir": "neighbors",
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {"format_version": 3, "action_horizon": 2, "retrieval_backend": "dino", "num_retrieved": 2, "tasks": [task]}
        )
    )
    dataset = LiberoVictrDataset(
        tmp_path,
        action_horizon=2,
        num_context_chunks=k,
        context_chunk_size=2,
        context_frames_per_chunk=2,
        retrieval_metric="vision",
        use_action_interpolation=True,
    )
    bank = LiberoVictrBank(tmp_path)
    query = np.array([0.5], dtype=np.float32)
    indices = bank.retrieve(0, k=k, query_embedding=query, query_progress=None)
    offline = dataset[0]
    online = bank.context(0, indices, chunk_size=2, frames_per_chunk=2)
    for key, value in online.items():
        np.testing.assert_array_equal(offline[key], value)
    np.testing.assert_allclose(offline["exp_lamda_distance"], np.exp(-3 * 0.5 / 2))
    np.testing.assert_allclose(offline["exp_lamda_distance"], bank.online_interpolation_weight(0, query, 3.0))
    with h5py.File(path) as h5:
        for camera_slot, key in enumerate(("agentview_rgb", "eye_in_hand_rgb")):
            np.testing.assert_array_equal(
                online["retrieved_context_images"][-1, camera_slot], h5[f"data/near/obs/{key}"][2][::-1, ::-1]
            )
        np.testing.assert_array_equal(online["retrieved_context_actions"][-1, 0], h5["data/near/actions"][2])
    np.testing.assert_array_equal(online["retrieved_context_actions"][-1, 1, :6], 0.0)
    cloned = pickle.loads(pickle.dumps(dataset))
    np.testing.assert_array_equal(cloned[0]["retrieved_context_actions"], online["retrieved_context_actions"])
    cloned.close()
    dataset.close()
    bank.close()


def test_blocked_pairwise_distance_matches_bruteforce():
    values = np.random.default_rng(0).normal(size=(20, 8))
    expected = np.linalg.norm(values[:, None] - values[None, :], axis=-1).max()
    np.testing.assert_allclose(max_pairwise_distance(values, block_size=3), expected)
    assert max_pairwise_distance(np.zeros((2, 8))) == 1e-6
