import json

import pytest

from openpi.models import pi0_config
from openpi.training import config as _config


def _write_dataset(root, *, version="v2.1"):
    meta = root / "meta"
    meta.mkdir(parents=True)
    features = {
        "observation.state": {"shape": [14]},
        "action": {"shape": [16]},
        "observation.images.base_0_rgb": {"shape": [32, 32, 3]},
        "observation.images.left_wrist_0_rgb": {"shape": [32, 32, 3]},
        "observation.images.right_wrist_0_rgb": {"shape": [32, 32, 3]},
    }
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": version, "total_episodes": 3, "features": features})
    )
    (meta / "train_context_split.json").write_text(
        json.dumps({"train_output_episodes": [0, 1], "context_output_episodes": [2]})
    )


def test_icl_config_uses_local_train_split(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    factory = _config.LeRobotIclDataConfig(
        repo_id="test/icl",
        dataset_root=str(dataset_root),
        split="train",
    )

    result = factory.create(tmp_path / "assets", pi0_config.Pi0Config(action_horizon=10))

    assert result.dataset_root == str(dataset_root.resolve())
    assert result.episodes == (0, 1)
    assert result.video_backend == "pyav"
    assert result.action_sequence_keys == ("action",)
    assert result.prompt_from_task


def test_icl_config_can_select_context(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    factory = _config.LeRobotIclDataConfig(
        repo_id="test/icl",
        dataset_root=str(dataset_root),
        split="context",
    )

    result = factory.create(tmp_path / "assets", pi0_config.Pi0Config(action_horizon=10))
    assert result.episodes == (2,)


def test_icl_config_rejects_v3(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root, version="v3.0")
    factory = _config.LeRobotIclDataConfig(repo_id="test/icl", dataset_root=str(dataset_root))

    with pytest.raises(ValueError, match="requires canonical LeRobot v2.1"):
        factory.create(tmp_path / "assets", pi0_config.Pi0Config(action_horizon=10))
