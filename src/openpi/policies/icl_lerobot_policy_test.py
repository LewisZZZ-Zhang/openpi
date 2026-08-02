import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import icl_lerobot_policy


def _sample() -> dict:
    return {
        "base_image": np.zeros((3, 8, 10), dtype=np.uint8),
        "left_wrist_image": np.ones((3, 8, 10), dtype=np.uint8),
        "right_wrist_image": np.full((3, 8, 10), 2, dtype=np.uint8),
        "state": np.arange(14, dtype=np.float32),
        "actions": np.zeros((10, 16), dtype=np.float32),
        "prompt": b"pick up the object",
    }


@pytest.mark.parametrize("model_type", [_model.ModelType.PI0, _model.ModelType.PI05])
def test_flow_model_camera_mapping(model_type):
    result = icl_lerobot_policy.IclLeRobotInputs(model_type=model_type)(_sample())

    assert tuple(result["image"]) == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert all(image.shape == (8, 10, 3) for image in result["image"].values())
    assert all(result["image_mask"].values())
    assert result["state"].shape == (14,)
    assert result["actions"].shape == (10, 16)
    assert result["prompt"] == "pick up the object"


def test_fast_camera_mapping_and_state_padding():
    result = icl_lerobot_policy.IclLeRobotInputs(model_type=_model.ModelType.PI0_FAST)(_sample())
    assert tuple(result["image"]) == ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
    np.testing.assert_array_equal(result["image"]["base_1_rgb"], np.full((8, 10, 3), 2, dtype=np.uint8))
    np.testing.assert_array_equal(result["image"]["wrist_0_rgb"], np.ones((8, 10, 3), dtype=np.uint8))

    padded = icl_lerobot_policy.PadIclState()(result)
    assert padded["state"].shape == (16,)
    np.testing.assert_array_equal(padded["state"][-2:], 0)


def test_rejects_wrong_action_width():
    sample = _sample()
    sample["actions"] = np.zeros((10, 15), dtype=np.float32)
    with pytest.raises(ValueError, match="16-D ICL actions"):
        icl_lerobot_policy.IclLeRobotInputs(model_type=_model.ModelType.PI0)(sample)


def test_output_removes_padding():
    result = icl_lerobot_policy.IclLeRobotOutputs()({"actions": np.zeros((10, 32), dtype=np.float32)})
    assert result["actions"].shape == (10, 16)
