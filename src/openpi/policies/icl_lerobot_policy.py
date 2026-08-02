"""Input and output transforms for the canonical LeRobot ICL dataset."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 14
ACTION_DIM = 16


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a rank-3 image, got {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class IclLeRobotInputs(transforms.DataTransformFn):
    """Map the canonical three-camera ICL sample to OpenPI's observation schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}-D ICL state, got {state.shape}")

        base_image = _parse_image(data["base_image"])
        left_wrist_image = _parse_image(data["left_wrist_image"])
        right_wrist_image = _parse_image(data["right_wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, left_wrist_image, right_wrist_image)
            case _model.ModelType.PI0_FAST:
                # FAST was pretrained with two scene-camera slots and one wrist-camera slot.
                # Preserve all three ICL views by assigning the right wrist to the second
                # scene slot, matching the local RICL pipeline's ordering.
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, right_wrist_image, left_wrist_image)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict.fromkeys(names, np.True_),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"Expected {ACTION_DIM}-D ICL actions, got {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class PadIclState(transforms.DataTransformFn):
    """Pad the 14-D state after normalization for the 16-D FAST tokenizer."""

    action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        data["state"] = transforms.pad_to_dim(data["state"], self.action_dim)
        return data


@dataclasses.dataclass(frozen=True)
class IclLeRobotOutputs(transforms.DataTransformFn):
    """Remove OpenPI's action padding at inference time."""

    action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
