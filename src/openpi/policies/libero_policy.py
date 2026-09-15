import dataclasses
import functools

import einops
import numpy as np
from scipy.fft import dct
from scipy.fft import idct

from openpi import transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import image_tools


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        for key in ("context_images", "context_image_masks", "context_tokens", "context_tokens_mask"):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoVictrInputs(transforms.DataTransformFn):
    """Convert a retrieved LIBERO sample into Pi0Victr's fixed context tensors."""

    model_type: _model.ModelType
    context_text_max_length: int
    context_action_keep_bins: int = 7

    def __call__(self, data: dict) -> dict:
        result = LiberoInputs(self.model_type)(data)
        images = np.asarray(data["retrieved_context_images"], dtype=np.uint8)
        states = np.asarray(data["retrieved_context_states"], dtype=np.float32)
        actions = np.asarray(data["retrieved_context_actions"], dtype=np.float32)
        if images.ndim != 5:
            raise ValueError(f"Expected retrieved context images [k,f,h,w,c], got {images.shape}")
        if states.shape[0] != images.shape[0] or actions.shape[0] != images.shape[0]:
            raise ValueError("Retrieved context image/state/action chunk counts do not align")

        flat_images = images.reshape((-1, *images.shape[-3:]))
        if flat_images.shape[1:3] != (224, 224):
            flat_images = np.asarray(image_tools.resize_with_pad(flat_images, 224, 224))
        context_images = flat_images.reshape((*images.shape[:2], 224, 224, 3))
        # Apply upstream's native-window DCT low-pass before query normalization.
        # The shared context encoder runs after Normalize, below.
        if not 1 <= self.context_action_keep_bins <= actions.shape[-2]:
            raise ValueError("Invalid context DCT budget for the action horizon")
        coefficients = dct(actions, axis=-2, norm="ortho")
        coefficients[..., self.context_action_keep_bins :, :] = 0
        actions = idct(coefficients, axis=-2, norm="ortho").astype(np.float32)
        result.update(
            {
                "context_images": context_images,
                "context_image_masks": np.ones(images.shape[:2], dtype=bool),
                "retrieved_context_states": states,
                "retrieved_context_actions": actions,
            }
        )
        if "exp_lamda_distance" in data:
            result["exp_lamda_distance"] = data["exp_lamda_distance"]
        return result


@functools.cache
def _victr_fast_tokenizer(path: str, max_length: int) -> _tokenizer.FASTTokenizer:
    return _tokenizer.FASTTokenizer(max_len=max_length, fast_tokenizer_path=path, strict=True)


@dataclasses.dataclass(frozen=True)
class EncodeLiberoVictrContext(transforms.DataTransformFn):
    """Shared train/rollout encoder; requires normalized neighbor states/actions."""

    max_length: int
    fast_tokenizer_path: str
    action_dim: int
    use_action_interpolation: bool = False

    def __call__(self, data: dict) -> dict:
        tokenizer = _victr_fast_tokenizer(self.fast_tokenizer_path, self.max_length)
        states = data["retrieved_context_states"]
        actions = data["retrieved_context_actions"]
        tokens = [
            tokenizer.tokenize(str(data["prompt"]), state, action)
            for state, action in zip(states, actions, strict=True)
        ]
        result = dict(data)
        result["context_tokens"] = np.stack([entry[0] for entry in tokens])
        result["context_tokens_mask"] = np.stack([entry[1] for entry in tokens])
        if self.use_action_interpolation:
            if "exp_lamda_distance" not in data:
                raise ValueError("Missing continuous RICL weight; validate retrieval preprocessing")
            # Farthest-to-nearest order: the last action is the nearest neighbor.
            # Blend continuous normalized actions, not FAST's clipped tokens.
            result["nearest_action"] = transforms.pad_to_dim(actions[-1], self.action_dim)
        return result


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][..., :7])}
