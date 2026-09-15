from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies.libero_victr_retrieval import LiberoVictrBank
from openpi.policies.libero_victr_retrieval import embed_dino
from openpi.policies.libero_victr_retrieval import load_dinov2
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class LiberoVictrPolicy(BasePolicy):
    """Add online DINO/progress retrieval before invoking a Pi0Victr policy."""

    def __init__(
        self,
        policy: Policy,
        *,
        corpus_dir: str,
        num_context_chunks: int,
        context_chunk_size: int,
        context_frames_per_chunk: int,
        progress_predictor: Any | None = None,
        context_camera_keys: tuple[str, ...] = ("agentview_rgb", "eye_in_hand_rgb"),
        use_action_interpolation: bool = False,
        lamda: float = 3.0,
        retrieval_metric: str | None = None,
    ) -> None:
        self._policy = policy
        self._bank = LiberoVictrBank(corpus_dir)
        if retrieval_metric is not None and self._bank.metric != retrieval_metric:
            raise ValueError(f"Corpus metric {self._bank.metric} does not match trained model {retrieval_metric}")
        if not 1 <= num_context_chunks <= int(self._bank.metadata.get("num_retrieved", num_context_chunks)):
            raise ValueError("Corpus has fewer retrieved neighbors than the trained model requires")
        self._num_context_chunks = num_context_chunks
        self._context_chunk_size = context_chunk_size
        self._context_frames_per_chunk = context_frames_per_chunk
        self._context_camera_keys = context_camera_keys
        self._use_action_interpolation = use_action_interpolation
        self._lamda = lamda
        if use_action_interpolation and self._bank.metric != "vision":
            raise ValueError("Continuous RICL interpolation requires DINO-only retrieval")
        self._progress_predictor = progress_predictor
        self._dinov2 = load_dinov2() if self._bank.metric in {"vision", "vision_progress"} else None

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if "retrieval_task_id" not in obs:
            raise KeyError("VICTR LIBERO rollout requires retrieval_task_id")
        task_id = int(np.asarray(obs["retrieval_task_id"]).item())
        query_embedding = None
        query_progress = None
        if self._dinov2 is not None:
            query_image = obs.get("observation/image", obs.get("query_top_image"))
            if query_image is None:
                raise KeyError("VICTR rollout requires observation/image or query_top_image")
            query_embedding = embed_dino(np.asarray(query_image), self._dinov2)[0]
        if self._bank.metric in {"progress", "vision_progress"}:
            if "query_progress" in obs:
                query_progress = float(np.asarray(obs["query_progress"]).item())
            elif self._progress_predictor is not None:
                query_progress = float(self._progress_predictor.predict(obs, task_id))
            else:
                raise KeyError("Progress retrieval requires query_progress or a VFE progress predictor")
            query_progress = float(np.clip(query_progress, 0.0, 1.0))
        bank_indices = self._bank.retrieve(
            task_id,
            k=self._num_context_chunks,
            query_embedding=query_embedding,
            query_progress=query_progress,
        )
        context = self._bank.context(
            task_id,
            bank_indices,
            chunk_size=self._context_chunk_size,
            frames_per_chunk=self._context_frames_per_chunk,
            camera_keys=self._context_camera_keys,
        )
        if self._use_action_interpolation:
            context["exp_lamda_distance"] = self._bank.online_interpolation_weight(
                task_id, query_embedding, self._lamda
            )
        runtime_keys = {"query_progress", "retrieval_task_id", "retrieval_episode_id", "retrieval_timestep"}
        policy_obs = {
            key: value for key, value in obs.items() if key not in runtime_keys and not key.startswith("vfe_")
        }
        # Accept both OpenPI's stock LIBERO request schema and VFE's richer
        # main_ricl.py schema so the existing VFE history/metrics evaluator is reusable.
        aliases = {
            "query_top_image": "observation/image",
            "query_wrist_image": "observation/wrist_image",
            "query_state": "observation/state",
            "query_prompt": "prompt",
        }
        for source, destination in aliases.items():
            if source in policy_obs:
                policy_obs[destination] = policy_obs.pop(source)
        return self._policy.infer({**policy_obs, **context}, noise=noise)

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._policy.metadata, "retrieval_backend": self._bank.backend}

    def close(self) -> None:
        self._bank.close()


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
