"""VICTR: pi0/pi0.5 with retrieved demonstration chunks as causal context."""

from __future__ import annotations

import dataclasses
from typing import Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


class Pi0Victr(pi0.Pi0):
    """Pi0/Pi0.5 whose prefix is preceded by retrieved image/text blocks.

    Every retrieved chunk starts a new causal block and receives a learned rank
    embedding. Chunks are supplied farthest-to-nearest; the query can attend to
    every chunk, while context chunks cannot attend to the query.
    """

    def __init__(self, config: Pi0VictrConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.victr_config = config
        self.neighbor_rank_embedding = nnx.Embed(max(config.num_context_chunks, 1), _gemma_width(config), rngs=rngs)

    @at.typecheck
    def embed_context_chunks(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        if (
            obs.context_images is None
            or obs.context_image_masks is None
            or obs.context_tokens is None
            or obs.context_tokens_mask is None
        ):
            raise ValueError("VICTR context fields must either all be present or all be absent")

        all_tokens, all_masks, all_ar = [], [], []
        for rank in range(obs.context_images.shape[1]):
            tokens, masks = [], []
            ar_mask: list[bool] = []
            for frame in range(obs.context_images.shape[2]):
                image_tokens, _ = self.PaliGemma.img(obs.context_images[:, rank, frame], train=False)
                tokens.append(image_tokens)
                masks.append(
                    einops.repeat(
                        obs.context_image_masks[:, rank, frame],
                        "b -> b s",
                        s=image_tokens.shape[1],
                    )
                )
                ar_mask.extend([False] * image_tokens.shape[1])

            text_tokens = self.PaliGemma.llm(obs.context_tokens[:, rank], method="embed")
            tokens.append(text_tokens)
            masks.append(obs.context_tokens_mask[:, rank])
            ar_mask.extend([False] * text_tokens.shape[1])

            block_tokens = jnp.concatenate(tokens, axis=1)
            rank_embedding = self.neighbor_rank_embedding(jnp.asarray(rank)).reshape(-1)
            block_tokens = block_tokens + rank_embedding[None, None, :].astype(block_tokens.dtype)
            block_ar = jnp.asarray(ar_mask).at[0].set(True)
            all_tokens.append(block_tokens)
            all_masks.append(jnp.concatenate(masks, axis=1))
            all_ar.append(block_ar)

        return (
            jnp.concatenate(all_tokens, axis=1),
            jnp.concatenate(all_masks, axis=1),
            jnp.concatenate(all_ar, axis=0),
        )

    @at.typecheck
    def embed_prefix_with_context(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        query_tokens, query_mask, query_ar = self.embed_prefix(obs)
        if obs.context_images is None:
            raise ValueError("Pi0Victr requires retrieved context; use Pi0Config for the no-context baseline")
        query_ar = query_ar.at[0].set(True)
        context_tokens, context_mask, context_ar = self.embed_context_chunks(obs)
        return (
            jnp.concatenate((context_tokens, query_tokens), axis=1),
            jnp.concatenate((context_mask, query_mask), axis=1),
            jnp.concatenate((context_ar, query_ar), axis=0),
        )

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        if self.victr_config.use_action_interpolation:
            actions = self._blended_action_target(actions, observation)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix_with_context(observation)
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
        ar_mask = jnp.concatenate((prefix_ar, suffix_ar), axis=0)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=pi0.make_attn_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _blended_action_target(self, actions, observation):
        """Upstream 2851160: train the flow on a convex blend of true/neighbor actions."""
        self._check_interpolation(observation)
        weight = observation.exp_lamda_distance[..., None, None]
        return (1 - weight) * actions + weight * observation.nearest_action

    def _blend_velocity(self, velocity, x_t, time, observation):
        self._check_interpolation(observation)
        weight = observation.exp_lamda_distance[..., None, None]
        return (1 - weight) * velocity + weight * (x_t - observation.nearest_action) / time

    def _check_interpolation(self, observation):
        if observation.exp_lamda_distance is None or observation.nearest_action is None:
            raise ValueError("RICL continuous interpolation requires exp_lamda_distance and nearest_action")
        if observation.nearest_action.shape[-2:] != (self.action_horizon, self.action_dim):
            raise ValueError("nearest_action must match the model action horizon and padded action dimension")

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix_with_context(observation)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=pi0.make_attn_mask(prefix_mask, prefix_ar),
            positions=positions,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attention = pi0.make_attn_mask(suffix_mask, suffix_ar)
            prefix_attention = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attention = jnp.concatenate((prefix_attention, suffix_attention), axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attention,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if self.victr_config.use_action_interpolation:
                v_t = self._blend_velocity(v_t, x_t, time, observation)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        return jax.lax.while_loop(cond, step, (noise, 1.0))[0]


def _gemma_width(config: pi0_config.Pi0Config) -> int:
    from openpi.models import gemma

    return gemma.get_config(config.paligemma_variant).width


@dataclasses.dataclass(frozen=True)
class Pi0VictrConfig(pi0_config.Pi0Config):
    num_context_chunks: int = 4
    context_chunk_size: int = 10
    context_frames_per_chunk: int = 2
    context_camera_keys: tuple[str, ...] = ("agentview_rgb", "eye_in_hand_rgb")
    context_text_max_length: int = 256
    context_fast_tokenizer_path: str = "physical-intelligence/fast"
    # LIBERO uses native 10-step actions; retain 7 DCT bins (upstream: 20/30).
    context_action_keep_bins: int = 7
    retrieval_metric: Literal["vision", "progress", "vision_progress"] = "vision_progress"
    use_action_interpolation: bool = False
    lamda: float = 3.0

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("VICTR LIBERO is implemented for pi0.5; set pi05=True")
        if self.num_context_chunks < 1:
            raise ValueError("num_context_chunks must be positive")
        if self.context_chunk_size < 1 or self.context_frames_per_chunk < 1:
            raise ValueError("context chunk/frame counts must be positive")
        if self.context_chunk_size != self.action_horizon:
            raise ValueError("VICTR context must use the native model action horizon")
        if len(self.context_camera_keys) != self.context_frames_per_chunk:
            raise ValueError("One context image slot is required per camera")
        if not 1 <= self.context_action_keep_bins <= self.action_horizon:
            raise ValueError("context_action_keep_bins must be within the native action horizon")
        if self.context_text_max_length < 1 or not self.context_fast_tokenizer_path:
            raise ValueError("A positive token budget and FAST tokenizer are required")
        if self.use_action_interpolation and self.retrieval_metric != "vision":
            raise ValueError("Upstream continuous RICL interpolation requires retrieval_metric='vision'")
        if not 0 <= self.lamda < float("inf"):
            raise ValueError("lamda must be finite and non-negative")

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0Victr:
        return Pi0Victr(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation = dataclasses.replace(
                observation,
                context_images=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.num_context_chunks,
                        self.context_frames_per_chunk,
                        *_model.IMAGE_RESOLUTION,
                        3,
                    ],
                    jnp.float32,
                ),
                context_image_masks=jax.ShapeDtypeStruct(
                    [batch_size, self.num_context_chunks, self.context_frames_per_chunk], bool
                ),
                context_tokens=jax.ShapeDtypeStruct(
                    [batch_size, self.num_context_chunks, self.context_text_max_length], jnp.int32
                ),
                context_tokens_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.num_context_chunks, self.context_text_max_length], bool
                ),
            )
        if self.use_action_interpolation:
            with at.disable_typechecking():
                observation = dataclasses.replace(
                    observation,
                    exp_lamda_distance=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                    nearest_action=jax.ShapeDtypeStruct(
                        [batch_size, self.action_horizon, self.action_dim], jnp.float32
                    ),
                )
        return observation, actions
