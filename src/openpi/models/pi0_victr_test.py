# ruff: noqa: SLF001
import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models.pi0_victr import Pi0Victr
from openpi.models.pi0_victr import Pi0VictrConfig


@pytest.mark.parametrize("train", [False, True])
def test_preprocess_preserves_context_and_interpolation(train):
    config = Pi0VictrConfig(pi05=True, action_horizon=10, use_action_interpolation=True, retrieval_metric="vision")
    obs = config.fake_obs(batch_size=1)
    restored = model_lib.Observation.from_dict(obs.to_dict())
    result = model_lib.preprocess_observation(jax.random.key(0), restored, train=train)
    for field in (
        "context_images",
        "context_image_masks",
        "context_tokens",
        "context_tokens_mask",
        "exp_lamda_distance",
        "nearest_action",
    ):
        assert getattr(result, field) is not None
        np.testing.assert_array_equal(getattr(result, field), getattr(obs, field))


def test_continuous_ricl_training_and_rollout_endpoints():
    # Test the actual model methods without allocating a multi-billion-parameter model.
    stub = SimpleNamespace(action_horizon=2, action_dim=3)
    stub._check_interpolation = lambda obs: Pi0Victr._check_interpolation(stub, obs)
    truth = jnp.full((3, 2, 3), 4.0)
    nearest = jnp.full_like(truth, 2.0)
    obs = SimpleNamespace(exp_lamda_distance=jnp.array([0.0, 0.5, 1.0]), nearest_action=nearest)
    target = Pi0Victr._blended_action_target(stub, truth, obs)
    np.testing.assert_allclose(target[:, 0, 0], [4.0, 3.0, 2.0])
    velocity = Pi0Victr._blend_velocity(stub, truth, jnp.full_like(truth, 3.0), 0.5, obs)
    np.testing.assert_allclose(velocity[:, 0, 0], [4.0, 3.0, 2.0])
    # w=1 and one final Euler step exactly reaches the retrieved action.
    np.testing.assert_allclose((3.0 - 0.5 * velocity)[-1], nearest[-1])
    with pytest.raises(ValueError, match="requires"):
        Pi0Victr._blended_action_target(stub, truth, SimpleNamespace(nearest_action=None, exp_lamda_distance=None))


def test_config_defaults_and_guards():
    config = Pi0VictrConfig(pi05=True, action_horizon=10)
    assert config.num_context_chunks == 4
    assert config.context_text_max_length == 256
    assert config.context_frames_per_chunk == len(config.context_camera_keys) == 2
    assert config.action_horizon == config.context_chunk_size == 10
    with pytest.raises(ValueError, match="vision"):
        dataclasses.replace(config, retrieval_metric="vision_progress", use_action_interpolation=True)
    with pytest.raises(ValueError, match="camera"):
        dataclasses.replace(config, context_frames_per_chunk=3)


@pytest.mark.parametrize("interpolation", [False, True])
@pytest.mark.parametrize("k", [1, 4])
def test_model_loss_and_rollout_abstract_forward(interpolation, k):
    config = Pi0VictrConfig(
        pi05=True,
        action_horizon=10,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        retrieval_metric="vision",
        use_action_interpolation=interpolation,
        num_context_chunks=k,
    )

    def forward(key):
        model = config.create(key)
        obs = config.fake_obs(batch_size=1)
        return (
            model.compute_loss(key, obs, config.fake_act(batch_size=1), train=False),
            model.sample_actions(key, obs, num_steps=2),
        )

    loss, actions = jax.eval_shape(forward, jax.random.key(0))
    assert loss.shape == (1, 10)
    assert actions.shape == (1, 10, 32)


def test_k1_four_gpu_schedule_preserves_sample_milestones():
    from openpi.training.config import get_config

    old = get_config("pi05_ricl_libero100_dino")
    new = get_config("pi05_ricl_libero100_dino_k1_4gpu")
    assert new.model.num_context_chunks == 1
    assert new.model.use_action_interpolation
    assert new.batch_size == 64
    assert new.fsdp_devices == 4
    assert new.batch_size // new.fsdp_devices == old.batch_size // old.fsdp_devices == 16
    for field in ("num_train_steps", "save_interval"):
        assert getattr(new, field) * new.batch_size == getattr(old, field) * old.batch_size
    for field in ("warmup_steps", "decay_steps"):
        assert getattr(new.lr_schedule, field) * new.batch_size == getattr(old.lr_schedule, field) * old.batch_size
    for field in ("peak_lr", "decay_lr"):
        assert getattr(new.lr_schedule, field) == getattr(old.lr_schedule, field)
    for field in ("seed", "optimizer", "weight_loader", "data"):
        assert getattr(new, field) == getattr(old, field)


def test_k1_four_gpu_victr_matches_ricl_training_settings():
    from openpi.training.config import get_config

    ricl = get_config("pi05_ricl_libero100_dino_k1_4gpu")
    victr = get_config("pi05_victr_libero100_dino_progress_k1_4gpu")
    for field in dataclasses.fields(ricl):
        if field.name not in {"name", "model", "data"}:
            assert getattr(victr, field.name) == getattr(ricl, field.name), field.name
    assert victr.model.num_context_chunks == ricl.model.num_context_chunks == 1
    assert victr.model.retrieval_metric == "vision_progress"
    assert not victr.model.use_action_interpolation
    assert victr.model == dataclasses.replace(
        ricl.model, retrieval_metric="vision_progress", use_action_interpolation=False
    )
    assert victr.data.corpus_dir.endswith("victr_libero100_70_30_dino_progress")


def test_ricl_config_matches_baseline_and_warmstart_is_opt_in():
    from openpi.training.config import get_config

    baseline = get_config("pi05_libero100_seed123")
    victr = get_config("pi05_victr_libero100_dino_progress")
    ricl = get_config("pi05_ricl_libero100_dino")
    for cfg in (victr, ricl):
        for field in ("batch_size", "fsdp_devices", "num_train_steps", "seed", "lr_schedule", "optimizer"):
            assert getattr(cfg, field) == getattr(baseline, field)
        assert cfg.model.action_horizon == baseline.model.action_horizon == 10
        assert cfg.lr_schedule.warmup_init_lr is None
    assert not victr.model.use_action_interpolation
    assert ricl.model.use_action_interpolation
    assert ricl.model.retrieval_metric == "vision"
    warm = get_config("pi05_ricl_libero100_dino_warmstart")
    control = get_config("pi05_libero100_seed123_warmstart")
    assert warm.lr_schedule == control.lr_schedule
    np.testing.assert_allclose(warm.lr_schedule.create()(0), 2.5e-6)
