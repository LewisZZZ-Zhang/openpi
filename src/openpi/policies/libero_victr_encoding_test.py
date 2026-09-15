# ruff: noqa: SLF001
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.fft import dct

from openpi import transforms
from openpi.models.model import ModelType
from openpi.models.tokenizer import FASTTokenizer
from openpi.policies import libero_policy
from openpi.shared.normalize import NormStats


class _SentencePiece:
    def encode(self, text, *, add_bos=False, add_eos=False):
        return [1] * (len(text) // 4 + int(add_bos)) + ([2] if add_eos else [])

    def vocab_size(self):
        return 262144


def test_fast_strict_budget_preserves_action_tail_and_clips_outliers():
    tokenizer = FASTTokenizer.__new__(FASTTokenizer)
    tokenizer._strict = True
    tokenizer._max_len = 256
    tokenizer._fast_skip_tokens = 128
    tokenizer._paligemma_tokenizer = _SentencePiece()
    received = []

    def encode_action(value):
        received.append(value)
        return [[17, 18, 19]]

    tokenizer._fast_tokenizer = encode_action
    tokens, mask, _, loss = tokenizer.tokenize("pick object", np.zeros(8), np.full((10, 7), 4.0))
    np.testing.assert_array_equal(received[0], np.ones((1, 10, 7)))
    assert tokens[mask][-1] == 2  # EOS after the entire action segment.
    assert (262144 - 1 - 128 - 19) in tokens[mask]
    assert loss[mask][-1]
    tokenizer._max_len = int(mask.sum()) - 1
    with pytest.raises(ValueError, match="Refusing to truncate actions"):
        tokenizer.tokenize("pick object", np.zeros(8), np.zeros((10, 7)))


def test_neighbors_use_checkpoint_normalization_before_encoding(monkeypatch):
    stats = {
        "state": NormStats(mean=np.zeros(8), std=np.ones(8), q01=np.zeros(8), q99=np.full(8, 4.0)),
        "actions": NormStats(mean=np.zeros(7), std=np.ones(7), q01=np.zeros(7), q99=np.full(7, 4.0)),
    }
    data = {
        "state": np.ones(8),
        "actions": np.ones((10, 7)),
        "prompt": "pick object",
        "retrieved_context_states": np.ones((4, 8)),
        "retrieved_context_actions": np.stack([np.full((10, 7), x) for x in [0.0, 1.0, 2.0, 6.0]]),
        "exp_lamda_distance": np.float32(0.5),
    }
    normalized = transforms.Normalize(stats, use_quantiles=True)(data)
    received = []

    def tokenize(prompt, state, action):
        received.append((state, action))
        return np.zeros(256, dtype=np.int32), np.ones(256, dtype=bool)

    monkeypatch.setattr(libero_policy, "_victr_fast_tokenizer", lambda *args: SimpleNamespace(tokenize=tokenize))
    result = libero_policy.EncodeLiberoVictrContext(256, "fake", 32, use_action_interpolation=True)(normalized)
    np.testing.assert_allclose(received[0][0], -0.5, atol=1e-5)
    assert result["context_tokens"].shape == (4, 256)
    # Nearest is last. Continuous RICL must not use the clipped FAST action.
    np.testing.assert_allclose(result["nearest_action"][:, :7], 2.0, atol=1e-5)
    np.testing.assert_array_equal(result["nearest_action"][:, 7:], 0.0)


def test_context_dct_lowpass_keeps_native_horizon_and_query_actions():
    context_actions = np.random.default_rng(0).normal(size=(4, 10, 7)).astype(np.float32)
    sample = {
        **libero_policy.make_libero_example(),
        "actions": np.ones((10, 7), dtype=np.float32),
        "retrieved_context_images": np.zeros((4, 2, 224, 224, 3), dtype=np.uint8),
        "retrieved_context_states": np.zeros((4, 8), dtype=np.float32),
        "retrieved_context_actions": context_actions,
    }
    result = libero_policy.LiberoVictrInputs(ModelType.PI05, 256)(sample)
    assert result["retrieved_context_actions"].shape == (4, 10, 7)
    coefficients = dct(result["retrieved_context_actions"], axis=-2, norm="ortho")
    np.testing.assert_allclose(coefficients[:, 7:], 0.0, atol=1e-6)
    np.testing.assert_allclose(coefficients[:, :7], dct(context_actions, axis=-2, norm="ortho")[:, :7], atol=1e-6)
    np.testing.assert_array_equal(result["actions"], sample["actions"])
