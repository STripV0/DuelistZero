"""Tests for the three-segment CardEmbeddingExtractor."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from duelist_zero.network.extractor import CardEmbeddingExtractor


@pytest.fixture
def obs_space():
    """Observation space matching GoatEnv defaults."""
    return spaces.Dict({
        "features": spaces.Box(low=-1, high=1, shape=(453,), dtype=np.float32),
        "card_ids": spaces.Box(low=0, high=14336, shape=(50,), dtype=np.float32),
        "action_features": spaces.Box(low=0, high=14336, shape=(71, 12), dtype=np.float32),
        "action_history": spaces.Box(low=0, high=14336, shape=(16, 10), dtype=np.float32),
    })


@pytest.fixture
def sample_obs(obs_space):
    """Batch of 4 random observations."""
    return {
        "features": torch.randn(4, 453),
        "card_ids": torch.randint(0, 100, (4, 50)).float(),
        "action_features": _make_action_features(4),
        "action_history": _make_action_history(4),
    }


def _make_action_features(batch_size: int) -> torch.Tensor:
    """Create realistic action_features: col 0 = card_id, cols 1-11 = continuous."""
    af = torch.zeros(batch_size, 71, 12)
    # Populate a few action slots with card IDs and features
    af[:, 0, 0] = 42.0  # card_id for summon slot 0
    af[:, 0, 1] = 1.0   # is_summon
    af[:, 0, 8] = 0.5   # ATK/5000
    af[:, 0, 9] = 0.4   # DEF/5000
    af[:, 0, 10] = 0.33  # level/12
    af[:, 0, 11] = 0.2   # location = hand
    af[:, 37, 0] = 10.0  # card_id for attack slot 0
    af[:, 37, 5] = 1.0   # is_attack
    af[:, 37, 8] = 0.6   # ATK
    af[:, 37, 11] = 0.4  # location = mzone
    return af


def _make_action_history(batch_size: int) -> torch.Tensor:
    """Create realistic action_history: col 0 = card_id, cols 1-9 = continuous."""
    ah = torch.zeros(batch_size, 16, 10)
    # Populate last 3 history slots (right-aligned)
    ah[:, 13, 0] = 5.0   # card_id
    ah[:, 13, 2] = 1.0   # is_summon
    ah[:, 13, 7] = 0.5   # ATK
    ah[:, 14, 0] = 10.0
    ah[:, 14, 4] = 1.0   # is_activate
    ah[:, 15, 0] = 20.0
    ah[:, 15, 5] = 1.0   # is_attack
    ah[:, 15, 9] = 0.025  # turns_ago
    return ah


def test_output_shape(obs_space, sample_obs):
    """Output shape should be (batch, hidden_dim)."""
    extractor = CardEmbeddingExtractor(obs_space, hidden_dim=512, vocab_size=200)
    out = extractor(sample_obs)
    assert out.shape == (4, 512)


def test_features_dim_attribute(obs_space):
    """features_dim property should equal hidden_dim."""
    extractor = CardEmbeddingExtractor(obs_space, hidden_dim=512, vocab_size=200)
    assert extractor.features_dim == 512


def test_has_transformer_and_segments(obs_space):
    """Extractor should have transformer, segment embedding, and projections."""
    extractor = CardEmbeddingExtractor(obs_space, hidden_dim=512, vocab_size=200)
    assert hasattr(extractor, "transformer")
    assert hasattr(extractor, "segment_embedding")
    assert hasattr(extractor, "board_project")
    assert hasattr(extractor, "action_project")
    assert hasattr(extractor, "history_project")


def test_pretrained_embeddings_load(obs_space, sample_obs):
    """Pretrained embeddings should be loaded into the embedding layer."""
    vocab_size = 200
    embed_dim = 32
    pretrained = np.random.randn(vocab_size, embed_dim).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, pretrained)
        tmp_path = f.name

    extractor = CardEmbeddingExtractor(
        obs_space, embed_dim=embed_dim, hidden_dim=512, vocab_size=vocab_size,
        pretrained_embeddings_path=tmp_path,
    )

    weights = extractor.embedding.weight.data.numpy()
    # Padding index 0 should be zero
    assert np.allclose(weights[0], 0.0)
    # Non-padding rows should match pretrained (except row 0 which was zeroed)
    assert np.allclose(weights[1:], pretrained[1:])

    # Forward pass should still work
    out = extractor(sample_obs)
    assert out.shape == (4, 512)

    Path(tmp_path).unlink()


def test_padding_idx_zero_after_init(obs_space):
    """Padding index 0 should remain zero even with pretrained weights."""
    vocab_size = 200
    embed_dim = 32
    pretrained = np.ones((vocab_size, embed_dim), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, pretrained)
        tmp_path = f.name

    extractor = CardEmbeddingExtractor(
        obs_space, embed_dim=embed_dim, hidden_dim=512, vocab_size=vocab_size,
        pretrained_embeddings_path=tmp_path,
    )

    assert np.allclose(extractor.embedding.weight.data[0].numpy(), 0.0)

    Path(tmp_path).unlink()


def test_pretrained_shape_mismatch(obs_space):
    """Loading embeddings with wrong shape should raise AssertionError."""
    wrong_shape = np.random.randn(100, 16).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, wrong_shape)
        tmp_path = f.name

    with pytest.raises(AssertionError, match="Pretrained embeddings shape"):
        CardEmbeddingExtractor(
            obs_space, embed_dim=32, hidden_dim=512, vocab_size=200,
            pretrained_embeddings_path=tmp_path,
        )

    Path(tmp_path).unlink()


def test_no_pretrained_random_init(obs_space, sample_obs):
    """Without pretrained path, embeddings should be random (not all zero)."""
    extractor = CardEmbeddingExtractor(obs_space, hidden_dim=512, vocab_size=200)
    weights = extractor.embedding.weight.data[1:]  # skip padding
    assert not torch.allclose(weights, torch.zeros_like(weights))

    out = extractor(sample_obs)
    assert out.shape == (4, 512)


def test_all_padded_no_nan(obs_space):
    """All-zero inputs (fully padded) should produce valid output, no NaN."""
    extractor = CardEmbeddingExtractor(obs_space, hidden_dim=256, vocab_size=200)
    obs = {
        "features": torch.zeros(2, 453),
        "card_ids": torch.zeros(2, 50),
        "action_features": torch.zeros(2, 71, 12),
        "action_history": torch.zeros(2, 16, 10),
    }
    out = extractor(obs)
    assert out.shape == (2, 256)
    assert not torch.isnan(out).any(), "NaN in output with all-padded input"


def test_phase_transition_actions(obs_space):
    """Phase transition actions (no card, just action_type flag) should work."""
    extractor = CardEmbeddingExtractor(obs_space, hidden_dim=256, vocab_size=200)
    af = torch.zeros(2, 71, 12)
    # toBP action (index 35): no card_id, just is_phase_pass_other flag
    af[:, 35, 7] = 1.0  # col 7 = is_phase_pass_other
    # toEP action (index 36)
    af[:, 36, 7] = 1.0
    obs = {
        "features": torch.randn(2, 453),
        "card_ids": torch.randint(0, 100, (2, 50)).float(),
        "action_features": af,
        "action_history": torch.zeros(2, 16, 10),
    }
    out = extractor(obs)
    assert out.shape == (2, 256)
    assert not torch.isnan(out).any()


def test_d_model_kwarg(obs_space, sample_obs):
    """d_model parameter should be respected."""
    extractor = CardEmbeddingExtractor(
        obs_space, hidden_dim=256, vocab_size=200, d_model=128
    )
    assert extractor._d_model == 128
    out = extractor(sample_obs)
    assert out.shape == (4, 256)
