"""
Three-segment transformer feature extractor with learnable card embeddings.

Architecture:
  Board tokens:   card_ids(50) → embed(32) → Linear(32, 64) → (50, 64)
  Action tokens:  action_features(71,12) → embed col0(32) + cols1-11 → Linear(43, 64) → (71, 64)
  History tokens: action_history(16,10) → embed col0(32) + cols1-9 → Linear(41, 64) → (16, 64)

  + segment_embedding(3, 64): board=0, action=1, history=2
  → concat: (137, 64)
  → TransformerEncoder(d_model=64, heads=4, layers=2)
  → segment-aware masked mean pooling: board_pool(64) | action_pool(64) | history_pool(64)
  → Linear(192, 256) + ReLU → embed_stream(256)

  Merge: features(453) | embed_stream(256) = 709
  → Linear(709, 256) → ReLU → Linear(256, 256) → ReLU → output(256)
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


EMBED_STREAM_DIM = 256


class TokenProjection(nn.Module):
    """Project embedded card ID + continuous features to d_model."""

    def __init__(self, embed_dim: int, n_continuous: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim + n_continuous, d_model)

    def forward(self, embed: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embed: (batch, seq, embed_dim)
            continuous: (batch, seq, n_continuous)
        Returns:
            (batch, seq, d_model)
        """
        return self.linear(torch.cat([embed, continuous], dim=-1))


class CardEmbeddingExtractor(BaseFeaturesExtractor):
    """
    Three-segment transformer feature extractor.

    Processes board cards, action features, and action history through a shared
    transformer with segment embeddings, then pools each segment separately.

    Input observation dict:
        "features"        : (batch, features_dim) continuous floats
        "card_ids"        : (batch, 50) integer-valued floats
        "action_features" : (batch, 71, 12) float — col 0 is card_id, cols 1-11 continuous
        "action_history"  : (batch, 16, 10) float — col 0 is card_id, cols 1-9 continuous

    Output: (batch, hidden_dim)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        vocab_size: int = 14337,
        pretrained_embeddings_path: str | None = None,
        d_model: int = 64,
        attn_heads: int = 4,
        attn_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim=hidden_dim)

        features_dim = observation_space["features"].shape[0]
        self._num_card_slots = observation_space["card_ids"].shape[0]  # 50
        self._action_feat_shape = observation_space["action_features"].shape  # (71, 12)
        self._history_shape = observation_space["action_history"].shape  # (16, 10)
        self._embed_dim = embed_dim
        self._d_model = d_model

        # Shared card embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # Load pretrained embeddings if provided
        if pretrained_embeddings_path is not None:
            pretrained = np.load(pretrained_embeddings_path)
            pretrained = torch.from_numpy(pretrained).float()
            assert pretrained.shape == (vocab_size, embed_dim), (
                f"Pretrained embeddings shape {pretrained.shape} != "
                f"expected ({vocab_size}, {embed_dim})"
            )
            pretrained[0] = 0.0
            self.embedding.weight.data.copy_(pretrained)

        # Token projections
        self.board_project = nn.Linear(embed_dim, d_model)
        # action_features: col 0 = card_id (embedded), cols 1-11 = 11 continuous
        n_action_continuous = self._action_feat_shape[1] - 1  # 11
        self.action_project = TokenProjection(embed_dim, n_action_continuous, d_model)
        # action_history: col 0 = card_id (embedded), cols 1-9 = 9 continuous
        n_history_continuous = self._history_shape[1] - 1  # 9
        self.history_project = TokenProjection(embed_dim, n_history_continuous, d_model)

        # Segment embeddings: 0=board, 1=action, 2=history
        self.segment_embedding = nn.Embedding(3, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attn_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)

        # Pool projection: 3 segments × d_model → embed_stream_dim
        self.pool_project = nn.Sequential(
            nn.Linear(3 * d_model, EMBED_STREAM_DIM),
            nn.ReLU(),
        )

        # Merge MLP: features + embed_stream → hidden_dim
        merge_dim = features_dim + EMBED_STREAM_DIM
        self.mlp = nn.Sequential(
            nn.Linear(merge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        features = observations["features"]
        card_ids = observations["card_ids"].long()
        action_features = observations["action_features"]
        action_history = observations["action_history"]

        batch_size = features.shape[0]
        device = features.device

        # ---- Board tokens (50) ----
        board_embed = self.embedding(card_ids)  # (B, 50, embed_dim)
        board_tokens = self.board_project(board_embed)  # (B, 50, d_model)
        board_pad = card_ids == 0  # (B, 50)

        # ---- Action tokens (71) ----
        action_card_ids = action_features[:, :, 0].long()  # (B, 71)
        action_continuous = action_features[:, :, 1:]  # (B, 71, 11)
        action_embed = self.embedding(action_card_ids)  # (B, 71, embed_dim)
        action_tokens = self.action_project(action_embed, action_continuous)  # (B, 71, d_model)
        # Pad where card_id==0 AND all continuous features==0
        action_pad = (action_card_ids == 0) & (action_continuous.abs().sum(dim=-1) == 0)

        # ---- History tokens (16) ----
        history_card_ids = action_history[:, :, 0].long()  # (B, 16)
        history_continuous = action_history[:, :, 1:]  # (B, 16, 9)
        history_embed = self.embedding(history_card_ids)  # (B, 16, embed_dim)
        history_tokens = self.history_project(history_embed, history_continuous)  # (B, 16, d_model)
        # Pad where all features are zero (empty history slot)
        history_pad = (history_card_ids == 0) & (history_continuous.abs().sum(dim=-1) == 0)

        # ---- Add segment embeddings ----
        n_board = board_tokens.shape[1]
        n_action = action_tokens.shape[1]
        n_history = history_tokens.shape[1]

        seg_board = self.segment_embedding(
            torch.zeros(batch_size, n_board, dtype=torch.long, device=device)
        )
        seg_action = self.segment_embedding(
            torch.ones(batch_size, n_action, dtype=torch.long, device=device)
        )
        seg_history = self.segment_embedding(
            torch.full((batch_size, n_history), 2, dtype=torch.long, device=device)
        )

        board_tokens = board_tokens + seg_board
        action_tokens = action_tokens + seg_action
        history_tokens = history_tokens + seg_history

        # ---- Concatenate and run transformer ----
        all_tokens = torch.cat([board_tokens, action_tokens, history_tokens], dim=1)
        all_pad = torch.cat([board_pad, action_pad, history_pad], dim=1)

        # Transformer with padding mask
        out = self.transformer(all_tokens, src_key_padding_mask=all_pad)

        # ---- Segment-aware masked mean pooling ----
        # Split back into segments
        board_out = out[:, :n_board]
        action_out = out[:, n_board:n_board + n_action]
        history_out = out[:, n_board + n_action:]

        board_pool = self._masked_mean(board_out, ~board_pad)
        action_pool = self._masked_mean(action_out, ~action_pad)
        history_pool = self._masked_mean(history_out, ~history_pad)

        pooled = torch.cat([board_pool, action_pool, history_pool], dim=-1)
        embed_stream = self.pool_project(pooled)  # (B, 256)

        # ---- Merge with features ----
        combined = torch.cat([features, embed_stream], dim=1)
        return self.mlp(combined)

    @staticmethod
    def _masked_mean(
        x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pool over non-padded positions. Returns zeros if all padded."""
        mask_f = mask.unsqueeze(-1).float()  # (B, seq, 1)
        count = mask_f.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return (x * mask_f).sum(dim=1) / count  # (B, d_model)
