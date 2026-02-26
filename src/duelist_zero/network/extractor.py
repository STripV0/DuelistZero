"""
Custom feature extractor with learnable card ID embeddings.

Combines continuous game-state features with nn.Embedding lookups
for card identities, producing a fixed-size feature vector for
the policy and value heads.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CardEmbeddingExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that embeds discrete card IDs and concatenates
    them with continuous game-state features.

    Input observation dict:
        "features" : (batch, features_dim) continuous floats
        "card_ids" : (batch, num_card_slots) integer-valued floats

    Output: (batch, hidden_dim) after two hidden layers.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        vocab_size: int = 14337,
    ):
        # BaseFeaturesExtractor needs features_dim set in super().__init__
        super().__init__(observation_space, features_dim=hidden_dim)

        features_dim = observation_space["features"].shape[0]
        num_card_slots = observation_space["card_ids"].shape[0]
        num_action_cards = observation_space["action_cards"].shape[0]

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        flat_embed_dim = num_card_slots * embed_dim
        action_embed_dim = num_action_cards * embed_dim
        combined_dim = features_dim + flat_embed_dim + action_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        features = observations["features"]
        card_ids = observations["card_ids"].long()
        action_cards = observations["action_cards"].long()

        embedded = self.embedding(card_ids)  # (batch, num_slots, embed_dim)
        embedded_flat = embedded.view(embedded.size(0), -1)

        action_embedded = self.embedding(action_cards)  # (batch, 71, embed_dim)
        action_flat = action_embedded.view(action_embedded.size(0), -1)

        combined = torch.cat([features, embedded_flat, action_flat], dim=1)
        return self.mlp(combined)
