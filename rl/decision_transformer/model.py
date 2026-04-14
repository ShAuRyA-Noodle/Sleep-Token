"""
Decision Transformer for SupplyMind.

GPT-2 backbone that treats RL as sequence prediction.
Interleaves (return_to_go, state, action) tuples and predicts actions
from state token positions.

Killer feature: Return-to-go conditioning. At inference, set
desired_return=0.9 for aggressive or 0.5 for conservative.
Same model, different behavior. No retraining.

Architecture:
    GPT2Config: n_embd=128, n_layer=3, n_head=1
    Embeddings: return_to_go(1→128), state(408→128), action(280→128), timestep(60→128)
    Context length: 20 timesteps (60 tokens)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention block."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    """Transformer block: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer for supply chain risk management.

    Sequence structure (per timestep):
        [return_to_go_embed, state_embed, action_embed]

    At inference, only state positions predict actions.

    Args:
        state_dim:    Observation dimension (408).
        action_dim:   Number of discrete actions (280).
        n_embd:       Embedding dimension (128).
        n_layer:      Number of transformer layers (3).
        n_head:       Number of attention heads (1).
        context_len:  Number of timesteps in context window (20).
        max_timestep: Maximum episode timestep (60).
        dropout:      Dropout rate (0.1).
    """

    def __init__(
        self,
        state_dim: int = 408,
        action_dim: int = 280,
        n_embd: int = 128,
        n_layer: int = 3,
        n_head: int = 1,
        context_len: int = 20,
        max_timestep: int = 60,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.context_len = context_len
        self.max_timestep = max_timestep

        block_size = 3 * context_len  # (r, s, a) per timestep

        # Embedding layers: project each modality to n_embd
        self.embed_return = nn.Linear(1, n_embd)
        self.embed_state = nn.Linear(state_dim, n_embd)
        self.embed_action = nn.Linear(action_dim, n_embd)
        self.embed_timestep = nn.Embedding(max_timestep + 1, n_embd)

        self.embed_ln = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        # Prediction head (from state positions → action logits)
        self.action_head = nn.Linear(n_embd, action_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        returns_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        All inputs have shape (batch, context_len, ...) and are LEFT-PADDED
        with zeros where context is shorter than context_len.

        Args:
            returns_to_go: (B, T, 1) — desired future return at each step.
            states:        (B, T, state_dim)
            actions:       (B, T, action_dim) — one-hot encoded.
            timesteps:     (B, T) int64 — episode timestep indices.
            attention_mask: (B, T) bool — True for real tokens, False for padding.

        Returns:
            action_logits: (B, T, action_dim) — logits at each state position.
        """
        B, T = states.shape[0], states.shape[1]

        # Time embeddings
        time_emb = self.embed_timestep(timesteps.clamp(0, self.max_timestep))  # (B, T, E)

        # Modality embeddings + time
        r_emb = self.embed_return(returns_to_go) + time_emb  # (B, T, E)
        s_emb = self.embed_state(states) + time_emb          # (B, T, E)
        a_emb = self.embed_action(actions) + time_emb        # (B, T, E)

        # Interleave: (r1, s1, a1, r2, s2, a2, ...)
        # Shape: (B, 3*T, E)
        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2)  # (B, T, 3, E)
        tokens = stacked.reshape(B, 3 * T, self.n_embd)

        tokens = self.embed_ln(tokens)
        tokens = self.drop(tokens)

        # Apply attention mask if provided
        # The mask applies per-timestep to all 3 tokens (r, s, a)
        if attention_mask is not None:
            # Expand (B, T) → (B, 3*T)
            token_mask = attention_mask.unsqueeze(2).expand(-1, -1, 3).reshape(B, 3 * T)
            # Zero out padding positions
            tokens = tokens * token_mask.unsqueeze(-1).float()

        # Transformer
        hidden = self.blocks(tokens)
        hidden = self.ln_f(hidden)

        # Extract state positions: indices 1, 4, 7, ... (every 3rd starting at 1)
        state_positions = hidden[:, 1::3, :]  # (B, T, E)

        # Predict actions from state positions
        action_logits = self.action_head(state_positions)  # (B, T, action_dim)
        return action_logits

    @torch.no_grad()
    def predict_action(
        self,
        returns_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """Predict next action given history.

        This is the inference entry point. Sequences should be left-padded
        to context_len if shorter.

        Args:
            returns_to_go: (1, T, 1)
            states:        (1, T, state_dim)
            actions:       (1, T, action_dim)
            timesteps:     (1, T)
            action_mask:   (280,) boolean mask, True = valid.

        Returns:
            action: int — flat action index (0-279).
        """
        self.eval()
        logits = self.forward(returns_to_go, states, actions, timesteps)
        # Take the last timestep's prediction
        logits = logits[0, -1, :]  # (action_dim,)

        if action_mask is not None:
            logits[~action_mask] = float("-inf")

        return logits.argmax().item()

    def get_action_probs(
        self,
        returns_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get action probability distribution (softmax over logits).

        Used by ensemble to combine with QR-DQN.
        """
        logits = self.forward(returns_to_go, states, actions, timesteps)
        return F.softmax(logits[:, -1, :], dim=-1)
