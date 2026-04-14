"""
DreamerV3-style Recurrent State Space Model (RSSM) for SupplyMind.

Learns latent dynamics of the supply chain environment:
  - encoder:  state(408) -> latent_mean + log_var (VAE-style)
  - GRUCell:  (latent + action) -> hidden (recurrent transition)
  - decoder:  hidden -> reward, done, next_state predictions

Key method:
    imagine_rollout(initial_state, policy, horizon=15)
    Roll out imagined trajectories in latent space for 15 steps.
    Returns predicted rewards, states, uncertainty bounds.

Demo: "Watch our world model predict the cascade: TSMC disruption ->
chipmaker shortage -> OEM production halt -- 15 days before it happens."

Dimensions: state_dim=408, action_dim=280, latent_dim=128, hidden_dim=256
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


class SupplyChainRSSM(nn.Module):
    """Recurrent State Space Model for supply chain dynamics.

    Components:
      1. Encoder:     state(408) -> (latent_mean, latent_log_var) via VAE
      2. Transition:  GRUCell(latent_dim + action_dim, hidden_dim) -> next hidden
      3. Latent head: hidden -> (next_latent_mean, next_latent_log_var)
      4. Decoders:    hidden -> reward, done, reconstructed state

    Args:
        state_dim:  Observation dimension (408).
        action_dim: Action dimension (280, one-hot).
        latent_dim: Latent representation dimension (128).
        hidden_dim: GRU hidden state dimension (256).
    """

    def __init__(
        self,
        state_dim: int = 408,
        action_dim: int = 280,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder: state -> (mean, log_var) in latent space
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.enc_mean = nn.Linear(128, latent_dim)
        self.enc_logvar = nn.Linear(128, latent_dim)

        # Recurrent transition: GRUCell
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Latent prediction from hidden state
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.latent_mean = nn.Linear(128, latent_dim)
        self.latent_logvar = nn.Linear(128, latent_dim)

        # Decoder heads
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, state_dim),
        )

    def encode(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into latent distribution.

        Args:
            state: (batch, state_dim)

        Returns:
            mean:    (batch, latent_dim)
            log_var: (batch, latent_dim)
        """
        h = self.encoder(state)
        return self.enc_mean(h), self.enc_logvar(h)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def transition(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """One-step transition in latent space.

        Args:
            latent: (batch, latent_dim)
            action: (batch, action_dim) one-hot
            hidden: (batch, hidden_dim) previous GRU hidden state

        Returns:
            next_hidden: (batch, hidden_dim)
        """
        gru_input = torch.cat([latent, action], dim=-1)
        return self.gru(gru_input, hidden)

    def predict_latent(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent distribution from hidden state."""
        h = self.latent_head(hidden)
        return self.latent_mean(h), self.latent_logvar(h)

    def decode(
        self, hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode hidden state to reward, done, state.

        Returns:
            reward:    (batch, 1)
            done_prob: (batch, 1)
            state:     (batch, state_dim)
        """
        return (
            self.reward_head(hidden),
            self.done_head(hidden),
            self.state_decoder(hidden),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process a sequence of (state, action) pairs.

        Args:
            states:  (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)

        Returns:
            Dict with pred_states, pred_rewards, pred_dones,
            posterior_means, posterior_logvars, prior_means, prior_logvars.
        """
        B, T, _ = states.shape
        device = states.device

        hidden = torch.zeros(B, self.hidden_dim, device=device)

        pred_states = []
        pred_rewards = []
        pred_dones = []
        posterior_means = []
        posterior_logvars = []
        prior_means = []
        prior_logvars = []

        for t in range(T):
            # Posterior: encode actual observation
            post_mean, post_logvar = self.encode(states[:, t])
            latent = self.reparameterize(post_mean, post_logvar)

            # Transition
            hidden = self.transition(latent, actions[:, t], hidden)

            # Prior: predict latent from hidden (for KL loss)
            pri_mean, pri_logvar = self.predict_latent(hidden)

            # Decode
            reward, done, recon_state = self.decode(hidden)

            pred_states.append(recon_state)
            pred_rewards.append(reward)
            pred_dones.append(done)
            posterior_means.append(post_mean)
            posterior_logvars.append(post_logvar)
            prior_means.append(pri_mean)
            prior_logvars.append(pri_logvar)

        return {
            "pred_states": torch.stack(pred_states, dim=1),
            "pred_rewards": torch.stack(pred_rewards, dim=1),
            "pred_dones": torch.stack(pred_dones, dim=1),
            "posterior_means": torch.stack(posterior_means, dim=1),
            "posterior_logvars": torch.stack(posterior_logvars, dim=1),
            "prior_means": torch.stack(prior_means, dim=1),
            "prior_logvars": torch.stack(prior_logvars, dim=1),
        }

    @torch.no_grad()
    def imagine_rollout(
        self,
        initial_state: torch.Tensor,
        policy,
        horizon: int = 15,
    ) -> dict[str, Any]:
        """Roll out imagined trajectories in latent space.

        Args:
            initial_state: (batch, state_dim) starting observation.
            policy:        Callable(state) -> action_onehot (batch, action_dim).
            horizon:       Number of steps to imagine (default 15).

        Returns:
            Dict with imagined_states, imagined_rewards, imagined_dones,
            uncertainty_lower, uncertainty_upper (for visualization).
        """
        self.eval()
        B = initial_state.shape[0]
        device = initial_state.device
        hidden = torch.zeros(B, self.hidden_dim, device=device)

        imagined_states = [initial_state]
        imagined_rewards = []
        imagined_dones = []
        uncertainties = []

        current_state = initial_state

        for t in range(horizon):
            # Encode current state
            mean, logvar = self.encode(current_state)
            latent = self.reparameterize(mean, logvar)

            # Get action from policy
            action = policy(current_state)

            # Transition
            hidden = self.transition(latent, action, hidden)

            # Decode predictions
            reward, done_prob, next_state = self.decode(hidden)

            # Uncertainty from latent variance
            uncertainty = torch.exp(0.5 * logvar).mean(dim=-1, keepdim=True)

            imagined_states.append(next_state)
            imagined_rewards.append(reward)
            imagined_dones.append(done_prob)
            uncertainties.append(uncertainty)

            current_state = next_state

        return {
            "imagined_states": torch.stack(imagined_states, dim=1),    # (B, H+1, state_dim)
            "imagined_rewards": torch.stack(imagined_rewards, dim=1),  # (B, H, 1)
            "imagined_dones": torch.stack(imagined_dones, dim=1),      # (B, H, 1)
            "uncertainties": torch.stack(uncertainties, dim=1),         # (B, H, 1)
        }


def compute_rssm_loss(
    outputs: dict[str, torch.Tensor],
    target_states: torch.Tensor,
    target_rewards: torch.Tensor,
    target_dones: torch.Tensor,
    kl_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute RSSM training loss.

    Components:
      - State reconstruction (MSE)
      - Reward prediction (MSE)
      - Done prediction (BCE)
      - KL divergence (posterior || prior)
    """
    state_loss = F.mse_loss(outputs["pred_states"], target_states)
    reward_loss = F.mse_loss(outputs["pred_rewards"], target_rewards)
    done_loss = F.binary_cross_entropy(outputs["pred_dones"], target_dones)

    # KL divergence between posterior and prior
    post_mean = outputs["posterior_means"]
    post_logvar = outputs["posterior_logvars"]
    pri_mean = outputs["prior_means"]
    pri_logvar = outputs["prior_logvars"]

    kl = 0.5 * (
        pri_logvar - post_logvar
        + (post_logvar.exp() + (post_mean - pri_mean).pow(2)) / pri_logvar.exp()
        - 1
    ).sum(dim=-1).mean()

    total_loss = state_loss + reward_loss + 0.1 * done_loss + kl_weight * kl

    return {
        "total": total_loss,
        "state": state_loss,
        "reward": reward_loss,
        "done": done_loss,
        "kl": kl,
    }
