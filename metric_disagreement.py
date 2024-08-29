import torch
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass


def args_for_disagreement(algo_args):
    @dataclass
    class ArgsDisagreement(algo_args):
        ensemble_size: int = 10
        """Ensemble size"""

    return ArgsDisagreement


class Ensemble(nn.Module):
    def __init__(self, obs_dim, action_dim, n=10):
        super().__init__()
        self.n = n
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim + action_dim, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, obs_dim),
                )
                for _ in range(n)
            ]
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([self.networks[i](x) for i in range(self.n)])
    
    def get_ensemble_loss(self, obs, action, next_obs):
        preds = self.forward(obs, action)
        losses = torch.stack([F.mse_loss(pred, next_obs) for pred in preds])
        return losses.sum()

    @torch.no_grad()
    def get_disagreement(self, obs, action):
        preds = self.forward(obs, action)
        return preds.var(dim=-1).mean()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
