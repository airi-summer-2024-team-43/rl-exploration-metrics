from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def args_for_rnd(algo_args):
    @dataclass
    class ArgsRND(algo_args):
        # RND
        ext_coef: float = 1.0
        """RND extrinsic reward coef"""
        int_coef: float = 0.01
        """RND intrinsic reward coef"""
        output_rnd: int = 32
        """last layer RND output size"""

    return ArgsRND


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size=32):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
            layer_init(nn.Linear(input_size, 16)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(16, 32)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(32, 32)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(32, output_size)),
        )

        self.target = nn.Sequential(
            layer_init(nn.Linear(input_size, 16)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(16, 32)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(32, 32)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(32, output_size)),
        )

        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, obs):
        target_feature = self.target(obs)
        predict_feature = self.predictor(obs)

        return predict_feature, target_feature

    @torch.no_grad()
    def get_intrinsic_reward(self, obs):
        predict_feature, target_feature = self.forward(obs)
        return 0.5 * (predict_feature - target_feature).pow(2).mean(-1)

    def get_forward_loss(self, obs, mask_proportion=0.25):
        predict_feature, target_feature = self.forward(obs)
        loss = F.mse_loss(predict_feature, target_feature.detach(), reduction="none")
        mask = torch.rand_like(loss, device=loss.device)
        mask = (mask < mask_proportion).type(torch.FloatTensor).to(loss.device)
        loss = (loss * mask).sum() / torch.max(
            mask.sum(), torch.tensor([1], device=loss.device, dtype=torch.float32)
        )
        return loss

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
