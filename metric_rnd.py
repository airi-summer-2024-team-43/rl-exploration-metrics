from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def args_for_RND(algo_args):
    @dataclass
    class ArgsRND(algo_args):
        # RND
        ext_coef: float = 1.0
        """RND extrinsic reward coef"""
        int_coef: float = 0.5
        """RND intrinsic reward coef"""
        output_rnd: int = 256

    return ArgsRND


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RNDModel(nn.Module):
    def __init__(self, output_size=512):
        super().__init__()

        self.output_size = output_size

        self.image_base = nn.Sequential(
            layer_init(
                nn.Conv2d(3, 32, 7),
                std=np.sqrt(2),
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            layer_init(
                nn.Conv2d(32, 64, 3),
                std=np.sqrt(2),
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            layer_init(
                nn.Conv2d(64, 128, 3),
                std=np.sqrt(2),
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            layer_init(
                nn.Conv2d(128, 128, 3),
                std=np.sqrt(2),
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            layer_init(
                nn.Linear(128, 128),
                std=np.sqrt(2),
            ),
            nn.LeakyReLU(),
            layer_init(
                nn.Linear(128, 64),
                std=np.sqrt(2),
            ),
        )

        self.obs_base = nn.Sequential(
            layer_init(nn.Linear(4, 16)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(16, 64)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(64, 64)),
        )

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(64 + 64, 128)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(128, 256)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, output_size)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Linear(64 + 64, 128)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(128, 256)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, output_size)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs, next_obs_img):
        x = torch.concat(
            [self.obs_base(next_obs), self.image_base(next_obs_img)], dim=1
        )
        target_feature = self.target(x)
        predict_feature = self.predictor(x)

        return predict_feature, target_feature


class RND_metric:
    def __init__(self, output_rnd: int, device) -> None:
        self.rnd_model = RNDModel(output_rnd).to(device)
