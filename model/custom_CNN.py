from sb3_contrib.ppo_mask.policies import MaskableActorCriticCnnPolicy
from torch import nn
import torch as th
import gym

from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import os




class CustomCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:

        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1,  padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1,  padding=1),
            nn.ReLU(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float().unsqueeze(1)).shape[1]

        self.conv_last = nn.Conv2d(32, 4, kernel_size=3, stride=1,  padding=1)

        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.cnn(observations.unsqueeze(1))
        x = self.conv_last(x)
        x = self.flatten(x)
        return x

