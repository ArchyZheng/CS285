import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # implement the forward pass of the critic network
        return self.network.forward(obs)


    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        q_values = torch.from_numpy(q_values).float().to(device=ptu.device)

        # TODO: update the critic using the observations and q_values
        self.optimizer.zero_grad()
        predictions = self.network.forward(obs)
        loss = torch.nn.MSELoss()(predictions.squeeze(1), q_values)
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }