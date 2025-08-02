from typing import Callable

import torch
from torch import nn

from .agent import Agent


class Server(Agent):
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        aggregate_fn: Callable,
        agent_id: str = "server",
    ):
        super().__init__(device, model, criterion, optimizer, agent_id)
        self.aggregate_fn = aggregate_fn
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def reset_trainable_parameters(self) -> None:
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def update_model_with_gradient(self, aggregated_gradient: torch.Tensor) -> None:
        """Update global model with aggregated gradients.

        Args:
            aggregated_gradient: Aggregated gradient tensor to update model
        """
        self.optimizer.zero_grad()

        for param, grad in zip(self.trainable_params, aggregated_gradient):
            param.grad = grad

        self.optimizer.step()
