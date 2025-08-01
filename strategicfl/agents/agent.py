from typing import Tuple

import torch
from torch import nn


class Agent:
    """Base agent class for federated learning."""

    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        agent_id: str = "agent",
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.agent_id = agent_id

    def update(self, inputs, labels) -> torch.Tensor:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()

        self.optimizer.step()

        return loss

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make predictions on input data."""
        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)

        if was_training:
            self.model.train()
        return predictions

    def evaluate(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float]:
        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)

            accuracy = (predictions == labels).float().mean().item()
            loss = self.criterion(outputs, labels).item()

        if was_training:
            self.model.train()

        return accuracy, loss

    def __str__(self):
        """String representation showing agent ID."""
        return f"{self.__class__.__name__}(id={self.agent_id})"

    def __repr__(self):
        return self.__str__()
