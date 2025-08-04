import copy
from typing import Callable, List, Tuple

import torch
from torch import nn

from .base_client import BaseClient


class Client(BaseClient):
    def __init__(
        self,
        device: torch.device,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        action: Callable,
        local_steps: int = 1,
        agent_id: str = "client",
    ):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.agent_id = agent_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_iterator = iter(train_dataloader)
        self.action = action
        self.local_steps = local_steps

    def apply_action(self, gradient):
        return self.action(gradient)

    def receive_global_model(self, trainable_state_dict: dict) -> None:
        """Receive and apply only trainable parameters from server."""
        # Get current full state dict
        current_state = self.model.state_dict()

        # Update only the trainable parameters that were sent
        current_state.update(trainable_state_dict)

        # Load the updated state dict
        self.model.load_state_dict(current_state)

    def update(self, inputs, labels) -> torch.Tensor:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()

        self.optimizer.step()

        return loss

    def local_train(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Perform local training and return gradients for trainable parameters only."""
        try:
            inputs, labels = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            inputs, labels = next(self.train_iterator)

        initial_model = copy.deepcopy(self.model)
        if self.local_steps > 1:  # Multi-step local training
            for _ in range(self.local_steps):
                loss = self.update(inputs, labels)

            grad = []
            for local_param, server_param in zip(
                self.model.parameters(), initial_model.parameters()
            ):
                if local_param.requires_grad:  # Only trainable parameters
                    grad.append((server_param - local_param).detach().clone())
        else:  # Single step case
            outputs = self.model(inputs.to(self.device))
            loss = self.criterion(outputs, labels.to(self.device))
            loss.backward()
            # Only get gradients for trainable parameters
            grad = [
                p.grad.detach().clone()
                for p in self.model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            self.model.zero_grad()

        # Apply action to each trainable gradient
        sent_grad = [self.apply_action(g) for g in grad]
        return sent_grad, loss

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

    def evaluate_on_test_set(self):
        """Evaluate model on entire test set using batched processing."""
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        if was_training:
            self.model.train()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return accuracy, avg_loss
