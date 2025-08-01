from typing import Callable

import torch
from torch import nn

from .agent import Agent


class Client(Agent):
    def __init__(
        self,
        device: torch.device,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        action: Callable,
        agent_id: str = "client",
    ):
        super().__init__(device, model, criterion, optimizer, agent_id)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_iterator = iter(train_dataloader)
        self.action = action

    def apply_action(self, gradient):
        return self.action(gradient)

    def get_next_train_batch(self):
        """Get next training batch."""
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            return next(self.train_iterator)

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
