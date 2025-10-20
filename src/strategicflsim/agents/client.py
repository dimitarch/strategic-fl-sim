import copy
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from .base_client import BaseClient


class Client(BaseClient):
    """
    Standard federated learning client with strategic gradient manipulation.

    Supports multi-step local training and applies configurable strategic actions
    to gradients before sending them to the server.
    """

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
        """
        Initialize client with model, data, and strategic behavior.

        Args:
            device: Computing device (CPU/GPU)
            train_dataloader: Training data loader
            test_dataloader: Test data loader
            model: Neural network model
            criterion: Loss function
            optimizer: Parameter optimizer
            action: Strategic action function for gradient manipulation
            local_steps: Number of local SGD steps per round (default: 1)
            agent_id: Client identifier (default: "client")
        """
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

    @classmethod
    def create_clients(
        cls,
        n_clients: int,
        devices: List[torch.device],
        data_splits: List[Tuple],
        model_fn: Callable,
        criterion_fn: Callable,
        optimizer_fn: Callable,
        action_fn: Callable,
        local_steps: int = 1,
        agent_ids: Optional[List[str]] = None,
    ) -> List["Client"]:
        """
        Factory method to create multiple clients with different configurations. Passes an action function that returns the specific action according to the client index
        """
        clients = []

        for i in range(n_clients):
            # Cycle through devices if we have fewer devices than clients
            device = devices[i % len(devices)]
            train_loader, test_loader = data_splits[i]

            model = model_fn().to(device)

            criterion = criterion_fn()
            if hasattr(criterion, "to"):
                criterion = criterion.to(device)

            optimizer = optimizer_fn(model.parameters())

            # Get agent_id, while handling missing data
            agent_id = agent_ids[i] if agent_ids else f"client_{i}"

            # Create client
            client = cls(
                device=device,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                action=action_fn(i),
                local_steps=local_steps,
                agent_id=agent_id,
            )

            clients.append(client)
            print(f"Created {agent_id} on {device}")

        return clients

    def apply_action(self, gradient):
        """Apply configured strategic action to gradient."""
        return self.action(gradient)

    def receive_global_model(self, trainable_state_dict: dict) -> None:
        """Update local model with trainable parameters from server."""
        current_state = self.model.state_dict()
        current_state.update(trainable_state_dict)
        self.model.load_state_dict(current_state)

    def update(self, inputs, labels) -> torch.Tensor:
        """Perform single SGD step on given batch."""
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss

    def local_train(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Perform local training with multi-step support.

        For multi-step: computes gradient as parameter difference.
        For single-step: uses direct backpropagation gradients.
        """
        try:
            inputs, labels = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            inputs, labels = next(self.train_iterator)

        if self.local_steps > 1:  # Multi-step local training
            initial_model = copy.deepcopy(self.model)

            for _ in range(self.local_steps):
                loss = self.update(inputs, labels)

            grad = []
            for local_param, server_param in zip(
                self.model.parameters(), initial_model.parameters()
            ):
                if local_param.requires_grad:
                    grad.append((server_param - local_param).detach().clone())

            # Delete copied initial model manually
            del initial_model
        else:  # Single step case
            outputs = self.model(inputs.to(self.device))
            loss = self.criterion(outputs, labels.to(self.device))
            loss.backward()
            grad = [
                p.grad.detach().clone()
                for p in self.model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            self.model.zero_grad()

        # Apply strategic action to each gradient
        sent_grad = [self.apply_action(g) for g in grad]

        # Clear cache to handle big models and little memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        client_loss = loss.detach().cpu().item()
        return sent_grad, client_loss

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate predictions with temporary eval mode."""
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
        """Evaluate on batch with temporary eval mode."""
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

    def evaluate_on_test_set(self) -> Tuple[float, float]:
        """Evaluate on entire test set with batched processing."""
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for inputs, labels in self.test_dataloader:
            batch_size = labels.size(0)

            accuracy, loss = self.evaluate(inputs, labels)

            # Accumulate metrics
            total_correct += accuracy * batch_size
            total_loss += loss * batch_size
            total_samples += batch_size

        # Compute weighted averages
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return avg_accuracy, avg_loss

    # def evaluate_on_test_set(self):
    #     """Evaluate on entire test set with batched processing."""
    #     was_training = self.model.training
    #     self.model.eval()

    #     total_loss = 0.0
    #     total_correct = 0
    #     total_samples = 0

    #     with torch.no_grad():
    #         for inputs, labels in self.test_dataloader:
    #             inputs = inputs.to(self.device)
    #             labels = labels.to(self.device)

    #             outputs = self.model(inputs)
    #             loss = self.criterion(outputs, labels)

    #             total_loss += loss.item() * inputs.size(0)
    #             _, predicted = torch.max(outputs, 1)
    #             total_correct += (predicted == labels).sum().item()
    #             total_samples += labels.size(0)

    #     if was_training:
    #         self.model.train()

    #     avg_loss = total_loss / total_samples
    #     accuracy = total_correct / total_samples
    #     return accuracy, avg_loss
