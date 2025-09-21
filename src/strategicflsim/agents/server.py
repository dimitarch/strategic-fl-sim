import random
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from .base_client import BaseClient
from .base_server import BaseServer


class Server(BaseServer):
    """
    Standard federated learning server with configurable gradient aggregation.

    Coordinates training across multiple clients using selectable aggregation methods
    for robustness against strategic client behavior.
    """

    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        aggregate_fn: Callable,
        agent_id: str = "server",
    ):
        """
        Initialize server with model, optimizer, and aggregation strategy.

        Args:
            device: Computing device (CPU/GPU)
            model: Global neural network model
            criterion: Loss function for evaluation
            optimizer: Optimizer for global model updates
            aggregate_fn: Function for aggregating client gradients
            agent_id: Server identifier (default: "server")
        """
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.aggregate_fn = aggregate_fn
        self.agent_id = agent_id
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def train(
        self,
        clients: List[BaseClient],
        T: int = 1000,
        client_fraction: float = 1.0,
        get_metrics: Optional[Any] = None,
    ) -> Tuple[List[List[float]], Optional[List[dict]]]:
        """
        Execute federated learning protocol with progress tracking.

        Implements: client selection → model broadcast → local training →
        gradient aggregation → global update cycle.
        """
        losses_global = []
        metrics_global = []

        for _ in tqdm(range(T), total=T, desc="Federated Training"):
            selected_clients = self.select_clients(clients, fraction=client_fraction)
            self.broadcast_model(selected_clients)

            client_gradients = []
            round_losses = []

            for client in selected_clients:
                gradient, loss = client.local_train()
                client_gradients.append(gradient)
                round_losses.append(loss)

            self.update(client_gradients)

            if get_metrics is not None:
                aggregated_gradient = self.aggregate(client_gradients)
                metrics_global.append(
                    get_metrics(client_gradients, aggregated_gradient)
                )

            losses_global.append(round_losses)

        torch.cuda.empty_cache()
        return losses_global, metrics_global

    def select_clients(
        self,
        clients: List[BaseClient],
        fraction: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> List[BaseClient]:
        """Random sampling of clients with minimum of 1 selected."""
        if random_seed is not None:
            random.seed(random_seed)

        n_selected = max(1, int(len(clients) * fraction))
        return random.sample(clients, n_selected)

    def aggregate(self, gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Apply configured aggregation function to client gradients."""
        return self.aggregate_fn(gradients)

    def broadcast_model(self, clients: List[BaseClient]) -> None:
        """Send only trainable parameters to minimize communication overhead."""
        trainable_state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        for client in clients:
            client.receive_global_model(trainable_state)

    def reset_trainable_parameters(self) -> None:
        """Update parameter list after requires_grad changes."""
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def update(self, gradients: List[List[torch.Tensor]]) -> None:
        """Update global model with aggregated client gradients."""
        aggregated_gradient = self.aggregate(gradients)
        self.optimizer.zero_grad()

        for param, grad in zip(self.trainable_params, aggregated_gradient):
            param.grad = grad

        self.optimizer.step()

    def evaluate(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Evaluate with temporary eval mode."""
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

    def __str__(self):
        return f"Server(id={self.agent_id})"

    def __repr__(self):
        return self.__str__()
