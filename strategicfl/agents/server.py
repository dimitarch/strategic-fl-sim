import random
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from .base_client import BaseClient
from .base_server import BaseServer


class Server(BaseServer):
    """Federated learning server with aggregation support."""

    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        aggregate_fn: Callable,
        agent_id: str = "server",
    ):
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
        get_metrics: Optional[
            Any
        ] = None,  # TODO Hacky, make it so the function returns the gradients and aggregate, nothing else, just a record of the training
    ) -> Tuple[List[List[float]], Optional[List[dict]]]:
        """Coordinate federated training across clients.

        Args:
            clients: List of client agents
            T: Number of training rounds
            client_fraction: Fraction of clients to select each round
            get_metrics: Optional function to compute metrics

        Returns:
            Tuple of (losses_per_round, metrics_per_round)
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
                client_loss = loss.detach().cpu().item()
                client_gradients.append(gradient)
                round_losses.append(client_loss)

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
        """Select subset of clients for training round."""
        if random_seed is not None:
            random.seed(random_seed)

        n_selected = max(1, int(len(clients) * fraction))
        return random.sample(clients, n_selected)

    def aggregate(self, gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Aggregate gradients from multiple clients."""
        return self.aggregate_fn(gradients)

    def broadcast_model(self, clients: List[BaseClient]) -> None:
        """Broadcast only trainable parameters to all clients."""
        trainable_state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        for client in clients:
            client.receive_global_model(trainable_state)

    def reset_trainable_parameters(self) -> None:
        """Reset the list of trainable parameters."""
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def update(self, gradients: List[List[torch.Tensor]]) -> None:
        """Update global model with aggregated gradients for trainable parameters only."""
        aggregated_gradient = self.aggregate(gradients)

        self.optimizer.zero_grad()

        # Only update trainable parameters
        for param, grad in zip(self.trainable_params, aggregated_gradient):
            param.grad = grad

        self.optimizer.step()

    def evaluate(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Evaluate global model on specific inputs and labels."""
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
        """Make predictions using global model."""
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
