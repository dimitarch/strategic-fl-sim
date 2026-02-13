from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from .base_client import BaseClient
from .base_server import BaseServer


class Server(BaseServer):
    """
    Federated learning server with configurable aggregation and client selection.
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
        Initialize server.

        Args:
            device: Computing device
            model: Global model
            criterion: Loss function
            optimizer: Optimizer
            aggregate_fn: Aggregation function
            agent_id: Server identifier
        """
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.aggregate_fn = aggregate_fn

        self.agent_id = agent_id

    def train(
        self,
        clients: List[BaseClient],
        T: int = 1000,
        selector_fn=None,
        metrics_fn: Optional[Any] = None,
    ) -> List[List[float]]:
        """
        Execute federated learning protocol.

        Args:
            clients: List of client agents
            T: Number of training rounds
            selector_fn: Client selection strategy (default: AllSelector())
            metrics_fn: Optional metrics_fn callback

        Returns:
            List of losses per round
        """
        if selector_fn is None:
            from strategicflsim.utils.selection import AllSelector

            selector_fn = AllSelector()
        self.selector_fn = selector_fn

        # Set server device for all clients, this is used to management ownership of gradients in single-node setup
        for client in clients:
            client.server_device = self.device

        for round_idx in tqdm(range(T), total=T, desc="Federated Training"):
            selected_clients = self.selector_fn.select(clients)

            self.broadcast_model(selected_clients)

            round_losses = []
            client_gradients = []
            client_num_samples = []

            for client in selected_clients:
                gradient, loss, num_samples = (
                    client.local_train()
                )  # Gradient on self.device

                client_gradients.append(gradient)
                client_num_samples.append(num_samples)
                round_losses.append(loss)

            aggregated_gradient = self.aggregate(client_gradients, client_num_samples)
            self.update(aggregated_gradient)

            if metrics_fn is not None:
                metrics_fn(
                    round=round_idx,
                    server=self,
                    selected_clients=selected_clients,
                    round_losses=round_losses,
                    client_gradients=client_gradients,
                    client_num_samples=client_num_samples,
                    aggregated_gradient=aggregated_gradient,
                )

            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

        return

    def aggregate(
        self,
        gradients: List[List[torch.Tensor]],
        sizes: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Apply configured aggregation function."""
        return self.aggregate_fn(gradients, sizes)

    def broadcast_model(self, clients: List[BaseClient]) -> None:
        """Send trainable parameters to clients."""
        trainable_state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        for client in clients:
            client.receive_global_model(trainable_state)

    def update(self, aggregated_gradient: List[torch.Tensor]) -> None:
        """Update global model with aggregated gradients."""
        self.optimizer.zero_grad()

        for param, grad in zip(
            [p for p in self.model.parameters() if p.requires_grad], aggregated_gradient
        ):
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
