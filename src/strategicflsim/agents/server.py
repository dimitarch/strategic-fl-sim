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
        selector=None,
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
            selector: Client selection strategy (default: AllSelector())
            agent_id: Server identifier
        """
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.aggregate_fn = aggregate_fn

        if selector is None:
            from strategicflsim.utils.selection import AllSelector

            selector = AllSelector()
        self.selector = selector

        self.agent_id = agent_id

    def train(
        self,
        clients: List[BaseClient],
        T: int = 1000,
        client_fraction: float = 1.0,
        metrics: Optional[Any] = None,
    ) -> List[List[float]]:
        """
        Execute federated learning protocol.

        Args:
            clients: List of client agents
            T: Number of training rounds
            client_fraction: Fraction of clients to select (used by selector)
            metrics: Optional metrics callback

        Returns:
            List of losses per round
        """
        # Set server device for all clients, this is used to management ownership of gradients in single-node setup
        for client in clients:
            client.server_device = self.device

        # losses_global = []

        for round_idx in tqdm(range(T), total=T, desc="Federated Training"):
            selected_clients = self.selector.select(clients, fraction=client_fraction)

            self.broadcast_model(selected_clients)

            round_losses = []
            client_gradients = []
            client_num_samples = []

            for client in selected_clients:
                gradient, loss, num_samples = client.local_train()
                # Gradient on self.device

                client_gradients.append(gradient)
                client_num_samples.append(num_samples)
                round_losses.append(loss)

            aggregated_gradient = self.aggregate(client_gradients, client_num_samples)
            self.update(aggregated_gradient)

            if metrics is not None:
                metrics(
                    round=round_idx,
                    server=self,
                    selected_clients=selected_clients,
                    round_losses=round_losses,
                    client_gradients=client_gradients,
                    client_num_samples=client_num_samples,
                    aggregated_gradient=aggregated_gradient,
                )

            # losses_global.append(round_losses)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # return losses_global

    # def select_clients(
    #     self,
    #     clients: List[BaseClient],
    #     fraction: float = 1.0,
    #     random_seed: Optional[int] = None,
    # ) -> List[BaseClient]:
    #     """
    #     DEPRECATED: Use self.selector.select() instead.

    #     Kept for backward compatibility.
    #     """
    #     import warnings

    #     warnings.warn(
    #         "Server.select_clients() is deprecated. Use selector.select() instead.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )

    #     if random_seed is not None:
    #         random.seed(random_seed)

    #     n_selected = max(1, int(len(clients) * fraction))
    #     return random.sample(clients, n_selected)

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
