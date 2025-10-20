from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import torch

from .base_client import BaseClient


class BaseServer(ABC):
    """Abstract base class for federated learning servers."""

    @abstractmethod
    def update(self, gradients: List[List[torch.Tensor]]) -> None:
        """Update global model with aggregated gradients.

        Args:
            gradients: List of gradient lists from clients
        """
        pass

    @abstractmethod
    def evaluate(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Evaluate global model on specific inputs and labels.

        Args:
            inputs: Input tensor
            labels: Label tensor

        Returns:
            Tuple of (accuracy, loss)
        """
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make predictions using global model.

        Args:
            inputs: Input tensor

        Returns:
            Predictions tensor
        """
        pass

    @abstractmethod
    def broadcast_model(self, clients: List[BaseClient]) -> None:
        """Broadcast current global model to all clients.

        Args:
            clients: List of clients to broadcast to
        """
        pass

    @abstractmethod
    def select_clients(self, clients: List[BaseClient], **kwargs) -> List[BaseClient]:
        """Select subset of clients for training round.

        Args:
            clients: All available clients
            **kwargs: Selection parameters (e.g., fraction, random_seed)

        Returns:
            Selected clients for this round
        """
        pass

    @abstractmethod
    def aggregate(self, gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Aggregate gradients from multiple clients.

        Args:
            gradients: List of gradient lists from each client

        Returns:
            Aggregated gradient
        """
        pass

    @abstractmethod
    def train(
        self,
        clients: List[BaseClient],
        T: int = 1000,
        client_fraction: float = 1.0,
        get_metrics: Optional[Any] = None,
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
        pass
