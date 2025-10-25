"""Client selection strategies for federated learning."""

import random
from abc import ABC, abstractmethod
from typing import List, Optional

from strategicflsim.agents.base_client import BaseClient


class BaseSelector(ABC):
    """
    Abstract base class for client selection strategies.

    Example:
        class MySelector(BaseSelector):
            def __init__(self, param):
                self.param = param

            def select(self, clients, fraction=1.0):
                # Your selection logic
                return selected_clients
    """

    @abstractmethod
    def select(
        self, clients: List[BaseClient], fraction: float = 1.0, **kwargs
    ) -> List[BaseClient]:
        """
        Select subset of clients for training round.

        Args:
            clients: Available clients
            fraction: Fraction of clients to select (0, 1]
            **kwargs: Strategy-specific parameters

        Returns:
            List of selected clients
        """
        pass

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


class RandomSelector(BaseSelector):
    """
    Uniform random sampling (standard FedAvg).

    Args:
        seed: Random seed for reproducibility (default: None)

    Example:
        selector = RandomSelector(seed=42)
        selected = selector.select(clients, fraction=0.3)
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = random.Random(seed)

    def select(
        self, clients: List[BaseClient], fraction: float = 1.0, **kwargs
    ) -> List[BaseClient]:
        """Uniformly sample clients at random."""
        if not 0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if not clients:
            raise ValueError("clients list is empty")

        n_selected = max(1, int(len(clients) * fraction))
        return self._rng.sample(clients, n_selected)


class AllSelector(BaseSelector):
    """
    Select all clients every round (full participation).

    Example:
        selector = AllSelector()
        selected = selector.select(clients)  # Returns all clients
    """

    def select(
        self, clients: List[BaseClient], fraction: float = 1.0, **kwargs
    ) -> List[BaseClient]:
        """Return all clients (ignores fraction)."""
        return clients


class RoundRobinSelector(BaseSelector):
    """
    Cycle through clients deterministically.

    Args:
        clients_per_round: Number of clients to select each round

    Example:
        selector = RoundRobinSelector(clients_per_round=3)
        # Round 0: clients[0:3]
        # Round 1: clients[3:6]
        # Round 2: clients[6:9], then wraps to clients[0:1]
    """

    def __init__(self, clients_per_round: int = 1):
        self.clients_per_round = clients_per_round
        self._offset = 0

    def select(
        self, clients: List[BaseClient], fraction: float = 1.0, **kwargs
    ) -> List[BaseClient]:
        """Select next batch in round-robin order."""
        n_clients = len(clients)
        n_selected = min(self.clients_per_round, n_clients)

        selected = []
        for i in range(n_selected):
            idx = (self._offset + i) % n_clients
            selected.append(clients[idx])

        self._offset = (self._offset + n_selected) % n_clients
        return selected
