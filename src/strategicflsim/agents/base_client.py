from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class BaseClient(ABC):
    """Abstract base class for federated learning clients."""

    @abstractmethod
    def update(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform a single gradient update step.

        Args:
            inputs: Input tensor
            labels: Label tensor

        Returns:
            Loss tensor
        """
        pass

    @abstractmethod
    def evaluate(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Evaluate on specific inputs and labels.

        Args:
            inputs: Input tensor
            labels: Label tensor

        Returns:
            Tuple of (accuracy, loss)
        """
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make predictions on input data.

        Args:
            inputs: Input tensor

        Returns:
            Predictions tensor
        """
        pass

    @abstractmethod
    def receive_global_model(self, global_state_dict: dict) -> None:
        """Receive and apply global model state from server.

        Args:
            global_state_dict: State dict of the global model
        """
        pass

    @abstractmethod
    def local_train(self) -> Tuple[List[torch.Tensor], float, int]:
        """Perform local training and return gradients and loss.

        Returns:
            Tuple of (gradient tensors to send to server, final loss)
        """
        pass

    @abstractmethod
    def apply_action(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply strategic action to a gradient tensor.

        Args:
            gradient: Input gradient tensor

        Returns:
            Modified gradient tensor
        """
        pass

    @abstractmethod
    def evaluate_on_test_set(self) -> Tuple[float, float]:
        """Evaluate model on test set.

        Returns:
            Tuple of (accuracy, average_loss)
        """
        pass
