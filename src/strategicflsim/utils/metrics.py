"""Metrics collection for federated learning with kwargs-based callbacks."""

import csv
from typing import List, Optional

import torch


class BaseMetrics:
    """
    Base metrics class with CSV logging and kwargs-based callbacks.

    To create custom metrics, inherit and override __call__ which receives:
    - round: int - Current training round
    - server: Server - Server instance (READ-ONLY)
    - selected_clients: List[Client] - Clients selected this round
    - round_losses: List[float] - Training losses
    - client_gradients: List[List[Tensor]] - Client gradients
    - client_num_samples: List[int] - Dataset sizes
    - aggregated_gradient: List[Tensor] - Aggregated gradient

    Example:
        class MyMetrics(BaseMetrics):
            def __init__(self, save_path, client_ids):
                super().__init__(save_path, client_ids)
                self._init_csv(f"{save_path}_my_metric.csv")

            def __call__(self, round, server, selected_clients, **kwargs):
                # Your logic
                values = [...]
                self._append_csv(f"{save_path}_my_metric.csv", values)
    """

    def __init__(self, save_path: str, client_ids: List[str]):
        """
        Initialize base metrics logger.

        Args:
            save_path: Path prefix for CSV files
            client_ids: List of client identifiers
        """
        self.save_path = save_path
        self.client_ids = client_ids
        self.round = 0

    def _init_csv(self, filepath: str, header: Optional[List[str]] = None):
        """Initialize CSV file with header."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header or ["round"] + self.client_ids)

    def _append_csv(self, filepath: str, values: List):
        """Append row to CSV file (automatically prepends round number)."""
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.round] + values)

    def __call__(
        self,
        round: int,
        server,
        selected_clients,
        round_losses: List[float],
        client_gradients: List[List[torch.Tensor]],
        client_num_samples: List[int],
        aggregated_gradient: List[torch.Tensor],
        **kwargs,
    ):
        """
        Callback invoked after each training round.

        **WARNING**: Treat server/clients as read-only.

        Override this in subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement __call__. See BaseMetrics docstring."
        )


class NormMetrics(BaseMetrics):
    """
    Standard metrics: losses, gradient norms, cosine similarities.

    Output files:
        - {save_path}_losses.csv
        - {save_path}_grad_norms.csv
        - {save_path}_cosine_sims.csv

    Example:
        metrics = NormMetrics("results/exp1", ["good0", "good1", "bad"])
        server.train(clients, T=100, metrics=metrics)
    """

    def __init__(self, save_path: str, client_ids: List[str]):
        super().__init__(save_path, client_ids)
        self._init_csv(f"{save_path}_losses.csv")
        self._init_csv(f"{save_path}_grad_norms.csv")
        self._init_csv(f"{save_path}_cosine_sims.csv")

    def __call__(
        self,
        round: int,
        server,
        selected_clients,
        round_losses: List[float],
        client_gradients: List[List[torch.Tensor]],
        client_num_samples: List[int],
        aggregated_gradient: List[torch.Tensor],
        **kwargs,
    ):
        """Log losses, norms, and cosine similarities."""
        self.round = round

        # Log losses
        self._append_csv(f"{self.save_path}_losses.csv", round_losses)

        # Compute gradient metrics
        grad_norms = []
        cosine_sims = []

        flat_agg = torch.cat([g.flatten() for g in aggregated_gradient])
        agg_norm = torch.norm(flat_agg)

        for grad in client_gradients:
            flat_grad = torch.cat([g.flatten() for g in grad])
            grad_norm = torch.norm(flat_grad)
            grad_norms.append(grad_norm.item())

            dot_product = torch.dot(flat_grad, flat_agg)
            cosine_sim = (dot_product / (grad_norm * agg_norm + 1e-8)).item()
            cosine_sims.append(cosine_sim)

        # Log gradient metrics
        self._append_csv(f"{self.save_path}_grad_norms.csv", grad_norms)
        self._append_csv(f"{self.save_path}_cosine_sims.csv", cosine_sims)


class GlobalModelMetrics(BaseMetrics):
    """
    Global model evaluation on test set.

    Output files:
        - {save_path}_global_test.csv: [round, accuracy, loss]

    Args:
        save_path: Path prefix
        client_ids: Client IDs (not used, kept for API consistency)
        test_loader: DataLoader for test set

    Example:
        metrics = GlobalModelMetrics(
            "results/exp1",
            ["good0", "bad"],
            test_dataloader
        )
    """

    def __init__(self, save_path: str, client_ids: List[str], test_loader):
        super().__init__(save_path, client_ids)
        self.test_loader = test_loader
        self._init_csv(
            f"{save_path}_global_test.csv", header=["round", "accuracy", "loss"]
        )

    def __call__(
        self,
        round: int,
        server,
        selected_clients,
        round_losses: List[float],
        client_gradients: List[List[torch.Tensor]],
        client_num_samples: List[int],
        aggregated_gradient: List[torch.Tensor],
        **kwargs,
    ):
        """Evaluate global model on test set."""
        self.round = round

        total_correct = 0
        total_loss = 0.0
        total_samples = 0

        for inputs, labels in self.test_loader:
            batch_size = len(labels)
            acc, loss = server.evaluate(inputs, labels)

            total_correct += acc * batch_size
            total_loss += loss * batch_size
            total_samples += batch_size

        global_acc = total_correct / total_samples if total_samples > 0 else 0.0
        global_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Log global metrics (override _append_csv to avoid prepending round twice)
        with open(f"{self.save_path}_global_test.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.round, global_acc, global_loss])


class ComprehensiveMetrics(BaseMetrics):
    """
    Comprehensive metrics combining gradient analysis + global evaluation.

    Output files: All files from NormMetrics + GlobalModelMetrics

    Example:
        metrics = ComprehensiveMetrics(
            "results/exp1",
            ["good0", "bad"],
            test_dataloader
        )
    """

    def __init__(self, save_path: str, client_ids: List[str], test_loader):
        super().__init__(save_path, client_ids)
        self.norm_metrics = NormMetrics(save_path, client_ids)
        self.global_metrics = GlobalModelMetrics(save_path, client_ids, test_loader)

    def __call__(
        self,
        round: int,
        server,
        selected_clients,
        round_losses: List[float],
        client_gradients: List[List[torch.Tensor]],
        client_num_samples: List[int],
        aggregated_gradient: List[torch.Tensor],
        **kwargs,
    ):
        """Delegate to both sub-metrics."""
        # Use kwargs to pass all arguments to both metrics
        call_kwargs = {
            "round": round,
            "server": server,
            "selected_clients": selected_clients,
            "round_losses": round_losses,
            "client_gradients": client_gradients,
            "client_num_samples": client_num_samples,
            "aggregated_gradient": aggregated_gradient,
        }
        call_kwargs.update(kwargs)

        self.norm_metrics(**call_kwargs)
        self.global_metrics(**call_kwargs)

        self.round = round


# Backward compatibility utilities
def compute_gradient_metrics(gradients, aggregate_grad):
    """Compute gradient norms and cosine similarities (backward compat)."""
    grad_norms = []
    cosine_sims = []

    flat_agg = torch.cat([g.flatten() for g in aggregate_grad])
    agg_norm = torch.norm(flat_agg)

    for grad in gradients:
        flat_grad = torch.cat([g.flatten() for g in grad])
        grad_norm = torch.norm(flat_grad)
        grad_norms.append(grad_norm.item())

        dot_product = torch.dot(flat_grad, flat_agg)
        cosine_sim = (dot_product / (grad_norm * agg_norm + 1e-8)).item()
        cosine_sims.append(cosine_sim)

    return grad_norms, cosine_sims


def get_gradient_metrics(gradients, aggregate_grad):
    """Get metrics dict (backward compat)."""
    grad_norms, cosine_sims = compute_gradient_metrics(gradients, aggregate_grad)
    return {"grad_norms": grad_norms, "cosine_similarities": cosine_sims}
