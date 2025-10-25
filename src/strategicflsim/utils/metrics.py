import csv
from typing import List, Optional

import torch


class BaseMetrics:
    """
    Base metrics class providing CSV logging infrastructure.

    To create custom metrics, inherit from this class and override __call__.

    Example:
        class MyMetrics(BaseMetrics):
            def __init__(self, save_path, client_ids):
                super().__init__(save_path, client_ids)
                self._init_csv(f"{save_path}_my_metric.csv")

            def __call__(self, server, clients, round_losses, ...):
                # Your custom logic here
                my_values = [...]
                self._append_csv(f"{save_path}_my_metric.csv", my_values)
                self.round += 1
    """

    def __init__(self, save_path: str, client_ids: List[str]):
        """
        Initialize base metrics logger.

        Args:
            save_path: Path prefix for CSV files (e.g., "results/experiment1")
            client_ids: List of client identifiers (e.g., ["good0", "good1", "bad"])
        """
        self.save_path = save_path
        self.client_ids = client_ids
        self.round = 0

    def _init_csv(self, filepath: str, header: Optional[List[str]] = None):
        """
        Initialize CSV file with header.

        Args:
            filepath: Full path to CSV file
            header: Custom header row (default: ["round"] + client_ids)
        """
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header or ["round"] + self.client_ids)

    def _append_csv(self, filepath: str, values: List[float]):
        """
        Append row to CSV file.

        Args:
            filepath: Full path to CSV file
            values: List of values to append (automatically prepends round number)
        """
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.round] + values)

    def __call__(
        self,
        server,
        selected_clients,
        round_losses,
        client_gradients,
        client_num_samples,
        aggregated_gradient,
    ):
        """
        Callback invoked after each training round.

        Args:
            server: Server instance (READ-ONLY)
            selected_clients: List of clients selected this round
            round_losses: List of training losses from each client
            client_gradients: List of gradient tensors from each client
            client_num_samples: List of sample counts from each client
            aggregated_gradient: Server's aggregated gradient

        **WARNING**: Treat server as read-only. Do not modify server state.

        Override this method in subclasses to implement custom metrics.
        """
        raise NotImplementedError(
            "Subclasses must implement __call__. See BaseMetrics docstring for example."
        )


class NormMetrics(BaseMetrics):
    """
    Standard metrics: losses, gradient norms, and cosine similarities.

    Output files:
        - {save_path}_losses.csv: Training losses per client
        - {save_path}_grad_norms.csv: Gradient norms per client
        - {save_path}_cosine_sims.csv: Cosine similarity with aggregate

    Example:
        metrics = NormMetrics("results/exp1", ["good0", "good1", "bad"])
        server.train(clients, T=100, metrics=metrics)
    """

    def __init__(self, save_path: str, client_ids: List[str]):
        super().__init__(save_path, client_ids)

        # Initialize CSV files
        self._init_csv(f"{save_path}_losses.csv")
        self._init_csv(f"{save_path}_grad_norms.csv")
        self._init_csv(f"{save_path}_cosine_sims.csv")

    def __call__(
        self,
        server,
        selected_clients,
        round_losses,
        client_gradients,
        client_num_samples,
        aggregated_gradient,
    ):
        """Log losses, gradient norms, and cosine similarities."""
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

        self.round += 1


class GlobalModelMetrics(BaseMetrics):
    """
    Global evaluation metrics: test accuracy.

    Output files:
        - {save_path}_global_test.csv: Global model accuracy and loss

    Example:
        metrics = GlobalModelMetrics(
            save_path="results/exp1",
            client_ids=["good0", "good1", "bad"],
            test_loader=test_dataloader
        )
        server.train(clients, T=100, metrics=metrics)
    """

    def __init__(self, save_path: str, client_ids: List[str], test_loader):
        super().__init__(save_path, client_ids)
        self.test_loader = test_loader

        # Initialize CSV files
        self._init_csv(
            f"{save_path}_global_test.csv", header=["round", "accuracy", "loss"]
        )

    def __call__(
        self,
        server,
        selected_clients,
        round_losses,
        client_gradients,
        client_num_samples,
        aggregated_gradient,
    ):
        """Evaluate global model on test set."""
        # Evaluate global model
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

        # Log global metrics
        with open(f"{self.save_path}_global_test.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.round, global_acc, global_loss])

        self.round += 1


class ComprehensiveMetrics(BaseMetrics):
    """
    Comprehensive metrics combining gradient analysis and global evaluation.

    Output files:
        - {save_path}_losses.csv: Training losses per client
        - {save_path}_grad_norms.csv: Gradient norms per client
        - {save_path}_cosine_sims.csv: Cosine similarity with aggregate
        - {save_path}_global_test.csv: Global model accuracy and loss

    Example:
        metrics = ComprehensiveMetrics(
            save_path="results/exp1",
            client_ids=["good0", "good1", "bad"],
            test_loader=test_dataloader
        )
        server.train(clients, T=100, metrics=metrics)
    """

    def __init__(self, save_path: str, client_ids: List[str], test_loader):
        super().__init__(save_path, client_ids)

        # Compose existing metrics classes
        self.norm_metrics = NormMetrics(save_path, client_ids)
        self.global_metrics = GlobalModelMetrics(save_path, client_ids, test_loader)

    def __call__(
        self,
        server,
        selected_clients,
        round_losses,
        client_gradients,
        client_num_samples,
        aggregated_gradient,
    ):
        """Delegate to both NormMetrics and GlobalModelMetrics."""
        # Log gradient metrics
        self.norm_metrics(
            server,
            selected_clients,
            round_losses,
            client_gradients,
            client_num_samples,
            aggregated_gradient,
        )

        # Log global evaluation
        self.global_metrics(
            server,
            selected_clients,
            round_losses,
            client_gradients,
            client_num_samples,
            aggregated_gradient,
        )

        # Sync round counter
        self.round = self.norm_metrics.round


# Keep utility functions for backward compatibility
def compute_gradient_metrics(gradients, aggregate_grad):
    """Compute gradient norms and cosine similarities."""
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
    """Get metrics dictionary (for backward compatibility)."""
    grad_norms, cosine_sims = compute_gradient_metrics(gradients, aggregate_grad)
    return {"grad_norms": grad_norms, "cosine_similarities": cosine_sims}
