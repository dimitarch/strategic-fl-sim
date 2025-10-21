import csv
from typing import List

import torch

# def compute_gradient_metrics(gradients, aggregate_grad):
#     """Compute metrics without creating full stacked tensor."""
#     grad_norms = []
#     cosine_sims = []

#     # Flatten aggregate once
#     flat_agg = torch.cat([g.flatten() for g in aggregate_grad])
#     agg_norm = torch.norm(flat_agg)

#     for grad in gradients:
#         # Compute gradient norm
#         flat_grad = torch.cat([g.flatten() for g in grad])
#         grad_norm = torch.norm(flat_grad)
#         grad_norms.append(grad_norm.item())

#         # Compute cosine similarity
#         dot_product = torch.dot(flat_grad, flat_agg)
#         cosine_sim = (dot_product / (grad_norm * agg_norm + 1e-8)).item()
#         cosine_sims.append(cosine_sim)

#     return grad_norms, cosine_sims


# def get_gradient_metrics(gradients, aggregate_grad):
#     """Get metrics for gradients (used in train function)."""
#     grad_norms, cosine_sims = compute_gradient_metrics(gradients, aggregate_grad)

#     return {"grad_norms": grad_norms, "cosine_similarities": cosine_sims}


class Metrics:
    """Simple metrics logger that writes to three CSV files."""

    def __init__(self, save_path: str, client_ids: List[str]):
        """
        Initialize metrics logger.
        """
        self.save_path = save_path
        self.client_ids = client_ids
        self.round = 0

        # Create three CSV files with headers
        self._init_csv(f"{save_path}_losses.csv")
        self._init_csv(f"{save_path}_grad_norms.csv")
        self._init_csv(f"{save_path}_cosine_sims.csv")

    def _init_csv(self, filepath: str):
        """Initialize CSV file with header."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round"] + self.client_ids)

    def log(
        self,
        losses: List[float],
        gradients: List[List[torch.Tensor]],
        aggregate: List[torch.Tensor],
    ):
        """
        Compute metrics and log to CSV.
        """
        # Compute gradient norms
        grad_norms = []
        for grad in gradients:
            flat_grad = torch.cat([g.flatten() for g in grad])
            grad_norms.append(torch.norm(flat_grad).item())

        # Compute cosine similarities
        cosine_sims = []
        flat_agg = torch.cat([g.flatten() for g in aggregate])
        agg_norm = torch.norm(flat_agg)

        for grad in gradients:
            flat_grad = torch.cat([g.flatten() for g in grad])
            grad_norm = torch.norm(flat_grad)
            dot = torch.dot(flat_grad, flat_agg)
            cosine_sims.append((dot / (grad_norm * agg_norm + 1e-8)).item())

        # Write to three CSV files
        self._append_csv(f"{self.save_path}_losses.csv", losses)
        self._append_csv(f"{self.save_path}_grad_norms.csv", grad_norms)
        self._append_csv(f"{self.save_path}_cosine_sims.csv", cosine_sims)

        self.round += 1

    def _append_csv(self, filepath: str, values: List[float]):
        """Append row to CSV file."""
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.round] + values)

    def __call__(
        self,
        losses: List[float],
        gradients: List[List[torch.Tensor]],
        aggregate: List[torch.Tensor],
    ):
        return self.log(losses, gradients, aggregate)
