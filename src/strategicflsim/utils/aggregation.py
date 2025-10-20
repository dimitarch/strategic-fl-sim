from functools import partial
from typing import List, Optional

import torch


def mean_aggregate(
    gradients: List[List[torch.Tensor]], sizes: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """
    Aggregate gradients using simple mean (FedAvg). Computes element-wise average of all client gradients. Ignores dataset sizes - treats all clients equally.
    """
    num_layers = len(gradients[0])
    aggregated = []

    for layer_idx in range(num_layers):
        layer_grads = torch.stack([grad[layer_idx] for grad in gradients])
        aggregated.append(layer_grads.mean(dim=0))

    return aggregated


def weighted_average_aggregate(
    gradients: List[List[torch.Tensor]], sizes: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """
    Aggregate gradients using weighted average by dataset size. Weights each client's gradient proportionally to their dataset size. Falls back to equal weights if sizes not provided.
    """
    num_clients = len(gradients)
    num_layers = len(gradients[0])

    # Default here to equal weights if sizes not provided
    if sizes is None:
        sizes = [1] * num_clients

    total_size = sum(sizes)
    weights = [size / total_size for size in sizes]

    aggregated = []

    for layer_idx in range(num_layers):
        weighted_sum = torch.zeros_like(gradients[0][layer_idx])

        for client_idx, grad in enumerate(gradients):
            weighted_sum += weights[client_idx] * grad[layer_idx]

        aggregated.append(weighted_sum)

    return aggregated


def median_aggregate(
    gradients: List[List[torch.Tensor]], sizes: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """
    Aggregate gradients using coordinate-wise median. Computes median of each gradient coordinate across clients. Ignores dataset sizes.
    """
    num_layers = len(gradients[0])
    aggregated = []

    for layer_idx in range(num_layers):
        layer_grads = torch.stack([grad[layer_idx] for grad in gradients])
        aggregated.append(torch.median(layer_grads, dim=0).values)

    return aggregated


def trimmed_mean_aggregate(
    gradients: List[List[torch.Tensor]],
    sizes: Optional[List[int]] = None,
    trim_ratio: float = 0.1,
) -> List[torch.Tensor]:
    """
    Aggregate gradients using trimmed mean. Removes a fraction of extreme values before averaging. Uses weighted trimming if sizes provided.

    Comments explain the logic.
    """
    num_clients = len(gradients)
    num_layers = len(gradients[0])

    # Calculate how many clients to trim from each end
    num_trim = max(1, int(num_clients * trim_ratio))

    # Default to equal weights if sizes not provided
    if sizes is None:
        sizes = [1] * num_clients

    total_size = sum(sizes)
    weights = [size / total_size for size in sizes]

    aggregated = []

    for layer_idx in range(num_layers):
        # Get this layer's gradients from all clients
        layer_grads = [grad[layer_idx] for grad in gradients]

        # Flatten for easier trimming
        flattened = torch.stack(
            [g.flatten() for g in layer_grads]
        )  # [num_clients, num_params]
        original_shape = layer_grads[0].shape

        # Weight each client's contribution
        weighted_grads = torch.stack(
            [weights[i] * flattened[i] for i in range(num_clients)]
        )

        # Sort along client dimension for each parameter
        sorted_grads, _ = torch.sort(weighted_grads, dim=0)

        # Trim extremes and compute mean
        if num_clients > 2 * num_trim:
            trimmed = sorted_grads[num_trim:-num_trim]
            aggregated_flat = trimmed.mean(dim=0)
        else:
            # Not enough clients to trim, just use mean
            aggregated_flat = weighted_grads.mean(dim=0)

        # Reshape back to original
        aggregated.append(aggregated_flat.reshape(original_shape))

    return aggregated


def krum_aggregate(
    gradients: List[List[torch.Tensor]],
    sizes: Optional[List[int]] = None,
    num_byzantine: int = 0,
) -> List[torch.Tensor]:
    """
    Aggregate using the Krum algorithm, which selects the most "representative" gradient. In other words, choose the gradient that is closest to the majority of other gradients, given we know the number of Byzantine clients.

    Comments explain the logic.
    """
    num_clients = len(gradients)

    # Flatten all gradients for distance computation
    flattened_grads = []
    for grad in gradients:
        flat = torch.cat([g.flatten() for g in grad])
        flattened_grads.append(flat)

    # Compute pairwise distances
    distances = torch.zeros(num_clients, num_clients)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = torch.norm(flattened_grads[i] - flattened_grads[j])
            distances[i, j] = dist
            distances[j, i] = dist

    # For each client, sum distances to closest (n - num_byzantine - 2) clients
    num_closest = num_clients - num_byzantine - 2
    scores = torch.zeros(num_clients)

    for i in range(num_clients):
        closest_distances = torch.topk(
            distances[i], k=num_closest, largest=False
        ).values
        scores[i] = closest_distances.sum()

    # Select client with minimum score
    selected_idx = torch.argmin(scores).item()

    return gradients[selected_idx]


def get_aggregate(method: str = "mean", **kwargs):
    """
    Factory function to get aggregation method by name.
    """
    if method == "mean":
        return mean_aggregate
    elif method in ["weighted_mean", "weighted_average"]:
        return weighted_average_aggregate
    elif method == "median":
        return median_aggregate
    elif method == "trimmed_mean":
        trim_ratio = kwargs.get("trim_ratio", 0.1)
        return partial(trimmed_mean_aggregate, trim_ratio=trim_ratio)
    elif method == "krum":
        num_byzantine = kwargs.get("num_byzantine", 0)
        return partial(krum_aggregate, num_byzantine=num_byzantine)
    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. "
            f"Available: mean, weighted_mean, median, trimmed_mean, krum"
        )
