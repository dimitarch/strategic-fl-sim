from functools import partial
from typing import List, Optional

import torch


def mean_aggregate(
    gradients: List[List[torch.Tensor]], num_sample: Optional[List[int]] = None
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
    gradients: List[List[torch.Tensor]], num_sample: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """
    Aggregate gradients using weighted average by dataset size. Weights each client's gradient proportionally to their dataset size. Falls back to equal weights if sizes not provided.
    """
    num_clients = len(gradients)
    num_layers = len(gradients[0])

    # Default here to equal weights if sizes not provided
    if num_sample is None:
        num_sample = [1] * num_clients

    total_size = sum(num_sample)
    weights = [size / total_size for size in num_sample]

    aggregated = []

    for layer_idx in range(num_layers):
        weighted_sum = torch.zeros_like(gradients[0][layer_idx])

        for client_idx, grad in enumerate(gradients):
            weighted_sum += weights[client_idx] * grad[layer_idx]

        aggregated.append(weighted_sum)

    return aggregated


def median_aggregate(
    gradients: List[List[torch.Tensor]], num_sample: Optional[List[int]] = None
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


def geometric_mean_aggregate(
    gradients: List[List[torch.Tensor]], num_sample: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """
    Coordinate-wise geometric mean. Sign-preserving: sign(mean) * exp(mean(log(|x|))).
    """
    num_layers = len(gradients[0])
    aggregated = []

    for layer_idx in range(num_layers):
        layer_grads = torch.stack([grad[layer_idx] for grad in gradients])
        signs = torch.sign(layer_grads)
        abs_grads = torch.abs(layer_grads) + 1e-10

        log_abs = torch.log(abs_grads)
        mean_log = log_abs.mean(dim=0)
        geom_mean = torch.exp(mean_log)

        final_sign = torch.sign(signs.sum(dim=0))
        aggregated.append(final_sign * geom_mean)

    return aggregated


def weighted_geometric_mean_aggregate(
    gradients: List[List[torch.Tensor]], num_sample: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """Weighted coordinate-wise geometric mean by dataset size."""
    num_clients = len(gradients)
    num_layers = len(gradients[0])

    if num_sample is None:
        num_sample = [1] * num_clients

    total_size = sum(num_sample)
    weights = torch.tensor([size / total_size for size in num_sample])

    aggregated = []

    for layer_idx in range(num_layers):
        layer_grads = torch.stack([grad[layer_idx] for grad in gradients])
        signs = torch.sign(layer_grads)
        abs_grads = torch.abs(layer_grads) + 1e-10

        log_abs = torch.log(abs_grads)
        weighted_log = (log_abs * weights.view(-1, *([1] * (log_abs.ndim - 1)))).sum(
            dim=0
        )
        geom_mean = torch.exp(weighted_log)

        weighted_signs = (signs * weights.view(-1, *([1] * (signs.ndim - 1)))).sum(
            dim=0
        )
        final_sign = torch.sign(weighted_signs)
        aggregated.append(final_sign * geom_mean)

    return aggregated


def trimmed_mean_aggregate(
    gradients: List[List[torch.Tensor]],
    num_sample: Optional[List[int]] = None,
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
    if num_sample is None:
        num_sample = [1] * num_clients

    total_size = sum(num_sample)
    weights = [size / total_size for size in num_sample]

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
    num_sample: Optional[List[int]] = None,
    num_byzantine: int = 0,
) -> List[torch.Tensor]:
    """Krum: select single most representative gradient."""
    num_clients = len(gradients)
    flattened_grads = torch.stack(
        [torch.cat([g.flatten() for g in grad]) for grad in gradients]
    )

    distances = torch.cdist(flattened_grads, flattened_grads, p=2)

    num_closest = num_clients - num_byzantine - 2
    scores = (
        torch.topk(distances, k=num_closest + 1, dim=1, largest=False)
        .values[:, 1:]
        .sum(dim=1)
    )

    selected_idx = torch.argmin(scores).item()
    return gradients[selected_idx]


def multi_krum_aggregate(
    gradients: List[List[torch.Tensor]],
    num_sample: Optional[List[int]] = None,
    num_byzantine: int = 0,
    num_selected: Optional[int] = None,
) -> List[torch.Tensor]:
    """Multi-Krum: average top-m most representative gradients."""
    num_clients = len(gradients)

    if num_selected is None:
        num_selected = num_clients - num_byzantine
    num_selected = min(num_selected, num_clients)

    flattened_grads = torch.stack(
        [torch.cat([g.flatten() for g in grad]) for grad in gradients]
    )

    distances = torch.cdist(flattened_grads, flattened_grads, p=2)

    num_closest = num_clients - num_byzantine - 2
    scores = (
        torch.topk(distances, k=num_closest + 1, dim=1, largest=False)
        .values[:, 1:]
        .sum(dim=1)
    )

    selected_indices = torch.topk(
        scores, k=num_selected, largest=False
    ).indices.tolist()
    selected_gradients = [gradients[i] for i in selected_indices]

    return mean_aggregate(selected_gradients)


def get_aggregate(method: str = "mean", **kwargs):
    """Factory function to get aggregation method by name."""
    if method == "mean":
        return mean_aggregate
    elif method in ["weighted_mean", "weighted_average"]:
        return weighted_average_aggregate
    elif method == "median":
        return median_aggregate
    elif method == "geometric_mean":
        return geometric_mean_aggregate
    elif method == "weighted_geometric_mean":
        return weighted_geometric_mean_aggregate
    elif method == "trimmed_mean":
        trim_ratio = kwargs.get("trim_ratio", 0.1)
        return partial(trimmed_mean_aggregate, trim_ratio=trim_ratio)
    elif method == "krum":
        num_byzantine = kwargs.get("num_byzantine", 0)
        return partial(krum_aggregate, num_byzantine=num_byzantine)
    elif method == "multi_krum":
        num_byzantine = kwargs.get("num_byzantine", 0)
        num_selected = kwargs.get("num_selected", None)
        return partial(
            multi_krum_aggregate, num_byzantine=num_byzantine, num_selected=num_selected
        )
    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. "
            f"Available: mean, weighted_mean, median, geometric_mean, weighted_geometric_mean, "
            f"trimmed_mean, krum, multi_krum"
        )


class AdaptiveAggregator:
    """
    Adaptive aggregation that switches methods based on gradient statistics.

    Monitors cosine similarity variance to detect adversarial behavior.
    Switches to robust aggregation when divergence detected.
    """

    def __init__(
        self,
        default_method: str = "mean",
        robust_method: str = "trimmed_mean",
        threshold: float = 0.3,
        window_size: int = 5,
        **method_kwargs,
    ):
        self.default_fn = get_aggregate(default_method, **method_kwargs)
        self.robust_fn = get_aggregate(robust_method, **method_kwargs)
        self.threshold = threshold
        self.window_size = window_size
        self.cosine_history = []
        self.use_robust = False

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_sample: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        flattened = torch.stack(
            [torch.cat([g.flatten() for g in grad]) for grad in gradients]
        )

        mean_grad = flattened.mean(dim=0)
        cosines = torch.nn.functional.cosine_similarity(
            flattened, mean_grad.unsqueeze(0), dim=1
        )

        self.cosine_history.append(cosines.std().item())
        if len(self.cosine_history) > self.window_size:
            self.cosine_history.pop(0)

        if len(self.cosine_history) == self.window_size:
            recent_variance = sum(self.cosine_history) / self.window_size
            self.use_robust = recent_variance > self.threshold

        return (
            self.robust_fn(gradients, num_sample)
            if self.use_robust
            else self.default_fn(gradients, num_sample)
        )
