import torch


def mean_aggregate(gradients):
    """Aggregate gradients using mean aggregation."""
    n = len(gradients[0])
    aggregated = []

    for i in range(len(gradients[0])):
        weighted_grads = [grad[i] / n for grad in gradients]
        aggregated.append(torch.sum(torch.stack(weighted_grads), 0))

    return aggregated


def weighted_average_aggregate(gradients, sizes):
    """Aggregate gradients using weighted average."""
    total_size = sum(sizes)
    aggregated = []

    for i in range(len(gradients[0])):
        weighted_grads = [
            grad[i] * size / total_size for grad, size in zip(gradients, sizes)
        ]
        aggregated.append(torch.sum(torch.stack(weighted_grads), 0))

    return aggregated


def median_aggregate(gradients):
    """Aggregate gradients using coordinate-wise median aggregation."""
    aggregated = []

    for i in range(len(gradients[0])):
        stacked_grads = torch.stack([grad[i] for grad in gradients])
        median_grad = torch.median(stacked_grads, 0).values
        aggregated.append(median_grad)

    return aggregated


def trimmed_mean_aggregate(gradients, sizes):
    """Aggregate gradients using trimmed mean (removes largest gradient)."""
    total_size = sum(sizes)
    aggregated = []

    for i in range(len(gradients[0])):
        weighted_grads = torch.stack(
            [grad[i] * size / total_size for grad, size in zip(gradients, sizes)]
        )

        # Sort and remove the largest value
        sorted_vals, _ = torch.sort(weighted_grads.view(-1))
        trimmed_vals = sorted_vals[:-1] if sorted_vals.numel() > 1 else sorted_vals
        aggregated.append(trimmed_vals.mean().expand_as(gradients[0][i]))

    return aggregated


def get_aggregate(method: str = "mean"):
    if method == "median":
        return median_aggregate
    elif method == "trimmed_mean":
        return trimmed_mean_aggregate
    elif method == "weighted_mean":
        return weighted_average_aggregate
    else:
        return mean_aggregate
