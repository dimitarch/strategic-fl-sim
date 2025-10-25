from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class BaseAggregator(ABC):
    """
    Abstract base class for gradient aggregation methods.

    All aggregator classes should inherit from this and implement __call__.
    Aggregators operate on lists of gradient tensors from multiple clients.

    Example:
        class MyAggregator(BaseAggregator):
            def __init__(self, param):
                self.param = param

            def __call__(self, gradients, num_samples=None):
                # Your aggregation logic
                return aggregated_gradient
    """

    @abstractmethod
    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Aggregate gradients from multiple clients.

        Args:
            gradients: List of gradient lists, one per client
                      Each gradient list contains tensors for each model layer
            num_samples: Optional list of dataset sizes per client
                        Used for weighted aggregation

        Returns:
            Aggregated gradient as list of tensors (one per layer)
        """
        pass

    def __repr__(self):
        """String representation showing aggregator parameters."""
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


class MeanAggregator(BaseAggregator):
    """
    Simple mean aggregation (FedAvg).

    Computes element-wise average of all client gradients.
    Treats all clients equally regardless of dataset size.

    Mathematical form:
        mean(g₁, ..., gₙ) = (1/n) Σᵢ gᵢ

    Example:
        aggregator = MeanAggregator()
        # All clients weighted equally
    """

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Compute simple mean of gradients."""
        num_layers = len(gradients[0])
        aggregated = []

        for layer_idx in range(num_layers):
            layer_grads = torch.stack([grad[layer_idx] for grad in gradients])
            aggregated.append(layer_grads.mean(dim=0))

        return aggregated


class WeightedAverageAggregator(BaseAggregator):
    """
    Weighted average by dataset size (standard FedAvg).

    Weights each client's gradient proportionally to their dataset size.
    Falls back to equal weights if sizes not provided.

    Mathematical form:
        weighted_avg(g₁, ..., gₙ) = Σᵢ (nᵢ/Σⱼnⱼ) gᵢ

    where nᵢ is the dataset size of client i.

    Example:
        aggregator = WeightedAverageAggregator()
        # Clients with more data have more influence
    """

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Compute weighted average of gradients."""
        num_clients = len(gradients)
        num_layers = len(gradients[0])

        # Default to equal weights if sizes not provided
        if num_samples is None:
            num_samples = [1] * num_clients

        total_size = sum(num_samples)
        weights = [size / total_size for size in num_samples]

        aggregated = []

        for layer_idx in range(num_layers):
            weighted_sum = torch.zeros_like(gradients[0][layer_idx])

            for client_idx, grad in enumerate(gradients):
                weighted_sum += weights[client_idx] * grad[layer_idx]

            aggregated.append(weighted_sum)

        return aggregated


class MedianAggregator(BaseAggregator):
    """
    Coordinate-wise median aggregation.

    Computes median of each gradient coordinate across clients.
    Robust to outliers but ignores dataset sizes.

    Mathematical form:
        [median(g₁, ..., gₙ)]ₖ = median([g₁]ₖ, ..., [gₙ]ₖ)

    where [·]ₖ denotes the k-th coordinate.

    Example:
        aggregator = MedianAggregator()
        # Robust to up to 50% Byzantine clients
    """

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Compute coordinate-wise median."""
        num_layers = len(gradients[0])
        aggregated = []

        for layer_idx in range(num_layers):
            layer_grads = torch.stack([grad[layer_idx] for grad in gradients])
            aggregated.append(torch.median(layer_grads, dim=0).values)

        return aggregated


class TrimmedMeanAggregator(BaseAggregator):
    """
    Trimmed mean aggregation.

    Removes a fraction of extreme values before averaging.
    Provides robustness while being less aggressive than median.

    Mathematical form:
        For each coordinate, sort values, remove top/bottom trim_ratio fraction,
        then average the remaining values.

    Args:
        trim_ratio: Fraction of clients to trim from each end (default: 0.1)
                   E.g., 0.1 means remove 10% largest and 10% smallest

    Example:
        # Remove 20% extremes (10% top, 10% bottom)
        aggregator = TrimmedMeanAggregator(trim_ratio=0.1)

        # More aggressive trimming
        aggregator = TrimmedMeanAggregator(trim_ratio=0.2)
    """

    def __init__(self, trim_ratio: float = 0.1):
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Compute trimmed mean of gradients."""
        num_clients = len(gradients)
        num_layers = len(gradients[0])

        # Calculate how many clients to trim from each end
        num_trim = max(1, int(num_clients * self.trim_ratio))

        # Default to equal weights if sizes not provided
        if num_samples is None:
            num_samples = [1] * num_clients

        total_size = sum(num_samples)
        weights = [size / total_size for size in num_samples]

        aggregated = []

        for layer_idx in range(num_layers):
            layer_grads = [grad[layer_idx] for grad in gradients]

            # Flatten for easier trimming
            flattened = torch.stack([g.flatten() for g in layer_grads])
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


class GeometricMeanAggregator(BaseAggregator):
    """
    Coordinate-wise geometric mean aggregation.

    Computes geometric mean while preserving sign information.
    More robust to outliers than arithmetic mean.

    Mathematical form:
        geom_mean(g₁, ..., gₙ) = sign(Σᵢ sign(gᵢ)) ⊙ exp((1/n) Σᵢ log|gᵢ|)

    where ⊙ denotes element-wise multiplication.

    Example:
        aggregator = GeometricMeanAggregator()
        # Reduces impact of large gradient manipulations
    """

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Compute coordinate-wise geometric mean."""
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


class WeightedGeometricMeanAggregator(BaseAggregator):
    """
    Weighted coordinate-wise geometric mean by dataset size.

    Combines geometric mean robustness with dataset size weighting.

    Mathematical form:
        weighted_geom_mean(g₁, ..., gₙ) = sign(Σᵢ wᵢ sign(gᵢ)) ⊙ exp(Σᵢ wᵢ log|gᵢ|)

    where wᵢ = nᵢ/Σⱼnⱼ are normalized weights.

    Example:
        aggregator = WeightedGeometricMeanAggregator()
        # Geometric mean with larger clients having more influence
    """

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Compute weighted geometric mean."""
        num_clients = len(gradients)
        num_layers = len(gradients[0])

        if num_samples is None:
            num_samples = [1] * num_clients

        total_size = sum(num_samples)
        weights = torch.tensor([size / total_size for size in num_samples])

        aggregated = []

        for layer_idx in range(num_layers):
            layer_grads = torch.stack([grad[layer_idx] for grad in gradients])
            signs = torch.sign(layer_grads)
            abs_grads = torch.abs(layer_grads) + 1e-10

            log_abs = torch.log(abs_grads)
            weighted_log = (
                log_abs * weights.view(-1, *([1] * (log_abs.ndim - 1)))
            ).sum(dim=0)
            geom_mean = torch.exp(weighted_log)

            weighted_signs = (signs * weights.view(-1, *([1] * (signs.ndim - 1)))).sum(
                dim=0
            )
            final_sign = torch.sign(weighted_signs)
            aggregated.append(final_sign * geom_mean)

        return aggregated


class KrumAggregator(BaseAggregator):
    """
    Krum aggregation: select single most representative gradient.

    Selects the gradient with minimum sum of distances to its k nearest neighbors.
    Provides Byzantine robustness by selecting a single trusted gradient.

    Args:
        num_byzantine: Expected number of Byzantine clients (default: 0)

    Mathematical form:
        Let d(gᵢ, gⱼ) = ||gᵢ - gⱼ||₂. For each client i, compute:
        score(i) = Σⱼ∈N(i) d(gᵢ, gⱼ)
        where N(i) are the n-f-2 nearest neighbors of i.
        Select i* = argmin score(i).

    Example:
        # Tolerate up to 2 Byzantine clients
        aggregator = KrumAggregator(num_byzantine=2)

        # No expected Byzantine clients
        aggregator = KrumAggregator(num_byzantine=0)
    """

    def __init__(self, num_byzantine: int = 0):
        if num_byzantine < 0:
            raise ValueError("num_byzantine must be non-negative")
        self.num_byzantine = num_byzantine

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Select single most representative gradient."""
        num_clients = len(gradients)
        flattened_grads = torch.stack(
            [torch.cat([g.flatten() for g in grad]) for grad in gradients]
        )

        distances = torch.cdist(flattened_grads, flattened_grads, p=2)

        num_closest = num_clients - self.num_byzantine - 2
        scores = (
            torch.topk(distances, k=num_closest + 1, dim=1, largest=False)
            .values[:, 1:]
            .sum(dim=1)
        )

        selected_idx = torch.argmin(scores).item()
        return gradients[selected_idx]


class MultiKrumAggregator(BaseAggregator):
    """
    Multi-Krum aggregation: average top-m most representative gradients.

    Extension of Krum that averages multiple selected gradients instead of one.
    Provides better utility while maintaining Byzantine robustness.

    Args:
        num_byzantine: Expected number of Byzantine clients (default: 0)
        num_selected: Number of gradients to select and average
                     If None, uses n - num_byzantine (default: None)

    Mathematical form:
        1. Compute Krum scores for all clients (same as Krum)
        2. Select m clients with lowest scores
        3. Return average of selected gradients

    Example:
        # Select 5 best gradients, tolerate 2 Byzantine
        aggregator = MultiKrumAggregator(num_byzantine=2, num_selected=5)

        # Auto-select all non-Byzantine (n-f gradients)
        aggregator = MultiKrumAggregator(num_byzantine=2)
    """

    def __init__(self, num_byzantine: int = 0, num_selected: Optional[int] = None):
        if num_byzantine < 0:
            raise ValueError("num_byzantine must be non-negative")
        self.num_byzantine = num_byzantine
        self.num_selected = num_selected

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Average top-m most representative gradients."""
        num_clients = len(gradients)

        num_selected = self.num_selected
        if num_selected is None:
            num_selected = num_clients - self.num_byzantine
        num_selected = min(num_selected, num_clients)

        flattened_grads = torch.stack(
            [torch.cat([g.flatten() for g in grad]) for grad in gradients]
        )

        distances = torch.cdist(flattened_grads, flattened_grads, p=2)

        num_closest = num_clients - self.num_byzantine - 2
        scores = (
            torch.topk(distances, k=num_closest + 1, dim=1, largest=False)
            .values[:, 1:]
            .sum(dim=1)
        )

        selected_indices = torch.topk(
            scores, k=num_selected, largest=False
        ).indices.tolist()
        selected_gradients = [gradients[i] for i in selected_indices]

        return MeanAggregator()(selected_gradients)


class NormClippingAggregator(BaseAggregator):
    """
    Gradient norm clipping aggregator.

    Clips each gradient to maximum norm before aggregation.
    Simple defense against large gradient attacks.

    Args:
        max_norm: Maximum allowed gradient norm (default: 10.0)
        base_aggregator: Underlying aggregator to use after clipping
                        (default: MeanAggregator())

    Mathematical form:
        For each gradient gᵢ:
        clip(gᵢ) = gᵢ if ||gᵢ|| ≤ max_norm else (max_norm/||gᵢ||) * gᵢ
        Then apply base aggregation to clipped gradients.

    Example:
        # Clip to norm 10, then average
        aggregator = NormClippingAggregator(max_norm=10.0)

        # Clip then use median
        aggregator = NormClippingAggregator(
            max_norm=15.0,
            base_aggregator=MedianAggregator()
        )
    """

    def __init__(
        self, max_norm: float = 10.0, base_aggregator: Optional[BaseAggregator] = None
    ):
        if max_norm <= 0:
            raise ValueError("max_norm must be positive")
        self.max_norm = max_norm
        self.base_aggregator = base_aggregator or MeanAggregator()

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Clip gradients then aggregate."""
        clipped_gradients = []

        for grad in gradients:
            # Compute total gradient norm
            flat_grad = torch.cat([g.flatten() for g in grad])
            grad_norm = torch.norm(flat_grad)

            # Clip if necessary
            if grad_norm > self.max_norm:
                scale = self.max_norm / grad_norm
                clipped_grad = [g * scale for g in grad]
            else:
                clipped_grad = grad

            clipped_gradients.append(clipped_grad)

        # Apply base aggregation to clipped gradients
        return self.base_aggregator(clipped_gradients, num_samples)


class AdaptiveAggregator(BaseAggregator):
    """
    Adaptive aggregation that switches methods based on gradient statistics.

    Monitors cosine similarity variance to detect adversarial behavior.
    Switches to robust aggregation when divergence detected.

    Args:
        default_aggregator: Aggregator to use normally (default: MeanAggregator())
        robust_aggregator: Aggregator to use when attack detected
                          (default: TrimmedMeanAggregator(trim_ratio=0.1))
        threshold: Cosine similarity variance threshold for switching (default: 0.3)
        window_size: Number of rounds to monitor (default: 5)

    Example:
        # Switch from mean to trimmed mean when suspicious
        aggregator = AdaptiveAggregator(
            default_aggregator=MeanAggregator(),
            robust_aggregator=TrimmedMeanAggregator(trim_ratio=0.2),
            threshold=0.3
        )

        # More sensitive detection
        aggregator = AdaptiveAggregator(threshold=0.2, window_size=3)
    """

    def __init__(
        self,
        default_aggregator: Optional[BaseAggregator] = None,
        robust_aggregator: Optional[BaseAggregator] = None,
        threshold: float = 0.3,
        window_size: int = 5,
    ):
        self.default_aggregator = default_aggregator or MeanAggregator()
        self.robust_aggregator = robust_aggregator or TrimmedMeanAggregator(
            trim_ratio=0.1
        )
        self.threshold = threshold
        self.window_size = window_size
        self.cosine_history = []
        self.use_robust = False

    def __call__(
        self,
        gradients: List[List[torch.Tensor]],
        num_samples: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Adaptively select aggregation method."""
        # Flatten gradients for analysis
        flattened = torch.stack(
            [torch.cat([g.flatten() for g in grad]) for grad in gradients]
        )

        # Compute mean and cosine similarities
        mean_grad = flattened.mean(dim=0)
        cosines = torch.nn.functional.cosine_similarity(
            flattened, mean_grad.unsqueeze(0), dim=1
        )

        # Track cosine similarity variance
        self.cosine_history.append(cosines.std().item())
        if len(self.cosine_history) > self.window_size:
            self.cosine_history.pop(0)

        # Detect if we should use robust aggregation
        if len(self.cosine_history) == self.window_size:
            recent_variance = sum(self.cosine_history) / self.window_size
            self.use_robust = recent_variance > self.threshold

        # Apply appropriate aggregation
        return (
            self.robust_aggregator(gradients, num_samples)
            if self.use_robust
            else self.default_aggregator(gradients, num_samples)
        )


# Factory function for backward compatibility
def get_aggregate(method: str = "mean", **kwargs) -> BaseAggregator:
    """
    Factory function to get aggregator by name.

    Args:
        method: Aggregation method name
        **kwargs: Additional parameters for the aggregator

    Returns:
        Aggregator instance

    Available methods:
        - "mean": MeanAggregator
        - "weighted_mean" / "weighted_average": WeightedAverageAggregator
        - "median": MedianAggregator
        - "trimmed_mean": TrimmedMeanAggregator(trim_ratio)
        - "geometric_mean": GeometricMeanAggregator
        - "weighted_geometric_mean": WeightedGeometricMeanAggregator
        - "krum": KrumAggregator(num_byzantine)
        - "multi_krum": MultiKrumAggregator(num_byzantine, num_selected)
        - "norm_clipping": NormClippingAggregator(max_norm)
        - "adaptive": AdaptiveAggregator(threshold, window_size)

    Example:
        >>> aggregator = get_aggregate("mean")
        >>> aggregator = get_aggregate("trimmed_mean", trim_ratio=0.2)
        >>> aggregator = get_aggregate("krum", num_byzantine=2)
    """
    if method == "mean":
        return MeanAggregator()
    elif method in ["weighted_mean", "weighted_average"]:
        return WeightedAverageAggregator()
    elif method == "median":
        return MedianAggregator()
    elif method == "trimmed_mean":
        trim_ratio = kwargs.get("trim_ratio", 0.1)
        return TrimmedMeanAggregator(trim_ratio=trim_ratio)
    elif method == "geometric_mean":
        return GeometricMeanAggregator()
    elif method == "weighted_geometric_mean":
        return WeightedGeometricMeanAggregator()
    elif method == "krum":
        num_byzantine = kwargs.get("num_byzantine", 0)
        return KrumAggregator(num_byzantine=num_byzantine)
    elif method == "multi_krum":
        num_byzantine = kwargs.get("num_byzantine", 0)
        num_selected = kwargs.get("num_selected", None)
        return MultiKrumAggregator(
            num_byzantine=num_byzantine, num_selected=num_selected
        )
    elif method == "norm_clipping":
        max_norm = kwargs.get("max_norm", 10.0)
        base = kwargs.get("base_aggregator", None)
        return NormClippingAggregator(max_norm=max_norm, base_aggregator=base)
    elif method == "adaptive":
        threshold = kwargs.get("threshold", 0.3)
        window_size = kwargs.get("window_size", 5)
        return AdaptiveAggregator(threshold=threshold, window_size=window_size)
    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. "
            f"Available: mean, weighted_mean, median, geometric_mean, "
            f"weighted_geometric_mean, trimmed_mean, krum, multi_krum, "
            f"norm_clipping, adaptive"
        )
