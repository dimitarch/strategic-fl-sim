from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class BaseAction(ABC):
    """
    Abstract base class for gradient manipulation actions.

    All action classes should inherit from this and implement __call__.
    Actions operate on individual gradient tensors (per-layer or flattened).

    Example:
        class MyAction(BaseAction):
            def __init__(self, strength):
                self.strength = strength

            def __call__(self, gradient):
                return self.strength * gradient
    """

    @abstractmethod
    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Apply action to gradient.

        Args:
            gradient: Input gradient tensor

        Returns:
            Modified gradient tensor
        """
        pass

    def __repr__(self):
        """String representation showing action parameters."""
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


class HonestAction(BaseAction):
    """
    Identity action - returns gradient unchanged (honest behavior).

    Example:
        action = HonestAction()
        # gradient remains unchanged
    """

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Return gradient unchanged."""
        return gradient


class ScalarAction(BaseAction):
    """
    Scale gradient and add Gaussian noise.

    Applies: modified_grad = alpha * gradient + noise

    where noise ~ N(0, beta / sqrt(d)), d = gradient dimension

    Args:
        alpha: Scaling factor (default: 1.0 for honest)
        beta: Noise level (default: 0.0 for no noise)

    Example:
        # Honest client
        action = ScalarAction(alpha=1.0, beta=0.0)

        # Adversarial: amplify gradient 2x with noise
        action = ScalarAction(alpha=2.0, beta=0.1)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply scaling and noise to gradient."""
        noise = self.beta * torch.randn_like(gradient) / np.sqrt(gradient.numel())
        return self.alpha * gradient + noise


class SignFlipAction(BaseAction):
    """
    Flip the sign of the gradient (optionally with scaling).

    Applies: modified_grad = -strength * gradient

    Args:
        strength: Scaling factor (default: 1.0)

    Example:
        # Standard sign flip attack
        action = SignFlipAction(strength=1.0)

        # Amplified sign flip
        action = SignFlipAction(strength=2.0)
    """

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Flip sign of gradient."""
        return -self.strength * gradient


class GradientAscentAction(BaseAction):
    """
    Gradient ascent attack - maximize loss instead of minimize.

    Useful for studying poisoning impact on model convergence.

    Args:
        strength: How aggressively to maximize loss (default: 1.0)

    Example:
        # Full gradient ascent
        action = GradientAscentAction(strength=1.0)

        # Partial ascent (stealthier)
        action = GradientAscentAction(strength=0.5)

    Note:
        This is equivalent to SignFlipAction but more semantically clear
        for studying optimization dynamics.
    """

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply gradient ascent (negative of gradient descent)."""
        return -self.strength * gradient


class FreeRidingAction(BaseAction):
    """
    Send zero gradient (free-riding behavior).

    Client doesn't contribute to training but benefits from others.

    Example:
        action = FreeRidingAction()
        # Always returns zero gradient
    """

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Return zero gradient."""
        return torch.zeros_like(gradient)


class GaussianNoiseAction(BaseAction):
    """
    Replace gradient with Gaussian noise.

    Each coordinate is sampled from N(mu, sigma^2).

    Args:
        mu: Mean of Gaussian distribution (default: 0.0)
        sigma: Standard deviation (default: 1.0)

    Example:
        # White noise attack
        action = GaussianNoiseAction(mu=0.0, sigma=1.0)

        # High-variance noise
        action = GaussianNoiseAction(mu=0.0, sigma=10.0)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.mu = mu
        self.sigma = sigma

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Replace gradient with Gaussian noise."""
        return torch.normal(
            mean=self.mu,
            std=self.sigma,
            size=gradient.shape,
            device=gradient.device,
            dtype=gradient.dtype,
        )


class InfinityAction(BaseAction):
    """
    Send gradient with all values set to positive infinity.

    Extreme attack to disrupt aggregation.

    Example:
        action = InfinityAction()
    """

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Return gradient filled with infinity."""
        return torch.full_like(
            gradient, float("inf"), dtype=torch.float32, device=gradient.device
        )


class NormBoundedAction(BaseAction):
    """
    Constrain attack to stay within norm bound (evade norm-based defenses).

    Many defenses reject gradients with large norms. This attack stays
    under the radar while still being adversarial.

    Args:
        max_norm: Maximum L2 norm allowed
        base_action: Underlying attack to apply before norm clipping

    Example:
        # Sign flip but stay under norm bound
        action = NormBoundedAction(
            max_norm=10.0,
            base_action=SignFlipAction()
        )

        # Scaled attack with norm constraint
        action = NormBoundedAction(
            max_norm=15.0,
            base_action=ScalarAction(alpha=3.0, beta=0.0)
        )
    """

    def __init__(self, max_norm: float, base_action: BaseAction):
        self.max_norm = max_norm
        self.base_action = base_action

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply attack then clip to norm bound."""
        # Apply base attack
        modified = self.base_action(gradient)

        # Clip to max norm
        grad_norm = torch.norm(modified)
        if grad_norm > self.max_norm:
            modified = modified * (self.max_norm / grad_norm)

        return modified


class RelativeNormAction(BaseAction):
    """
    Scale gradient relative to honest gradient norm.

    Makes attack adaptive to gradient magnitude (stealthier).

    Args:
        relative_scale: Target norm as ratio of honest norm
                       e.g., 1.5 means 50% larger
        direction: "same", "opposite", or "random"

    Example:
        # Stay close to honest gradient but flip direction
        action = RelativeNormAction(relative_scale=1.2, direction="opposite")

        # Large random direction with controlled magnitude
        action = RelativeNormAction(relative_scale=2.0, direction="random")
    """

    def __init__(self, relative_scale: float = 1.5, direction: str = "opposite"):
        self.relative_scale = relative_scale
        self.direction = direction

        if direction not in ["same", "opposite", "random"]:
            raise ValueError("direction must be 'same', 'opposite', or 'random'")

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply relative norm attack."""
        honest_norm = torch.norm(gradient)
        target_norm = self.relative_scale * honest_norm

        if self.direction == "same":
            direction = gradient / honest_norm
        elif self.direction == "opposite":
            direction = -gradient / honest_norm
        else:  # random
            direction = torch.randn_like(gradient)
            direction = direction / torch.norm(direction)

        return target_norm * direction


class ConvergenceAwareAction(BaseAction):
    """
    Adapt attack based on gradient convergence.

    Attack more aggressively when gradients are small (near convergence)
    and less when gradients are large (early training).

    Args:
        threshold: Gradient norm threshold for switching behavior
        early_action: Action when norm > threshold (early training)
        late_action: Action when norm < threshold (near convergence)

    Example:
        # Honest early, attack late (backdoor persistence)
        action = ConvergenceAwareAction(
            threshold=1.0,
            early_action=HonestAction(),
            late_action=SignFlipAction()
        )

        # Adaptive scaling
        action = ConvergenceAwareAction(
            threshold=5.0,
            early_action=ScalarAction(alpha=1.5, beta=0.0),
            late_action=ScalarAction(alpha=3.0, beta=0.1)
        )
    """

    def __init__(
        self, threshold: float, early_action: BaseAction, late_action: BaseAction
    ):
        self.threshold = threshold
        self.early_action = early_action
        self.late_action = late_action

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply action based on gradient magnitude."""
        grad_norm = torch.norm(gradient).item()

        if grad_norm > self.threshold:
            return self.early_action(gradient)
        else:
            return self.late_action(gradient)


class ComposedAction(BaseAction):
    """
    Compose multiple actions sequentially.

    Applies actions in order: action_n(...(action_2(action_1(gradient))))

    Args:
        actions: List of action instances to apply in sequence

    Example:
        # First scale, then add noise, then flip sign
        action = ComposedAction([
            ScalarAction(alpha=2.0, beta=0.0),
            GaussianNoiseAction(mu=0.0, sigma=0.1),
            SignFlipAction()
        ])
    """

    def __init__(self, actions: List[BaseAction]):
        if not actions:
            raise ValueError("actions list cannot be empty")
        self.actions = actions

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply actions sequentially."""
        result = gradient
        for action in self.actions:
            result = action(result)
        return result

    def __repr__(self):
        actions_str = ", ".join(repr(a) for a in self.actions)
        return f"ComposedAction([{actions_str}])"


class RandomizedAction(BaseAction):
    """
    Randomly choose between multiple actions each time.

    Args:
        actions: List of (action, probability) tuples
                probabilities should sum to 1.0

    Example:
        # 70% honest, 30% adversarial
        action = RandomizedAction([
            (HonestAction(), 0.7),
            (SignFlipAction(), 0.3)
        ])

        # Uniform random between 3 attacks
        action = RandomizedAction([
            (SignFlipAction(), 1/3),
            (FreeRidingAction(), 1/3),
            (GaussianNoiseAction(), 1/3)
        ])
    """

    def __init__(self, actions: List[tuple[BaseAction, float]]):
        if not actions:
            raise ValueError("actions list cannot be empty")

        self.actions = [a for a, _ in actions]
        self.probs = [p for _, p in actions]

        # Validate probabilities
        if not np.isclose(sum(self.probs), 1.0):
            raise ValueError(f"Probabilities must sum to 1.0, got {sum(self.probs)}")

    def __call__(self, gradient: torch.Tensor) -> torch.Tensor:
        """Randomly select and apply one action."""
        action = np.random.choice(self.actions, p=self.probs)
        return action(gradient)

    def __repr__(self):
        items = ", ".join(
            f"({repr(a)}, {p:.2f})" for a, p in zip(self.actions, self.probs)
        )
        return f"RandomizedAction([{items}])"


# Backward compatibility: keep factory functions
def create_scalar_action(alpha: float = 1.0, beta: float = 0.0):
    """
    Factory function for ScalarAction (backward compatibility).

    Prefer using ScalarAction() directly in new code.
    """
    return ScalarAction(alpha=alpha, beta=beta)


def create_sign_flip_action(strength: float = 1.0):
    """
    Factory function for SignFlipAction (backward compatibility).

    Prefer using SignFlipAction() directly in new code.
    """
    return SignFlipAction(strength=strength)


def create_freeriding_action():
    """
    Factory function for FreeRidingAction (backward compatibility).

    Prefer using FreeRidingAction() directly in new code.
    """
    return FreeRidingAction()
