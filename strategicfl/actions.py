import numpy as np
import torch


def create_scalar_action(alpha, beta):
    """Create a gradient modification action."""

    def action(gradient):
        noise = (
            beta
            * torch.normal(torch.zeros_like(gradient))
            / np.sqrt(len(gradient.flatten()))
        )
        return alpha * gradient + noise

    return action
