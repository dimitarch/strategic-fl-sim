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


def create_byzantine_action(strength: float = 10.0):
    def action(gradient):
        return -strength * gradient

    return action


def create_freeriding_action():
    def action(gradient):
        return torch.zeros_like(gradient)

    return action
