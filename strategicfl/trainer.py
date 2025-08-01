from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .agents import Client, Server


def compute_client_gradient(
    server: Server, client: Client, K: int
) -> Tuple[float, List[torch.Tensor], List[torch.Tensor]]:
    inputs, labels = next(client.train_iterator)

    if K > 1:
        client.model.load_state_dict(server.state_dict())

        for _ in range(K):
            loss = client.update(inputs, labels)

        grad = []
        for local_param, server_param in zip(
            client.model.parameters(), server.parameters()
        ):
            grad.append((server_param - local_param).detach().clone())
    else:  # Single step case
        outputs = server.model(inputs.to(server.device))
        loss = server.criterion(outputs, labels.to(server.device))
        loss.backward()
        grad = [p.grad.detach().clone() for p in server.model.parameters()]
        server.model.zero_grad()

    # Apply client action and clean up
    sent_grad = [client.apply_action(g) for g in grad]
    loss_value = loss.detach().cpu().item()

    return loss_value, grad, sent_grad


def train(
    server: Server,
    clients: List[Client],
    T: int = 10000,
    K: int = 1,
    get_metrics: Optional[Any] = None,
) -> Tuple[nn.Module, np.ndarray, Optional[List[dict]]]:
    losses_global = []
    metrics_global = []

    for _ in tqdm(range(T), total=T, desc="Training"):
        grads_sent = []
        grads_real = []
        losses = []

        # Compute gradients for all agents
        for client in clients:
            loss, grad, sent = compute_client_gradient(
                server, client, K
            )

            grads_sent.append(sent)
            grads_real.append(grad)
            losses.append(loss)

        losses_global.append(losses)
        aggregate_gradient = server.aggregate_fn(grads_sent)
        server.update_model_with_gradient(aggregate_gradient)

        # If there are any metrics to be recorded
        if get_metrics is not None:
            metrics_global.append(get_metrics(grads_sent, aggregate_gradient))

    torch.cuda.empty_cache()

    return losses_global, metrics_global
