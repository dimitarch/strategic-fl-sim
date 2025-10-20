from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm

from .client import Client
from .server import Server


class DistributedServer(Server):
    """
    Distributed version of Server for multi-GPU/multi-node federated learning.

    Extends Server with torch.distributed communication primitives.
    Must be instantiated on rank 0 process only.

    Usage:
        torchrun --nproc_per_node=4 experiment.py

        if rank == 0:
            server = DistributedServer(...)
            server.train_distributed(num_clients=3, T=1000)
    """

    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        aggregate_fn: Callable,
        agent_id: str = "server",
    ):
        """Initialize distributed server (same args as Server)."""
        super().__init__(device, model, criterion, optimizer, aggregate_fn, agent_id)

        # Verify we're rank 0
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                raise RuntimeError(f"DistributedServer must be rank 0, got rank {rank}")

    def broadcast_model(self) -> None:
        """Broadcast trainable parameters to all client processes."""
        for param in self.trainable_params:
            dist.broadcast(param.data, src=0)

    def gather_gradients(self, num_clients: int) -> List[List[torch.Tensor]]:
        """
        Gather gradients from all client processes.

        Args:
            num_clients: Number of client processes (world_size - 1)

        Returns:
            List of gradient lists, one per client
        """
        client_gradients = [[] for _ in range(num_clients)]

        # Gather each layer's gradient separately
        for param in self.trainable_params:
            # Prepare buffers: [server_dummy, client1, client2, ...]
            gathered_tensors = [
                torch.zeros_like(param.data) for _ in range(num_clients + 1)
            ]

            # Server sends dummy, clients send actual gradients
            dist.gather(
                torch.zeros_like(param.data), gather_list=gathered_tensors, dst=0
            )

            # Distribute to client gradient lists (skip rank 0)
            for client_idx in range(num_clients):
                client_gradients[client_idx].append(
                    gathered_tensors[client_idx + 1].clone()
                )

        return client_gradients

    def gather_losses(self, num_clients: int) -> List[float]:
        """Gather loss values from all client processes."""
        loss_buffer = [
            torch.zeros(1, device=self.device) for _ in range(num_clients + 1)
        ]

        dist.gather(torch.zeros(1, device=self.device), gather_list=loss_buffer, dst=0)

        return [loss_buffer[i + 1].item() for i in range(num_clients)]

    def train_distributed(
        self,
        num_clients: int,
        T: int = 1000,
        get_metrics: Optional[Any] = None,
    ) -> Tuple[List[List[float]], Optional[List[dict]]]:
        """
        Execute distributed federated learning protocol.

        Replaces train(clients, T) for distributed execution.

        Args:
            num_clients: Number of client processes (world_size - 1)
            T: Number of training rounds
            get_metrics: Optional function to compute gradient metrics

        Returns:
            (losses_global, metrics_global) - same format as Server.train()
        """
        losses_global = []
        metrics_global = []

        print(
            f"[Server] Starting distributed training: {num_clients} clients, {T} rounds"
        )

        for round_idx in tqdm(range(T), desc="Federated Training"):
            # 1. Broadcast model to all clients
            self.broadcast_model()

            # 2. Clients train locally (happens in parallel across processes)

            # 3. Gather gradients from all clients
            client_gradients = self.gather_gradients(num_clients)

            # 4. Gather losses from all clients
            round_losses = self.gather_losses(num_clients)

            # 5. Aggregate and update (same as single-process)
            self.update(client_gradients)

            # 6. Compute metrics if requested
            if get_metrics is not None:
                aggregated_gradient = self.aggregate(client_gradients)
                metrics_global.append(
                    get_metrics(client_gradients, aggregated_gradient)
                )

            losses_global.append(round_losses)

        torch.cuda.empty_cache()
        return losses_global, metrics_global


class DistributedClient(Client):
    """
    Distributed version of Client for multi-GPU/multi-node federated learning.

    Extends Client with torch.distributed communication primitives.
    Must be instantiated on rank > 0 processes only.

    Usage:
        torchrun --nproc_per_node=4 experiment.py

        if rank > 0:
            client = DistributedClient(...)
            client.train_distributed(T=1000)
    """

    def __init__(
        self,
        device: torch.device,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        action: Callable,
        local_steps: int = 1,
        agent_id: str = "client",
    ):
        """Initialize distributed client (same args as Client)."""
        super().__init__(
            device,
            train_dataloader,
            test_dataloader,
            model,
            criterion,
            optimizer,
            action,
            local_steps,
            agent_id,
        )

        # Verify we're not rank 0
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                raise RuntimeError(
                    f"DistributedClient must not be rank 0, got rank {rank}"
                )

    def receive_global_model_distributed(self) -> None:
        """Receive model from server via broadcast."""
        for param in [p for p in self.model.parameters() if p.requires_grad]:
            dist.broadcast(param.data, src=0)

    def send_gradient(self, gradient: List[torch.Tensor]) -> None:
        """Send gradient to server via gather."""
        for grad_tensor in gradient:
            dist.gather(grad_tensor, dst=0)

    def send_loss(self, loss: float) -> None:
        """Send loss value to server."""
        loss_tensor = torch.tensor([loss], device=self.device)
        dist.gather(loss_tensor, dst=0)

    def train_distributed(self, T: int = 1000) -> None:
        """
        Execute distributed training for this client.

        Replaces participation in server.train(clients, T).

        Args:
            T: Number of training rounds (must match server)
        """
        rank = dist.get_rank()

        for round_idx in range(T):
            # 1. Receive model from server
            self.receive_global_model_distributed()

            # 2. Train locally (same as single-process)
            gradient, loss = self.local_train()

            # 3. Send gradient to server
            self.send_gradient(gradient)

            # 4. Send loss to server
            self.send_loss(loss)

            # Periodic logging
            if (round_idx + 1) % 10 == 0:
                print(
                    f"[Client rank={rank}] Round {round_idx + 1}/{T}, Loss: {loss:.4f}"
                )
