import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
from femnistdataloader import FEMNISTDataset
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from models import CNN
from strategicflsim.agents import DistributedClient, DistributedServer
from strategicflsim.utils.actions import create_scalar_action
from strategicflsim.utils.aggregation import get_aggregate
from strategicflsim.utils.metrics import get_gradient_metrics
from utils.config import load_config, save_config
from utils.io import generate_save_name, make_dir


def get_data(path: str):
    """Load FEMNIST data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
        user_names = list(data.keys())
    return data, user_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to config file (YAML)"
    )
    args = parser.parse_args()

    # Initialize distributed training
    if "RANK" not in os.environ:
        raise RuntimeError(
            "This script requires distributed training. "
            "Launch with: torchrun --nproc_per_node=N experiment_distributed.py --config CONFIG"
        )

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    # Assign GPU to this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(
        f"[Rank {rank}] Initialized: world_size={world_size}, "
        f"local_rank={local_rank}, device={device}"
    )

    # Load configuration
    config = load_config(args.config)

    if rank == 0:
        print("\nConfiguration:")
        print(OmegaConf.to_yaml(config, resolve=True))

    # Load data (all processes need this)
    print(f"[Rank {rank}] Loading data...")
    data_dict, user_names = get_data(config.data.train_path)
    test_data_dict, _ = get_data(config.data.test_path)
    print(f"[Rank {rank}] Data loaded: {len(user_names)} users")

    if rank == 0:
        # ==================== SERVER CODE ====================
        print("\n[Server] Initializing server...")

        # Create directories
        make_dir("./results")
        make_dir(config.experiment.save_dir)

        # Create server model
        server_model = CNN().to(device)

        if device.type == "cuda":
            server_model = torch.compile(server_model, mode="reduce-overhead")

        server = DistributedServer(
            device=device,
            model=server_model,
            criterion=nn.CrossEntropyLoss().to(device),
            optimizer=torch.optim.SGD(server_model.parameters(), lr=config.training.lr),
            aggregate_fn=get_aggregate(method=config.aggregation.method),
            agent_id="server",
        )
        print("[Server] Created distributed server")

        # Train with distributed communication
        num_clients = world_size - 1
        print(f"[Server] Starting training with {num_clients} distributed clients")

        all_losses, all_metrics = server.train_distributed(
            num_clients=num_clients,
            T=config.training.T,
            get_metrics=get_gradient_metrics,
        )

        print("\n[Server] Training complete!")

        # Save results
        losses_array = np.array(
            [[loss for loss in round_losses] for round_losses in all_losses]
        )
        grad_norms_array = np.array([metrics["grad_norms"] for metrics in all_metrics])
        cosine_sims_array = np.array(
            [metrics["cosine_similarities"] for metrics in all_metrics]
        )

        results = {
            "config": OmegaConf.to_container(config, resolve=True),
            "n_players": num_clients,
            "alpha_0": config.clients.alpha_0,
            "alpha_1": config.clients.alpha_1,
            "beta_0": config.clients.beta_0,
            "beta_1": config.clients.beta_1,
            "T": config.training.T,
            "local_steps": config.training.get("local_steps", 1),
            "train_gradsizes_per_step": grad_norms_array,
            "train_cosine_per_step": cosine_sims_array,
            "train_losses_per_step": losses_array,
            "distributed": True,
            "world_size": world_size,
        }

        save_name = generate_save_name(config)

        with open(f"{save_name}_distributed.pkl", "wb") as f:
            pickle.dump(results, f)

        save_config(config, f"{save_name}_distributed")

        print(f"\n[Server] Results saved to: {save_name}_distributed.pkl")

    else:
        # ==================== CLIENT CODE ====================
        client_id = rank - 1  # Client IDs: 0, 1, 2, ...
        num_clients = world_size - 1

        print(f"\n[Client {client_id}] Initializing client...")

        # Partition data for this client
        split_index = len(user_names) // num_clients
        start_idx = client_id * split_index
        end_idx = (
            (client_id + 1) * split_index
            if client_id < num_clients - 1
            else len(user_names)
        )
        client_user_names = user_names[start_idx:end_idx]

        print(
            f"[Client {client_id}] Assigned {len(client_user_names)} users "
            f"(indices {start_idx}:{end_idx})"
        )

        # Determine if adversarial (last client)
        if client_id == num_clients - 1:
            alpha = config.clients.alpha_1
            beta = config.clients.beta_1
            agent_id = "bad"
        else:
            alpha = config.clients.alpha_0
            beta = config.clients.beta_0
            agent_id = f"good{client_id}"

        # Create datasets
        train_dataset = FEMNISTDataset(client_user_names, data_dict)
        test_dataset = FEMNISTDataset(client_user_names, test_data_dict)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.get("batch_size", 32),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.training.get("eval_batch_size", 128),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Create client model
        client_model = CNN().to(device)

        if device.type == "cuda":
            client_model = torch.compile(client_model, mode="reduce-overhead")

        # Create distributed client
        client = DistributedClient(
            device=device,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=client_model,
            criterion=nn.CrossEntropyLoss().to(device),
            optimizer=torch.optim.SGD(client_model.parameters(), lr=config.training.lr),
            action=create_scalar_action(alpha, beta),
            agent_id=agent_id,
            local_steps=config.training.get("local_steps", 1),
        )

        print(f"[Client {client_id}] Created {agent_id} (alpha={alpha}, beta={beta})")

        # Train with distributed communication
        client.train_distributed(T=config.training.T)

        print(f"\n[Client {client_id}] Training complete!")

    # Cleanup
    dist.destroy_process_group()
    print(f"[Rank {rank}] Shutdown complete")


if __name__ == "__main__":
    main()
