import argparse
import json
import pickle

import numpy as np
import torch
from femnistdataloader import FEMNISTDataset
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from models import CNN
from strategicflsim.agents import Client, Server
from strategicflsim.utils.actions import create_scalar_action
from strategicflsim.utils.aggregation import get_aggregate
from strategicflsim.utils.evaluate import evaluate_with_ids
from strategicflsim.utils.metrics import get_gradient_metrics
from utils.config import load_config, save_config
from utils.device import get_device
from utils.io import generate_save_name, make_dir


def get_data(path: str):
    """Load data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
        user_names = list(data.keys())
    return data, user_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to config file (YAML)"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config is None:
        print("Using default config! May break functionality!")
    config = load_config(args.config)

    # Make sure that the destination folders for results exist
    make_dir("./results")
    make_dir(config.experiment.save_dir)

    torch.backends.cudnn.benchmark = True

    print("Using configuration:")
    print(OmegaConf.to_yaml(config, resolve=True))

    # Load data
    print("Loading training data...")
    data_dict, user_names = get_data(config.data.train_path)
    print("Finished loading training data!")

    print("Loading test data...")
    test_data_dict, _ = get_data(config.data.test_path)
    print("Finished loading test data!")

    device = get_device()

    # Create the server agent
    print("Creating server agent...")
    server_model = CNN().to(device)
    print("Compiling server model...")

    if device.type == "cuda":
        server_model = torch.compile(server_model, mode="reduce-overhead")

    server = Server(
        device=device,
        model=server_model,
        criterion=nn.CrossEntropyLoss().to(device),
        optimizer=torch.optim.SGD(server_model.parameters(), lr=config.training.lr),
        aggregate_fn=get_aggregate(method=config.aggregation.method),
    )
    print(f"Created {server}")

    # Prepare data splits for all clients
    print("Preparing client data splits...")
    split_index = len(user_names) // config.clients.n_players

    data_splits = []
    agent_ids = []

    for i in range(config.clients.n_players):
        # Get the range of users for this client group
        start_idx = max(0, i * split_index)
        end_idx = min((i + 1) * split_index, len(user_names))
        if i == config.clients.n_players - 1:
            end_idx = len(user_names)  # Last client gets remaining users

        client_user_names = user_names[start_idx:end_idx]

        # Create datasets
        train_dataset = FEMNISTDataset(client_user_names, data_dict)
        test_dataset = FEMNISTDataset(client_user_names, test_data_dict)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.get("batch_size", 32),
            shuffle=True,
            pin_memory=True if device.type == "cuda" else False,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.training.get("eval_batch_size", 128),
            shuffle=False,
            pin_memory=True if device.type == "cuda" else False,
        )

        data_splits.append((train_dataloader, test_dataloader))

        # Determine agent ID
        if i == config.clients.n_players - 1:
            agent_ids.append("bad")
        else:
            agent_ids.append(f"good{i}")

    # Determine action based on last client (adversarial)
    def get_action(idx):
        if idx == config.clients.n_players - 1:
            return create_scalar_action(config.clients.alpha_1, config.clients.beta_1)
        else:
            return create_scalar_action(config.clients.alpha_0, config.clients.beta_0)

    # Create all clients using factory method
    print("Creating client array...")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    devices = [
        torch.device(f"cuda:{(i + 1) % num_gpus}")
        if torch.cuda.is_available()
        else device
        for i in range(config.clients.n_players)
    ]

    clients = Client.create_clients(
        n_clients=config.clients.n_players,
        devices=devices,
        data_splits=data_splits,
        model_fn=lambda: CNN(),
        criterion_fn=lambda: nn.CrossEntropyLoss(),
        optimizer_fn=lambda params: torch.optim.SGD(params, lr=config.training.lr),
        action_fn=lambda idx: get_action(idx),
        local_steps=config.training.get("local_steps", 1),
        agent_ids=agent_ids,
    )

    # Optionally compile client models
    if device.type == "cuda":
        for client in clients:
            client.model = torch.compile(client.model, mode="reduce-overhead")

    # Train
    print("Starting training...")
    all_losses, all_metrics = server.train(
        clients=clients,
        T=config.training.T,
        get_metrics=get_gradient_metrics,
    )
    print("Training finished!")

    # Evaluate
    print("Starting evaluation...")
    final_accuracy, final_loss = evaluate_with_ids(server, clients)
    print("Evaluation finished!")

    # Convert metrics to numpy arrays
    losses_array = np.array(
        [[loss for loss in round_losses] for round_losses in all_losses]
    )
    grad_norms_array = np.array([metrics["grad_norms"] for metrics in all_metrics])
    cosine_sims_array = np.array(
        [metrics["cosine_similarities"] for metrics in all_metrics]
    )

    results = {
        "config": OmegaConf.to_container(config, resolve=True),
        "n_players": config.clients.n_players,
        "alpha_0": config.clients.alpha_0,
        "alpha_1": config.clients.alpha_1,
        "beta_0": config.clients.beta_0,
        "beta_1": config.clients.beta_1,
        "T": config.training.T,
        "local_steps": config.training.get("local_steps", 1),
        "train_gradsizes_per_step": grad_norms_array,
        "train_cosine_per_step": cosine_sims_array,
        "train_losses_per_step": losses_array,
        "test_accuracy": final_accuracy,
        "test_losses": final_loss,
    }

    # Generate save name
    save_name = generate_save_name(config)

    # Save results
    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save configuration
    save_config(config, save_name)

    print("Federated learning completed!")
    print(f"Final test accuracies: {results['test_accuracy']}")
    print(f"Final test losses: {results['test_losses']}")
    print(f"Results saved to: {save_name}.pkl")
