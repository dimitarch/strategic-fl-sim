import argparse
import json
import pickle

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from twitterdataloader import TwitterDataset

from strategicfl.actions import create_scalar_action
from strategicfl.agents import Client, Server
from strategicfl.aggregation import get_aggregate
from strategicfl.models import BertWithClassifier
from strategicfl.trainer import train
from utils.config import load_config, save_config
from utils.device import get_device
from utils.evaluate import evaluate_with_ids
from utils.io import generate_save_name, make_dir
from utils.metrics import get_gradient_metrics


def get_data(path: str):
    """Load data from JSON file."""

    with open(path, "r") as f:
        data = json.load(f)
        user_names = data["users"]
        data_dict = data["user_data"]

    return data_dict, user_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to config file (YAML)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.training.T = 2

    # Make sure that the destination folders for results exist
    make_dir("./results")
    make_dir(config.experiment.save_dir)

    print("Using configuration:")
    print(OmegaConf.to_yaml(config, resolve=True))

    # Load data
    # Load data
    print("Loading training data...")
    data_dict, user_names = get_data(config.data.train_path)
    print("Finished loading training data!")

    print("Loading test data...")
    test_data_dict, user_names_test = get_data(config.data.test_path)
    print("Finished loading test data!")

    # Verify consistency between train and test user names
    usernames_diff = list(set(user_names) ^ set(user_names_test))
    assert not usernames_diff, "Inconsistent usernames between test and train"

    # Filter users based on sample count criteria
    if hasattr(config.data, "min_samples") and hasattr(config.data, "max_samples"):
        subset = [
            user_name
            for user_name in user_names
            if config.data.min_samples
            <= len(data_dict[user_name]["y"])
            <= config.data.max_samples
        ]
        print(
            f"Filtered to {len(subset)} users with {config.data.min_samples}-{config.data.max_samples} samples"
        )

        user_names = subset
        data_dict = {key: data_dict[key] for key in subset}
        test_data_dict = {key: test_data_dict[key] for key in subset}

    device = get_device()

    # Create a server agent with BERT model
    print("Creating server agent...")
    server_model = BertWithClassifier()

    server = Server(
        device=device,
        model=server_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(server_model.parameters(), lr=config.training.lr),
        aggregate_fn=get_aggregate(method=config.aggregation.method),
        agent_id="server",  # Add server ID
    )
    print(f"Created {server}")

    # Create array of clients, last client is "misreporting" and all others are truthful
    print("Creating client array...")
    split_index = len(user_names) // config.clients.n_players

    clients = []
    for i in range(config.clients.n_players):
        # Get the range of users for this client group
        start_idx = max(0, i * split_index)
        end_idx = min((i + 1) * split_index, len(user_names))
        client_user_names = user_names[start_idx:end_idx]

        # Determine client type (good vs bad) - last group is adversarial
        if i == config.clients.n_players - 1:
            alpha = config.clients.alpha_1
            beta = config.clients.beta_1
            client_id = "bad"
        else:
            alpha = config.clients.alpha_0
            beta = config.clients.beta_0
            client_id = f"good{i}"

        # Create datasets using the new TwitterDataset
        train_dataset = TwitterDataset(
            client_user_names,
            data_dict,
            max_length=config.model.get("max_length", 512),  # Configurable max length
        )
        test_dataset = TwitterDataset(
            client_user_names,
            test_data_dict,
            max_length=config.model.get("max_length", 512),
        )

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.get("batch_size", 16),  # Smaller batch for BERT
            shuffle=True,
            pin_memory=True if device.type == "cuda" else False,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.training.get(
                "eval_batch_size", 32
            ),  # Smaller eval batch for BERT
            shuffle=False,
            pin_memory=True if device.type == "cuda" else False,
        )

        # Create client
        client_model = BertWithClassifier().to(device)

        client = Client(
            device=device,
            train_dataloader=train_dataloader,  # Use standard DataLoader
            test_dataloader=test_dataloader,  # Use standard DataLoader
            model=client_model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(client_model.parameters(), lr=config.training.lr),
            action=create_scalar_action(alpha, beta),
            agent_id=client_id,  # Add client ID
        )

        clients.append(client)
        print(f"Created {client}")

    # Train
    print("Starting training...")
    all_losses, all_metrics = train(
        server=server,
        clients=clients,
        T=config.training.T,
        K=config.training.local_steps,
        get_metrics=get_gradient_metrics,
    )
    print("Training finished!")

    # Evaluate
    print("Starting evaluation...")
    final_accuracy, final_loss = evaluate_with_ids(server, clients)
    print("Evaluation finished!")

    # Convert metrics to numpy arrays for compatibility
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
        "local_steps": config.training.local_steps,
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
