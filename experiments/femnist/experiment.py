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

# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True


def get_data(path: str):
    """Load data from JSON file."""
    data = {}
    user_names = []

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

    torch.backends.cudnn.benchmark = True  # cuDNN optimization

    print("Using configuration:")
    print(OmegaConf.to_yaml(config, resolve=True))

    # Load data
    print("Loading training data...")
    data_dict, user_names = get_data(config.data.train_path)
    print("Finished loading training data!")

    print("Loading test data...")
    test_data_dict, _ = get_data(config.data.test_path)
    print("Finished loading test data!")

    # set_seed(42)
    device = get_device()

    # Create the server agent
    print("Creating server agent...")
    server_model = CNN().to(device)

    if device.type == "cuda":
        print("Compiling server model...")

        server_model = torch.compile(
            server_model, mode="reduce-overhead"
        )  # maybe not on apple silicon :(

    server = Server(
        device=device,
        model=server_model,
        criterion=nn.CrossEntropyLoss().to(device),
        optimizer=torch.optim.SGD(
            server_model.parameters(), lr=config.training.lr, foreach=True
        ),
        aggregate_fn=get_aggregate(method=config.aggregation.method),
    )
    print(f"Created {server}")

    # Create list of clients, last is "misreporting" and others are truthful
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
        else:
            alpha = config.clients.alpha_0
            beta = config.clients.beta_0

        train_dataset = FEMNISTDataset(client_user_names, data_dict)
        test_dataset = FEMNISTDataset(client_user_names, test_data_dict)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.get("batch_size", 32),
            shuffle=False,
            # num_workers=2,
            pin_memory=True if device.type == "cuda" else False,
            # persistent_workers=True,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.training.get("eval_batch_size", 128),
            shuffle=False,
            # num_workers=2,
            pin_memory=True if device.type == "cuda" else False,
            # persistent_workers=True,
        )

        # Create client
        client_model = CNN().to(device)  # Create local model

        if device.type == "cuda":
            client_model = torch.compile(client_model, mode="reduce-overhead")

        client = Client(
            device=device,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=client_model,
            criterion=nn.CrossEntropyLoss().to(device),
            optimizer=torch.optim.SGD(
                client_model.parameters(), lr=config.training.lr, foreach=True
            ),
            action=create_scalar_action(alpha, beta),
            agent_id="bad" if i == config.clients.n_players - 1 else f"good{i}",
        )

        clients.append(client)
        print(f"Created {client} with id {client.agent_id} on {client.device}")

    # Train
    print("Starting training...")
    all_losses, all_metrics = server.train(
        clients=clients,
        T=config.training.T,
        get_metrics=get_gradient_metrics,
    )
    print("Training finished!")

    # Prepare test data and evaluate
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
