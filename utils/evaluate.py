from typing import List

from strategicfl.agents import Client, Server


def evaluate_with_ids(server: Server, clients: List[Client]) -> tuple[dict, dict]:
    """
    Evaluate all clients using the server's global model weights.

    Args:
        server (Server): FL server
        clients (List[Client]): List of client

    Returns:
        tuple[dict, dict]: A tuple containing:
            - accuracy_all (dict): Dictionary mapping client IDs to their accuracy scores
            - loss_all (dict): Dictionary mapping client IDs to their loss values
    """
    print("Starting evaluation...")
    accuracy_all = {}
    loss_all = {}

    # Save original client model states
    original_states = {}
    for client in clients:
        original_states[client.agent_id] = {
            k: v.clone() for k, v in client.model.state_dict().items()
        }

    # Update all client models with server weights
    for client in clients:
        client.model.load_state_dict(server.model.state_dict())

    # Evaluate each client, using ID as key
    for client in clients:
        accuracy, loss = client.evaluate_on_test_set()
        accuracy_all[client.agent_id] = accuracy
        loss_all[client.agent_id] = loss
        print(f"{client.agent_id}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")

    # Restore original client model states
    for client in clients:
        client.model.load_state_dict(original_states[client.agent_id])

    return accuracy_all, loss_all
