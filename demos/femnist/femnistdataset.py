import torch
from torch.utils.data import Dataset


class FEMNISTDataset(Dataset):
    """PyTorch Dataset for FEMNIST data."""

    def __init__(self, client_names, data_dict):
        if isinstance(client_names, str):
            self.client_names = [client_names]
        else:
            self.client_names = client_names

        # Collect all data from specified clients
        self.inputs = []
        self.labels = []

        for client_name in self.client_names:
            if client_name in data_dict:
                client_inputs = torch.tensor(data_dict[client_name]["x"]).float()
                client_inputs = client_inputs.reshape(-1, 1, 28, 28)
                client_labels = torch.tensor(data_dict[client_name]["y"]).long()

                self.inputs.append(client_inputs)
                self.labels.append(client_labels)

        if self.inputs:
            self.inputs = torch.cat(self.inputs, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
        else:
            self.inputs = torch.empty(0, 1, 28, 28)
            self.labels = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
