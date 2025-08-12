import torch
from shakespeareutils import letter_to_vec, word_to_indices
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    """PyTorch Dataset for Shakespeare data."""

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
                # Convert words to indices and letters to vectors
                client_inputs = torch.tensor(
                    [word_to_indices(word) for word in data_dict[client_name]["x"]]
                )
                client_labels = torch.tensor(
                    [letter_to_vec(letter) for letter in data_dict[client_name]["y"]]
                )

                # Convert one-hot labels to class indices if needed
                if client_labels.dim() > 1:
                    client_labels = torch.argmax(client_labels, dim=1)

                self.inputs.append(client_inputs)
                self.labels.append(client_labels)

        if self.inputs:
            self.inputs = torch.cat(self.inputs, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
        else:
            # Create empty tensors with appropriate shapes
            self.inputs = torch.empty(
                0, dtype=torch.long
            )  # Assuming word_to_indices returns integers
            self.labels = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
