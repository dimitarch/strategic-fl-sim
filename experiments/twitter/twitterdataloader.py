import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TwitterDataset(Dataset):
    """PyTorch Dataset for Twitter data with pre-tokenization."""

    def __init__(self, client_names, data_dict, max_length=512):
        if isinstance(client_names, str):
            self.client_names = [client_names]
        else:
            self.client_names = client_names

        self.max_length = max_length

        # Collect all texts and labels
        texts = []
        labels_list = []

        for client_name in self.client_names:
            if client_name in data_dict:
                client_texts = [
                    input_item[4] for input_item in data_dict[client_name]["x"]
                ]
                client_labels = torch.tensor(data_dict[client_name]["y"]).long()

                texts.extend(client_texts)
                labels_list.append(client_labels)

        self.labels = (
            torch.cat(labels_list) if labels_list else torch.empty(0, dtype=torch.long)
        )

        # Pre-tokenize everything once so there is no overhead for tokenizing while training
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokenized[idx], self.labels[idx]


class TwitterDataLoader:
    """Legacy DataLoader for Twitter dataset - kept for backward compatibility."""

    def __init__(self, client_names, data_dict, test_data_dict):
        # Handle both single client and list of clients
        if isinstance(client_names, str):
            self.client_names = [client_names]
        else:
            self.client_names = client_names

        self.data_dict = {
            client_name: data_dict[client_name]
            for client_name in self.client_names
            if client_name in data_dict
        }
        self.test_data_dict = {
            client_name: test_data_dict[client_name]
            for client_name in self.client_names
            if client_name in test_data_dict
        }

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def get_batch(self):
        """Get a batch of data by sampling uniformly from the client pool."""
        # Sample a client uniformly at random
        client_name = np.random.choice(self.client_names)

        # Get data for this client
        client_data = self.data_dict[client_name]

        # Prepare inputs and labels for Twitter data
        # Extract text from the 5th element of each input (index 4)
        texts = [input_item[4] for input_item in client_data["x"]]

        # Tokenize the texts
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )["input_ids"]

        labels = torch.tensor(client_data["y"])

        return inputs, labels

    def get_test_data(self):
        """Get test data for evaluation."""
        pass
