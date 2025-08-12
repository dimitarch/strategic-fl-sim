import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TwitterDataset(Dataset):
    """PyTorch Dataset for Twitter data."""

    def __init__(self, client_names, data_dict, max_length=512):
        if isinstance(client_names, str):
            self.client_names = [client_names]
        else:
            self.client_names = client_names

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Collect all data from specified clients
        self.texts = []
        self.labels = []

        for client_name in self.client_names:
            if client_name in data_dict:
                # Extract text from the 5th element of each input (index 4)
                client_texts = [
                    input_item[4] for input_item in data_dict[client_name]["x"]
                ]
                client_labels = torch.tensor(data_dict[client_name]["y"]).long()

                self.texts.extend(client_texts)
                self.labels.append(client_labels)

        if self.labels:
            self.labels = torch.cat(self.labels, dim=0)
        else:
            self.labels = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Return the input_ids tensor (squeeze to remove batch dimension)
        return inputs["input_ids"].squeeze(0), label
