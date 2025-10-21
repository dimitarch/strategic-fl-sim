import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")  # GPU
    elif torch.backends.mps.is_built():
        return torch.device("mps")  # Apple M-series
    else:
        return torch.device("cpu")  # CPU
