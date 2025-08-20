import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU
    elif torch.backends.mps.is_built():
        device = torch.device("mps")  # Apple M-series
    else:
        device = torch.device("cpu")  # CPU

    return device
