from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def create_default_config():
    """Create default configuration."""
    config = {
        "experiment": {"id": "", "save_dir": "./results/femnist"},
        "training": {"T": 50, "lr": 0.06, "local_steps": 1},
        "clients": {
            "n_players": 5,
            "alpha_0": 1.0,
            "alpha_1": 1.0,
            "beta_0": 0.0,
            "beta_1": 0.0,
        },
        "aggregation": {
            "method": "mean",  # "weighted_average", "median", "trimmed_mean"
        },
        "data": {
            "train_path": "./data/femnist/train.json",
            "test_path": "./data/femnist/test.json",
        },
    }
    return OmegaConf.create(config)


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """Load configuration from file or create default."""
    if config_path is not None and Path(config_path).exists():
        print(f"Loading config from {config_path}")
        config = OmegaConf.load(config_path)

        # Merge with default config to ensure all keys exist
        default_config = create_default_config()
        config = OmegaConf.merge(default_config, config)
    else:
        if config_path is not None:
            print(f"Config file {config_path} not found, using default config")
        config = create_default_config()

    return config


def save_config(config: DictConfig, save_path: str):
    """Save configuration to file."""
    config_path = f"{save_path}_config.yaml"
    OmegaConf.save(config, config_path)
    print(f"Configuration saved to {config_path}")
