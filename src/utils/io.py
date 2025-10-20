import os

from omegaconf import DictConfig


def make_dir(path: str) -> None:
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    return None


def generate_save_name(config: DictConfig) -> str:
    """Generate save name from config."""
    uid = "_" + config.experiment.id if config.experiment.id != "" else ""

    save_name = (
        f"{config.experiment.save_dir}/np={config.clients.n_players}"
        f"_a0={config.clients.alpha_0}_a1={config.clients.alpha_1}"
        f"_b0={config.clients.beta_0}_b1={config.clients.beta_1}"
        f"_T={config.training.T}_lr={config.training.lr}"
        f"_ls={config.training.local_steps}"
    )

    if config.aggregation.method == "median":
        save_name += "_median"
    elif config.aggregation.method == "trimmed_mean":
        save_name += "_trimmed"

    save_name += uid

    return save_name
