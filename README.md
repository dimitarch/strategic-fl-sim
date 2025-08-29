# `strategic-fl-sim`: A Package for Simulating Strategic Federated Learning

A Python package for simulating strategic Federated Learning (FL) scenarios where clients can behave adversarially by manipulating gradient updates before sending them to the server. This package enables researchers to study the robustness of different aggregation methods against strategic client behavior.

The package was originally intended to study strategic gradient manipulations in heterogeneous FL. In its current form the package is highly flexible and extensible allowing for a wide variety of clients---with different datasets, data quality, batch sizes, optimizers, local steps, gradient actions, etc. It comes with multi-GPU support out of the box (on a single node for now, but that should suffice for simulation purposes). We hope it will be useful for researchers and practitioners interested in modeling strategic behavior and incentive dynamics in FL.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dimitarch/strategicfl.git
cd strategicfl

# If using venv and requirements.txt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from strategicfl.agents import Client, Server
from strategicfl.utils.actions import create_scalar_action
from strategicfl.utils.aggregation import get_aggregate
from strategicfl.utils.evaluate import evaluate_with_ids
from strategicfl.utils.metrics import get_gradient_metrics

# Create server with robust aggregation
server = Server(
    device=device,
    model=CNN(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.06),
    aggregate=get_aggregate(method="mean")
)

# Create clients with strategic behavior
honest_client = Client(
    device=device,
    train_dataloader=train_dataloader,  # Depending on the dataset
    test_dataloader=test_dataloader,    # Depending on the dataset
    model=CNN(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.06),
    action=create_scalar_action(alpha=1.0, beta=0.0)  # Honest
)

adversarial_client = Client(
    device=device,
    train_dataloader=train_dataloader,  # Depending on the dataset
    test_dataloader=test_dataloader,    # Depending on the dataset
    model=CNN(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.06),
    action=create_scalar_action(alpha=2.0, beta=0.1)  # Strategic
)

# Train the federated model
losses, metrics = server.train(
    clients=clients,
    T=config.training.T,
    get_metrics=get_gradient_metrics, # See below. Custom metrics function to extract per-step metrics; otherwise, memory usage is too high to store the entire history.
)
```

## Demos and Experiments

The package includes ready-to-use demo Jupyter notebooks for the three datasets---FeMNIST, Shakespeare and Sent140/Twitter (all generated from [LEAF](https://leaf.cmu.edu)). We also include example experiments with config management, ready for scheduling on GPU clusters.

### Data

We use the [LEAF](https://leaf.cmu.edu) datasets, and we use the following command from their toolkit for generating the data: ```./preprocess.sh -s niid --sf 1.0 -k 0 --tf 0.8 -t sample```.

### FEMNIST
- **Task**: Handwritten character recognition (62 classes)
- **Model**: Convolutional Neural Network
- **Clients**: Writers with different handwriting styles

### Shakespeare
- **Task**: Next character prediction
- **Model**: LSTM with character embeddings
- **Clients**: Different Shakespeare plays/characters

### Twitter
- **Task**: Sentiment analysis (binary classification)
- **Model**: BERT-based classifier
- **Clients**: Different Twitter users

## Training Metrics

We use a custom metrics function to extract gradient metrics at each training step. Otherwise, the memory usage is too high to store the entire history of client and aggregated gradients. Perhaps, a bit hacky, but sufficient for research purposes.

## Strategic Behaviors Already in the Package

### Client Actions
- **alpha**: Gradient scaling factor
- **beta**: Noise injection level (beta > 0 adds Gaussian noise)

### Aggregation Methods
- **Mean**: Standard federated averaging (vulnerable to adversaries)
- **Weighted Average**: Weight by dataset size
- **Median**: Coordinate-wise median (robust to outliers)
- **Trimmed Mean**: Remove largest gradients before averaging

## Extending the Package

### Adding New Aggregation Methods

```python
def custom_aggregate(gradients):
    """Your custom aggregation logic."""
    # Implementation here
    return aggregated_gradients

server = Server(
    # ...
    aggregate=custom_aggregate
)
```

### Adding New Client Strategies

```python
def create_custom_action(*params):
    def action(gradient):
        # Your strategic behavior
        return modified_gradient
    return action

client = Client(
    # ...
    action=create_custom_action(params)
)
```

## Project Structure

```
strategicfl/
├── src/
│   ├── strategicfl/    # Core package
│   │   ├── agents/         # Client and server implementations
│   │   └── utils/          # Predefined aggregation, actions, metrics, evaluation
│   ├── models/             # Models: CNN, LSTM, BERT wrapper
│   └── utils/              # Utils for io, device and config setup
├── notebooks/          # Dataset-specific Jupyter notebook demos
└── experiments/        # Dataset-specific experiments with config management ready for scheduling
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- OmegaConf
- tqdm
- transformers (optional, for Twitter experiments)
- matplotlib (optional, for notebook demos)

## Acknowledgements

We thank the authors of [LEAF](https://leaf.cmu.edu) for their work in providing the datasets. We acknowledge help from [Claude](https://claude.ai) during development.

## Citation

If you found this helpful and used this package in your research, please use the following citation:

```bibtex
@misc{chakarov2025incentivizing,
      title={Incentive-Compatible Collaboration in Heterogeneous Federated Learning},
      author={Dimitar Chakarov and Nikita Tsoy and Kristian Minchev and Nikola Konstantinov},
      year={2025},
      eprint={2412.00980},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.00980},
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
