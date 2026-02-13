# `strategic-fl-sim`: A Package for Simulating Strategic Federated Learning

A Python package for simulating strategic Federated Learning (FL) scenarios where clients can behave adversarially by manipulating gradient updates before sending them to the server. This package enables researchers to study the robustness of different aggregation methods against strategic client behavior.

The package was originally intended to study strategic gradient manipulations in heterogeneous FL. In its current form the package is highly flexible and extensible allowing for a wide variety of clients---with different datasets, data quality, batch sizes, optimizers, local steps, gradient actions, etc. It comes with multi-GPU support out of the box (on a single node for now, but that should suffice for simulation purposes). We hope it will be useful for researchers and practitioners interested in modeling strategic behavior and incentive dynamics in FL.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dimitarch/strategic-fl-sim.git
cd strategic-fl-sim

# If using venv and requirements.txt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from strategicflsim.agents import Client, Server
from strategicflsim.utils.actions import ScalarAction, SignFlippingAction
from strategicflsim.utils.aggregation import MeanAggregation
from strategicflsim.utils.evaluate import evaluate_with_ids
from strategicflsim.utils.metrics import NormMetrics

# Create server with robust aggregation
server = Server(
    device=device,
    model=Model(), # Generic PyTorch compatible model
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.06),
    aggregate=MeanAggregator # e.g. TrimmedMeanAggregator for a robust aggregator
)

# Create clients with strategic behavior
honest_client = Client(
    device=device,
    train_dataloader=train_dataloader,  # Depending on the dataset
    test_dataloader=test_dataloader,    # Depending on the dataset
    model=CNN(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.06),
    action=ScalarAction(alpha=1.0, beta=0.0)  # Honest
)

adversarial_client = Client(
    device=device,
    train_dataloader=train_dataloader,  # Depending on the dataset
    test_dataloader=test_dataloader,    # Depending on the dataset
    model=CNN(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.06),
    action=SignFlippingAction()  # Malicious
)

# Train the federated model
losses, metrics = server.train(
    clients=clients,
    T=config.training.T,
    selector_fn=RandomSelection(fraction=0.5)
    metrics_fn=NormMetrics(), # See below. Custom metrics hook to extract per-step metrics.
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
class CustomAggregator(ABC):
    def __init__(self, params):
        pass

    def __call__(self, params) -> List[torch.Tensor]:
        pass

server = Server(
    # ...
    aggregate=CustomAggregator()
)
```

### Adding New Client Strategies

```python
class LossAwareScaling(BaseAction):
    def __init__(self, threshold=1.0, scale=1.0):
        self.threshold = threshold
        self.scale = scale

    def __call__(self, gradient, **kwargs):
        client_state = kwargs.get(...)
        # Scale if loss too high
        if client_state.get('loss', 0) > self.threshold:
            return self.scale * gradient
        return gradient

client = Client(
    # ...
    action=LossAwareScaling()
)
```

## Project Structure

```
strategic-fl-sim/
├── src/
│   ├── strategicflsim/    # Core package
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
@inproceedings{
    chakarov2025strategicflsim,
    title={\${\textbackslash}texttt\{strategic-fl-sim\}\$: An Extensible Package for Simulating Strategic Behavior in Federated Learning},
    author={Dimitar A. Chakarov and Nikola Konstantinov},
    booktitle={NeurIPS 2025 Workshop: Reliable ML from Unreliable Data},
    year={2025},
    url={https://openreview.net/forum?id=JHgxY5vu9Y}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
