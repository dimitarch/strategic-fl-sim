# StrategicFL: A Framework for Strategic Federated Learning

A Python framework for simulating strategic federated learning scenarios where clients can behave adversarially by manipulating gradient updates before sending them to the server. This framework enables researchers to study the robustness of different aggregation methods against strategic client behavior.

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
from strategicfl.actions import create_scalar_action
from strategicfl.aggregation import get_aggregate
from strategicfl.models import CNN
from strategicfl.trainer import train

# Create server with robust aggregation
server = Server(
    device=device,
    model=CNN(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.06),
    aggregate=get_aggregate(method="median")  # Robust to adversaries
)

# Create clients with strategic behavior
honest_client = Client(
    device=device,
    dataloader=dataloader,
    model=CNN(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    action=create_scalar_action(alpha=1.0, beta=0.0)  # Honest
)

adversarial_client = Client(
    device=device,
    dataloader=dataloader,
    model=CNN(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    action=create_scalar_action(alpha=2.0, beta=0.1)  # Strategic
)

# Train the federated model
model, losses, metrics = train(
    server=server,
    clients=[honest_client, adversarial_client],
    T=1000,  # Training rounds
    K=1      # Local steps
)
```

### Running Experiments

The framework includes ready-to-use experiments for three datasets:

```bash
# FEMNIST (handwritten characters)
cd experiments/femnist
python experiment.py --config config.yaml

# Shakespeare (next character prediction)
cd experiments/shakespeare
python experiment.py --config config.yaml

# Twitter (sentiment analysis)
cd experiments/twitter
python experiment.py --config config.yaml
```

## Configuration

Experiments are configured using YAML files. Example configuration:

```yaml
experiment:
  id: "strategic_experiment"
  save_dir: "./results/femnist"

training:
  T: 1000                    # Training rounds
  lr: 0.06                   # Learning rate
  local_steps: 1             # Local SGD steps

clients:
  n_players: 3               # Clients per round
  alpha_0: 1.0               # Honest client scaling
  alpha_1: 2.0               # Adversarial client scaling
  beta_0: 0.0                # Honest client noise
  beta_1: 0.1                # Adversarial client noise

aggregation:
  method: "median"           # Aggregation method
```

## Strategic Behavior

### Client Actions
- **α (Alpha)**: Gradient scaling factor (α > 1 amplifies, α < 1 diminishes)
- **β (Beta)**: Noise injection level (β > 0 adds Gaussian noise)

### Aggregation Methods
- **Mean**: Standard federated averaging (vulnerable to adversaries)
- **Weighted Average**: Weight by dataset size
- **Median**: Coordinate-wise median (robust to outliers)
- **Trimmed Mean**: Remove largest gradients before averaging

## Datasets

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

## Project Structure

```
strategicfl/
├── strategicfl/           # Core framework
│   ├── agents/           # Client and server implementations
│   ├── models/           # Neural network architectures
│   ├── actions.py        # Strategic client behaviors
│   ├── aggregation.py    # Robust aggregation methods
│   └── trainer.py        # Federated training loop
├── experiments/          # Dataset-specific experiments
├── utils/               # Configuration and utilities
├── data/                # Raw datasets (via Git LFS)
└── results/             # Experiment outputs (via Git LFS)
```

## Results and Analysis

Each experiment saves comprehensive results including:
- **Training losses** per round and client
- **Gradient norms** and cosine similarities
- **Final test accuracies** for each client group
- **Configuration** for full reproducibility

Results are saved as pickle files and can be analyzed using standard Python data science tools.

## Extending the Framework

### Adding New Aggregation Methods

```python
def custom_aggregate(gradients):
    """Your custom aggregation logic."""
    # Implementation here
    return aggregated_gradients

server = Server(
    # ...
    aggregate=custom_aggregate  # Robust to adversaries
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
    action=create_custom_action(params)  # Honest
)
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- OmegaConf
- tqdm
- transformers (optional, for Twitter experiments)

## Acknowledgements

We thank the authors of [LEAF](https://leaf.cmu.edu). We are also grateful for [Claude](https://claude.ai).

<!-- ## Citation

If you use this framework in your research, please cite:

<!-- ```bibtex
@software{chakarov2025strategicfl,
  author = {Chakarov, Dimitar},
  title = {StrategicFL: A Framework for Strategic Federated Learning},
  url = {https://github.com/yourusername/strategicfl},
  version = {0.1.0},
  year = {2025}
}
``` --> -->

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
