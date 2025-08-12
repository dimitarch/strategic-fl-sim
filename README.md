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
    get_metrics=get_gradient_metrics, # Custom metrics function to extract per-step metrics; otherwise, memory usage is too high to store the entire history
)
```

### Examples

The framework includes ready-to-use examples for the three datasets: FeMNIST, Shakespeare and Sent140/Twitter.

### Metrics during Training

Use a custom metrics function to extract gradient metrics for each step. Otherwise, the memory usage is too high to store the entire history of client and aggregated gradients.

## Strategic Behavior

### Client Actions
- **alpha**: Gradient scaling factor
- **beta**: Noise injection level (beta > 0 adds Gaussian noise)

### Aggregation Methods
- **Mean**: Standard federated averaging (vulnerable to adversaries)
- **Weighted Average**: Weight by dataset size
- **Median**: Coordinate-wise median (robust to outliers)
- **Trimmed Mean**: Remove largest gradients before averaging

## Datasets

### Data

Download a compressed folder of the data files from [here](https://drive.google.com/file/d/1imILs8cKVf_ex3t3DpvGz7aoaugAjO4M/view?usp=sharing). We use the [LEAF](https://leaf.cmu.edu) datasets with some postprocessing to make them more uniform for loading in Python. Unzip and place in the base directory of the repo.

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
├── strategicfl/          # Core framework
│   ├── agents/           # Client and server implementations
│   └── utils/            # Predefined aggregation, actions, metrics, evaluation
├── examples/            # Dataset-specific experiments
├── models/              # Models: CNN, LSTM, BERT wrapper
└── utils/               # Configuration and utilities
```

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

## Citation

If you found this helpful and used this framework in your research, please use the following citation:

```bibtex
@misc{chakarov2025incentivizingtruthfulcollaborationheterogeneous,
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
