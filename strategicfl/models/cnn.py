from torch import nn


class CNN(nn.Module):
    """
    A simple CNN for image classification.

    Architecture:
    - Conv Layers:
        1. Conv2d(1, 32, kernel=5, padding=2) -> ReLU -> MaxPool2d(2, stride=2)
        2. Conv2d(32, 64, kernel=5, padding=2) -> ReLU -> MaxPool2d(2, stride=2)
    - Fully Connected Layers:
        1. Linear(7*7*64, 2048) -> ReLU
        2. Linear(2048, 62)

    Methods:
    - forward(x): Performs the forward pass.
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 64, 2048), nn.ReLU(), nn.Linear(2048, 62)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)

        return x
