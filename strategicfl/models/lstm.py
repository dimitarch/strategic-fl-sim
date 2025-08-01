from torch import nn


class LSTM(nn.Module):
    """
    A sequence-based neural network model. The model uses an
    embedding layer, a stacked LSTM for sequential data processing, and a fully
    connected output layer for predictions.

    Attributes:
    - seq_len : int
        The length of input sequences (number of time steps).
    - num_classes : int
        The number of output classes (used for classification).
    - n_hidden : int
        The number of hidden units in each LSTM layer.
    - embedding_dim : int, optional
        The dimensionality of the embedding vectors (default is 8).

    Methods:
    - forward(x): Defines the forward pass of the model. Takes input tensor `x`, processes it through the embedding, LSTM, and output layers, and returns predictions.
    """

    def __init__(self, seq_len=80, num_classes=80, n_hidden=256, embedding_dim=8):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden

        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, n_hidden, num_layers=2, batch_first=True)

        self.fc = nn.Linear(n_hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        prediction = self.fc(last_hidden)

        return prediction
