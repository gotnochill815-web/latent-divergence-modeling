import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    LSTM baseline for paired time-series prediction.
    """

    def __init__(
        self,
        input_dim=4,
        hidden_dim=64,
        num_layers=2,
        output_dim=4,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x : (batch, seq_len, input_dim)
        """

        h, _ = self.lstm(x)

        out = self.fc(h)

        return out


if __name__ == "__main__":

    model = LSTMBaseline()

    x = torch.randn(8, 300, 4)

    y = model(x)

    print(y.shape)
