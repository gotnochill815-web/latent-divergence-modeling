import time
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

        self.fc = nn.Linear(
            hidden_dim,
            output_dim,
        )

    def forward(self, x):

        h, _ = self.lstm(x)

        out = self.fc(h)

        return out

    def run(self, observations):
        """
        observations:
            Tensor of shape (T, input_dim)

        Returns:
            predictions, runtime
        """

        self.eval()

        with torch.no_grad():

            start = time.time()

            x = observations.unsqueeze(0)

            pred = self.forward(x)

            runtime = time.time() - start

        return pred.squeeze(0), runtime


if __name__ == "__main__":

    model = LSTMBaseline()

    observations = torch.randn(300, 4)

    pred, runtime = model.run(observations)

    print("Prediction Shape :", pred.shape)

    print("Runtime :", runtime)