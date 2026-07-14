import time
import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBaseline:
    """
    Classical Kalman Filter baseline for state estimation.
    """

    def __init__(self, dim=2):

        self.dim = dim

        self.kf = KalmanFilter(
            dim_x=dim,
            dim_z=dim
        )

        self.kf.F = np.eye(dim)

        self.kf.H = np.eye(dim)

        self.kf.P *= 10

        self.kf.R *= 0.5

        self.kf.Q *= 0.01

    def run(self, observations):

        start = time.time()

        predictions = []

        for obs in observations:

            self.kf.predict()

            self.kf.update(obs)

            predictions.append(self.kf.x.squeeze())

        runtime = time.time() - start

        return np.array(predictions), runtime


if __name__ == "__main__":

    observations = np.random.randn(300, 2)

    model = KalmanBaseline(dim=2)

    predictions, runtime = model.run(observations)

    print("Prediction Shape :", predictions.shape)

    print("Runtime :", runtime)