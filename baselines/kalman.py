import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBaseline:

    def __init__(self, dim=2):

        self.kf = KalmanFilter(dim_x=dim, dim_z=dim)

        self.kf.F = np.eye(dim)

        self.kf.H = np.eye(dim)

        self.kf.P *= 10

        self.kf.R *= 0.5

        self.kf.Q *= 0.01

    def run(self, observations):

        predictions = []

        for obs in observations:

            self.kf.predict()

            self.kf.update(obs)

            predictions.append(self.kf.x.copy())

        return np.array(predictions)


if __name__ == "__main__":

    obs = np.random.randn(300, 2)

    model = KalmanBaseline()

    pred = model.run(obs)

    print(pred.shape)
