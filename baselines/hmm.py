import time
import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMBaseline:
    """
    Hidden Markov Model baseline.
    """

    def __init__(
        self,
        n_states=3,
    ):

        self.n_states = n_states

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
        )

        self.state_means = None

    def fit(self, observations):

        self.model.fit(observations)

        states = self.model.predict(observations)

        self.state_means = np.zeros(
            (self.n_states, observations.shape[1])
        )

        for s in range(self.n_states):

            idx = states == s

            if np.any(idx):
                self.state_means[s] = observations[idx].mean(axis=0)

    def run(self, observations):

        start = time.time()

        states = self.model.predict(observations)

        predictions = np.array(
            [self.state_means[s] for s in states]
        )

        runtime = time.time() - start

        return predictions, states, runtime


if __name__ == "__main__":

    obs = np.random.randn(300, 2)

    hmm = HMMBaseline()

    hmm.fit(obs)

    pred, states, runtime = hmm.run(obs)

    print("Prediction Shape :", pred.shape)

    print("States Shape :", states.shape)

    print("Runtime :", runtime)