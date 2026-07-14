import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMBaseline:

    def __init__(
        self,
        n_states=3,
    ):

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
        )

    def fit(self, observations):

        self.model.fit(observations)

    def predict(self, observations):

        return self.model.predict(observations)


if __name__ == "__main__":

    obs = np.random.randn(300, 2)

    hmm = HMMBaseline()

    hmm.fit(obs)

    states = hmm.predict(obs)

    print(states.shape)
