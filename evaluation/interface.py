import time
import torch

from experiments.evaluate import Evaluator


class VSSMInterface:
    """
    Wrapper around the proposed Variational State Space Model.

    Provides a common interface for benchmarking.
    """

    def __init__(self):

        self.evaluator = Evaluator()

    def run(self):

        start = time.time()

        results = self.evaluator.run()

        runtime = time.time() - start

        return {
            "y_true": results["y1"],
            "y_pred": results["y1_hat"],
            "z1": results["z1"],
            "z2": results["z2"],
            "runtime": runtime,
            "metrics": results["metrics"],
        }


if __name__ == "__main__":

    model = VSSMInterface()

    output = model.run()

    print(output.keys())