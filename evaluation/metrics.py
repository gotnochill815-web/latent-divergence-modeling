import time
import torch
import torch.nn.functional as F


# ==========================================================
# Reconstruction Metrics
# ==========================================================

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()


def mae(y_true, y_pred):
    return F.l1_loss(y_pred, y_true).item()


def mse(y_true, y_pred):
    return F.mse_loss(y_pred, y_true).item()


# ==========================================================
# Divergence Metrics
# ==========================================================

def mean_divergence(z1, z2):
    return torch.norm(z1 - z2, dim=1).mean().item()


def max_divergence(z1, z2):
    return torch.norm(z1 - z2, dim=1).max().item()


def cosine_divergence(z1, z2):

    sim = F.cosine_similarity(z1, z2)

    return (1 - sim).mean().item()


# ==========================================================
# Smoothness
# ==========================================================

def smoothness(z):

    if len(z) < 2:
        return 0.0

    return ((z[1:] - z[:-1]) ** 2).mean().item()


# ==========================================================
# Runtime Helper
# ==========================================================

class Timer:

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start


# ==========================================================
# Master Evaluation
# ==========================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    z1=None,
    z2=None,
    runtime=None,
):

    results = {

        "Model": model_name,

        "RMSE": rmse(y_true, y_pred),

        "MAE": mae(y_true, y_pred),

        "MSE": mse(y_true, y_pred),

    }

    if z1 is not None and z2 is not None:

        results["Mean Divergence"] = mean_divergence(z1, z2)

        results["Max Divergence"] = max_divergence(z1, z2)

        results["Cosine Divergence"] = cosine_divergence(z1, z2)

        results["Smoothness"] = smoothness(
            torch.cat([z1, z2], dim=1)
        )

    else:

        results["Mean Divergence"] = None
        results["Max Divergence"] = None
        results["Cosine Divergence"] = None
        results["Smoothness"] = None

    results["Runtime"] = runtime

    return results


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    y = torch.randn(300,2)

    y_hat = y + 0.1*torch.randn(300,2)

    z1 = torch.randn(300,2)

    z2 = z1 + 0.2*torch.randn(300,2)

    metrics = evaluate_model(
        "Demo",
        y,
        y_hat,
        z1,
        z2,
        runtime=0.015,
    )

    print()

    for k,v in metrics.items():

        print(f"{k:20s}: {v}")