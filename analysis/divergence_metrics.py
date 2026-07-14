import torch


def euclidean_divergence(z1: torch.Tensor, z2: torch.Tensor):
    """
    Euclidean distance between paired latent vectors.

    Returns:
        Tensor of shape (T,)
    """
    return torch.norm(z1 - z2, dim=1)


def cosine_divergence(z1: torch.Tensor, z2: torch.Tensor):
    """
    Cosine divergence = 1 - cosine similarity.
    """

    similarity = torch.nn.functional.cosine_similarity(z1, z2, dim=1)
    return 1.0 - similarity


def mse_divergence(z1: torch.Tensor, z2: torch.Tensor):
    """
    Mean squared latent divergence.
    """

    return ((z1 - z2) ** 2).mean(dim=1)


def mean_divergence(divergence):
    """
    Average divergence.
    """

    return divergence.mean().item()


def max_divergence(divergence):
    """
    Maximum divergence.
    """

    return divergence.max().item()


def min_divergence(divergence):
    """
    Minimum divergence.
    """

    return divergence.min().item()


def std_divergence(divergence):
    """
    Standard deviation.
    """

    return divergence.std().item()


def divergence_auc(divergence):
    """
    Area under divergence curve.
    """

    return torch.trapz(divergence).item()


def divergence_summary(z1, z2):
    """
    Compute all divergence metrics.
    """

    euc = euclidean_divergence(z1, z2)

    return {
        "Mean": mean_divergence(euc),
        "Max": max_divergence(euc),
        "Min": min_divergence(euc),
        "Std": std_divergence(euc),
        "AUC": divergence_auc(euc),
        "Cosine Mean": cosine_divergence(z1, z2).mean().item(),
        "MSE Mean": mse_divergence(z1, z2).mean().item(),
    }


if __name__ == "__main__":

    torch.manual_seed(42)

    T = 300
    latent_dim = 4

    z1 = torch.randn(T, latent_dim)
    z2 = z1 + 0.25 * torch.randn(T, latent_dim)

    metrics = divergence_summary(z1, z2)

    print("\nLatent Divergence Metrics\n")

    for key, value in metrics.items():
        print(f"{key:15s}: {value:.4f}")
