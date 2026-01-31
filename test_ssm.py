import torch
from models.latent_ssm import LatentSSM

model = LatentSSM(latent_dim=2)

# joint latent state: 2 agents × latent_dim
z0 = torch.zeros(1, 4)

z1, dist = model(z0)
print("z1 shape:", z1.shape)
