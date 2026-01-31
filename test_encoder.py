import torch
from models.encoder import Encoder

B = 4
obs_dim = 2
latent_dim = 2

encoder = Encoder(obs_dim, latent_dim)

y1 = torch.randn(B, obs_dim)
y2 = torch.randn(B, obs_dim)

z, dist = encoder(y1, y2)

print("z shape:", z.shape)
print("mean shape:", dist.mean.shape)
