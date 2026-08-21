import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
import scipy.io

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)   # <-- NO sigmoid for StandardScaler data
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        logvar = torch.clamp(logvar, -10.0, 10.0)  # stabilizer
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar



class TotalLoss(nn.Module):
    def forward(self, x, recon, mu, logvar):
        # recon: per-sample then mean
        recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=1).mean()

        # clamp logvar BEFORE exp to prevent overflow
        logvar = torch.clamp(logvar, -10.0, 10.0)

        # KL per-sample then mean
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        total = recon_loss + kld
        return total, recon_loss.detach(), kld.detach()
