import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, channels, latent_dim=16):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_mu = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x).squeeze(-1)

        z = self.fc_mu(x)
        return z
    
class Decoder(nn.Module):
    def __init__(self, channels, latent_dim=16, target_length=500):
        super().__init__()

        self.initial_size = target_length // 4

        self.fc = nn.Linear(latent_dim, 128 * self.initial_size)

        self.block1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(32, channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, self.initial_size)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_dim=16, target_length=500):
        super().__init__()
        self.encoder = Encoder(channels, latent_dim)
        self.decoder = Decoder(channels, latent_dim, target_length)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)

        if recon_x.size(-1) != x.size(-1):
            recon_x = F.interpolate(recon_x, size=x.size(-1), mode='linear', align_corners=False)

        return recon_x, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


def loss_function(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean')