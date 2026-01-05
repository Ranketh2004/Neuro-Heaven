import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, eeg_data):
        """
        eeg_data: numpy array of shape (n_samples, n_channels, n_timepoints)
        """
        self.data = torch.FloatTensor(eeg_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class EEGVAE(nn.Module):
    def __init__(self, input_channels=64, seq_length=256, latent_dim=128):
        super(EEGVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, input_channels, kernel_size=4, stride=2, padding=1),
        )
        
        self.seq_length = seq_length
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.squeeze(-1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.unsqueeze(-1)
        x_recon = self.decoder(h)
        # Adjust output to match input length
        if x_recon.size(-1) != self.seq_length:
            x_recon = nn.functional.interpolate(x_recon, size=self.seq_length)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    beta: weight for KL divergence (beta-VAE)
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def train_vae(model, train_loader, epochs=100, lr=1e-3, beta=1.0, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model

if __name__ == "__main__":
    # Generate synthetic EEG data for demonstration
    n_samples = 1000
    n_channels = 64
    n_timepoints = 256
    eeg_data = np.random.randn(n_samples, n_channels, n_timepoints)
    
    # Create dataset and dataloader
    dataset = EEGDataset(eeg_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGVAE(input_channels=n_channels, seq_length=n_timepoints, latent_dim=128)
    
    # Train model
    trained_model = train_vae(model, train_loader, epochs=50, lr=1e-3, beta=1.0, device=device)
    
    # Save model
    torch.save(trained_model.state_dict(), 'eeg_vae_model.pth')
    print("Model saved successfully!")