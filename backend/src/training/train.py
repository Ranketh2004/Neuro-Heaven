import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

class H5Dataset(Dataset):
    def __init__(self, file_path, data_key= 'X'):
        
        self.file_path = file_path
        self.data_key = data_key
        with h5py.File(self.file_path, 'r') as f:
            self.length = f[self.data_key].shape[0]
            self.data_shape = f[self.data_key].shape[1:]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            data = f[self.data_key][idx]
        return torch.FloatTensor(data)
    
def train(
        h5_path, 
        data_key='X', 
        channels=16, 
        latent_dim=16, 
        target_length=500,
        batch_size=32,
        epochs=50,
        learning_rate=1e-3,
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        ):
    
    dataset = H5Dataset(h5_path, data_key)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = Autoencoder(channels, latent_dim, target_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []

    print("Starting training...")
    print(f"Using device: {device}")
    print(f"Number of training samples: {len(dataset)}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for idx, data in enumerate(dataloader):
            data = data.to(device)

            optimizer.zero_grad()
            recon_data, z = model(data)
            loss = loss_function(recon_data, data)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

    return model, train_losses

