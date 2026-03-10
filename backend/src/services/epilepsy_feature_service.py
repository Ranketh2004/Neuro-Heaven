import os
import numpy as np
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import random
import matplotlib
matplotlib.use('Agg')
from PIL import Image

# Spectrogram parameters
FS = 250
N_CHANNELS = 8
NPERSEG = 500
NOVERLAP = 490

def process_eeg_epoch(data):
    """Convert EEG data to a 3D Stacked Spectrogram (Freq, Time, Channels)."""
    channel_specs = []
    for i in range(N_CHANNELS):
        f, t, Sxx = signal.spectrogram(
            data[i],
            fs=FS,
            window='hann',
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            detrend='constant'
        )
        
        # Convert to log scale
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        
        # Keep frequencies 0-50 Hz
        mask = (f >= 0) & (f <= 50)
        Sxx_masked = Sxx_log[mask, :]
        
        channel_specs.append(Sxx_masked)

    # 1. Create Stacked Depth (Freq, Time, Channels)
    # This is the (26, 24, 8) shape for the CNN
    stacked = np.stack(channel_specs, axis=-1)
    
    # 2. Normalize (Z-score) - Crucial for CNN feature extraction
    stacked = (stacked - np.mean(stacked)) / (np.std(stacked) + 1e-10)
    
    return stacked

def save_spectrogram_as_image(spectrogram_3d, output_path):
    """Stack all 8 channels vertically, resize, and save as a single image."""
    # spectrogram_3d has shape (Freq, Time, Channels)
    
    # Create a list of channels to stack, each with shape (Freq, Time)
    channels_to_stack = [spectrogram_3d[:, :, i] for i in range(spectrogram_3d.shape[2])]
    
    # Vertically stack the channels to create a single tall image
    stacked_spectrogram = np.vstack(channels_to_stack)

    # Normalize to 0-255 for image conversion
    img_array = (stacked_spectrogram - np.min(stacked_spectrogram)) / (np.max(stacked_spectrogram) - np.min(stacked_spectrogram))
    img_array = (img_array * 255).astype(np.uint8)
    
    # Create image and resize
    img = Image.fromarray(img_array)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Apply a colormap using matplotlib
    cmap = plt.get_cmap('magma')
    colored_img_array = cmap(np.array(img))
    
    # Convert to RGBA image and save
    colored_img = Image.fromarray((colored_img_array * 255).astype(np.uint8))
    colored_img.save(output_path)