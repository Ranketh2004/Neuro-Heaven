import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import mne 
import torch
from mne.preprocessing import ICA
from mne.decoding import Scaler
from autoreject import AutoReject
from sklearn.preprocessing import RobustScaler

from models.autoencoder import AutoEncoder, Encoder, Decoder

#from src.helper.logger import get_logger
import warnings

warnings.filterwarnings("ignore")

# logger = get_logger(__name__)

class EEGPreprocessor:

    def __init__(self, 
                 target_sfreq=250, 
                 l_freq=1.0, 
                 h_freq=40.0,
                 notch_freq=[60, 120],
                 reference='average',
                 epoch_length=2,
                 overlap=0.0
                 ):
        
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.reference = reference
        self.epoch_length = epoch_length
        self.overlap = overlap

        # common EEG channels according to 10-20 system of electrode placement
        self.common_channels = [
            'FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T5','T6','FZ','PZ'
        ]

    def pick_common_channels(self, raw):
        
        #rename channel names in .edf file
        mapping = {ch: ch.replace("EEG ", "").replace("-LE", "").replace(" ","").replace("-REF", "") 
                   for ch in raw.ch_names}
        raw.rename_channels(mapping)

        #pick only relevant channels
        picked_channels = [ch for ch in raw.info['ch_names'] if ch in self.common_channels]

        if len(picked_channels) < len(self.common_channels):
            missing = set(self.common_channels) - set(picked_channels)
            print(f"Warning: Missing channels - {missing}")
        
        raw.pick(picked_channels)

        return raw

    def downsample(self, raw):
        
        original_sfreq = raw.info['sfreq']
        if original_sfreq > self.target_sfreq:
            raw.resample(self.target_sfreq)
        
        return raw
    
    def set_montage(self, raw):

        # ch names as in the standard 10-20 system
        montage_ch_names = ['Fp1','Fp2','F3','F4', 'C3', 'C4', 'P3', 'P4',
                            'O1', 'O2', 'F7', 'F8', 'T5', 'T6', 'Fz','Pz']
        
        channels = {ch: c for c in montage_ch_names for ch in raw.info['ch_names'] if ch.lower() == c.lower()}
        raw.rename_channels(channels)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn', verbose=False)
    
        return raw
    
    def notch_filter(self, raw):

        raw.notch_filter(freqs=self.notch_freq, picks='eeg', verbose=False)
        return raw
    
    def bandpass_filter(self, raw):

        raw.filter(l_freq=self.l_freq,
                   h_freq=self.h_freq,
                   fir_design='firwin',
                   picks='eeg')
        
        return raw
    
    
    def rereference(self, raw):

        raw.set_eeg_reference(ref_channels=self.reference
                            , projection=False,
                              verbose=False)
    
        return raw
    

    def run_pipeline(self, file_path):

        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        raw_1 = self.pick_common_channels(raw)
        raw_2 = self.downsample(raw_1)
        raw_3 = self.set_montage(raw_2)
        raw_4 = self.notch_filter(raw_3)
        raw_5 = self.bandpass_filter(raw_4)

        # 2. Create Epochs
        events = mne.make_fixed_length_events(raw_5, duration=self.epoch_length, overlap=0)
        epochs = mne.Epochs(
            raw_5, events, tmin=0, tmax=self.epoch_length - (1 / self.target_sfreq), 
            baseline=None, preload=True, verbose=False
        )

        # 6. Scaling
        scaler = Scaler(info=epochs.info, scalings='mean')
        X_scaled = scaler.fit_transform(epochs.get_data())
        
        return X_scaled


if __name__ == "__main__":

    preprocessor = EEGPreprocessor()

    sample_file = 'data/raw/aaaaalim_s001_t001.edf'  

    processed_data = preprocessor.run_pipeline(sample_file)
    print(f'Processed data shape: {processed_data.shape}')

    processed_tensor = torch.from_numpy(processed_data).float()
    print(f'Processed tensor shape: {processed_tensor.shape}')

    model_path = 'src/models/eeg_autoencoder.pth'
    #autoencoder = AutoEncoder(channels=16, latent_dim=16, target_length=500)
    torch.serialization.add_safe_globals([AutoEncoder, Encoder, Decoder])
    #autoencoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))

    autoencoder = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    autoencoder.eval()

    with torch.no_grad():
        _, z = autoencoder(processed_tensor)
        print(f'Latent representation shape: {z.shape}')
        print(f'Latent representation: {z}')


    


