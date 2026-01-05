import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import mne 
from mne.preprocessing import ICA
from autoreject import AutoReject
from sklearn.preprocessing import RobustScaler

from src.helper.logger import get_logger
import warnings

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

class EEGPreprocessor:

    def __init__(self, 
                 target_sfreq=250, 
                 l_freq=1.0, 
                 h_freq=40.0,
                 notch_freq=[60, 120],
                 reference='average',
                 epoch_length=10,
                 overlap=0.5
                 ):
        
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.reference = reference
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.scaler = RobustScaler()

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
    
        # # Verify all channels have positions
        # missing_pos = [ch for ch in raw.ch_names 
        #                if raw.info['chs'][raw.ch_names.index(ch)]['loc'][:3].sum() == 0]
    
        # if missing_pos:
        #     logger.warning(f"Channels without positions: {missing_pos}")
        #     # Drop channels without positions
        #     raw.drop_channels(missing_pos)
        #     logger.info(f"Dropped {len(missing_pos)} channels without valid positions")
    
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
    
    def run_ica(self, raw):

        ica = ICA(n_components=15,
                  max_iter='auto',
                  random_state=97,
                  method='fastica'
                  )

        ica.fit(raw, verbose=False)

        eog_inds, _ = ica.find_bads_eog(raw, ch_name='FP1')

        if eog_inds and len(eog_inds) > 2:
            eog_inds = eog_inds[:2]

        ica.exclude = eog_inds

        raw_clean = raw.copy()
        ica.apply(raw_clean, verbose=False)
        
        return raw_clean
    
    def rereference(self, raw):

        raw.set_eeg_reference(ref_channels=self.reference
                            , projection=False,
                              verbose=False)
    
        return raw
    

    def create_epochs_and_reject(self, raw):

        # 5s steps
        step_duration = self.epoch_length * (1 - self.overlap)

        events = mne.make_fixed_length_events(raw,
                                              duration= step_duration,
                                              overlap=0.0) #overlap handled in step_duration
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=self.epoch_length,
            baseline=None,
            preload=True,
            verbose=False
        )

        print(f'Original Epoch count: {len(epochs)}')

        #artifact rejection
        ar = AutoReject(
            n_interpolate=[1,2],
            random_state=42,
            n_jobs=-1,
            verbose=False
        )

        #find bad epochs, repairs sensors, and drops unfixable data
        epochs_clean, reject_log = ar.fit_transform(epochs)

        print(f'Cleaned Epoch count: {len(epochs_clean)}')

        return epochs_clean
    
    def normalize(self, epochs):

        data = epochs.get_data() # (n_epochs, n_channels, n_times)

        for i in range(data.shape[1]):
            scaler = RobustScaler()
            #reshape to (n_epochs * n_times, 1) to learn scale for a channel
            channel_data = data[:, i, :].reshape(-1, 1)
            scaler.fit(channel_data)
            # transform and put back
            data[:, i, :] = scaler.transform(channel_data).reshape(data.shape[0], data.shape[2])

        return data
    
    def run_pipeline(self, raw_path):

        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)

        raw = self.pick_common_channels(raw)
        raw = self.set_montage(raw)

        raw = self.notch_filter(raw)
        raw = self.bandpass_filter(raw)

        raw = self.downsample(raw)

        raw = self.run_ica(raw)

        raw = self.rereference(raw)

        epochs = self.create_epochs_and_reject(raw)

        final_data = self.normalize(epochs)

        return final_data

if __name__ == "__main__":

    preprocessor = EEGPreprocessor()

    sample_file = 'data/aaaaaebo_s003_t001.edf'  

    processed_data = preprocessor.run_pipeline(sample_file)

    print(f'Processed data shape: {processed_data.shape}')


