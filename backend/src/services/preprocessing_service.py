import numpy as np
import mne

class Preprocess:

    def __init__(self):
        self.channels = ['T3', 'T4', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2']

    def pick_common_channels(self, raw_obj):
        mapping = {ch: ch.replace("EEG ", "").replace("-LE", "").replace(" ","").replace("-REF", "") 
                   for ch in raw_obj.ch_names}
        
        raw_obj.rename_channels(mapping)

        picked_channels = [ch for ch in raw_obj.info['ch_names'] if ch in self.channels]
        if (len(picked_channels) < len(self.channels)):
            missing = set(self.channels) - set(picked_channels)
            print(f'Missing channels: {missing}')   

        raw_obj.pick_channels(picked_channels)
        raw_obj.reorder_channels(['F3', 'F4','C3','C4','O1','O2','T3','T4'])
        return raw_obj
    
    def set_montage(self, raw_obj):

        channels = {ch: c for c in self.channels for ch in raw_obj.info['ch_names'] if ch.lower() == c.lower()}
        raw_obj.rename_channels(channels)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw_obj.set_montage(montage, match_case=False, on_missing='ignore')
        return raw_obj
    
    def bandpass_filter(self, 
                        raw_obj, 
                        l_freq=0.5, 
                        h_freq=40.0
                        ):
        raw_obj.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False, picks='eeg')
        return raw_obj
    
    def notch_filter(self, raw_obj):
        raw_obj.notch_filter(freqs=[60.0, 120.0], picks='eeg', verbose=False)
        return raw_obj
    
    def reference(self, raw_obj):

        anode = ['F3', 'C3', 'F4', 'C4', 'T3', 'C4', 'C3', 'F3']
        cathode = ['C3', 'O1', 'C4', 'O2', 'C3', 'T4', 'C4', 'F4']

        raw_bipolar = mne.set_bipolar_reference(
            raw_obj, anode=anode, cathode=cathode, drop_refs=True, verbose=False
        )
        return raw_bipolar
    
    def normalize(self, raw_obj):

        data = raw_obj.get_data()
        mu = np.mean(data, axis=1, keepdims=True)
        sigma = np.std(data, axis=1, keepdims=True)

        norm_data = (data - mu) / (sigma + 1e-10)
        raw_obj._data = norm_data

        return raw_obj
    
    def create_epochs(self, raw_obj, epoch_length=6.0):
        total_duration = raw_obj.times[-1]
        n_epochs = int(total_duration // epoch_length)

        epochs = []
        for i in range(n_epochs):
            tmin = i * epoch_length
            tmax = tmin + epoch_length
            epoch_data = raw_obj.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False).get_data()
            epochs.append(epoch_data)

        return epochs

    def preprocess(self, raw_obj):
        
        #raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw = self.pick_common_channels(raw_obj)
        raw = self.set_montage(raw)
        raw = self.notch_filter(raw)
        raw = self.bandpass_filter(raw)
        raw = self.reference(raw)
        raw = self.normalize(raw)
        epochs = self.create_epochs(raw)
        
        return epochs

