import os
import shutil
import tempfile
import json
import logging
from matplotlib.pyplot import stem
import numpy as np
import pandas as pd
import joblib
import mne
from scipy.signal import stft
import tensorflow as tf
from tensorflow.keras.models import Model

from src.services.preprocessing_service import Preprocess

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
CNN_MODEL_PATH = os.path.normpath(os.path.join(_MODELS_DIR, "epilepsy_diagnosis/seizure_model.keras"))
CLASSIFIER_MODEL_PATH = os.path.normpath(os.path.join(_MODELS_DIR, "epilepsy_diagnosis/svm_model.pkl"))

logger = logging.getLogger(__name__)


class EpilepsyPipeline:

    def __init__(self):
        self.preprocessor = Preprocess()
        self.stft_features = None
        self.predictions = []
        self.n_channels = 8
        self.fs = 250
        self.nperseg = 64
        self.noverlap = 32
        self.feature_extractor_model = None
        self.classifier_model = None
        

    def process_eeg_epoch(self, data):
        """Convert EEG data to stft features."""

        channel_specs = []
        for i in range(self.n_channels):
            f, t, Sxx = stft(
                data[i],
                fs=self.fs,
                window='hann',
                nperseg=self.nperseg,
                noverlap=self.noverlap,
            )
            # Power in dB, with a small epsilon to avoid log(0)
            pw = 10 * np.log10((np.abs(Sxx) ** 2) + 1e-10)
            channel_specs.append(pw)
        # Stack along a new axis so the last dimension is the channel index
        return np.stack(channel_specs, axis=-1)

    def adjust_feature_shapes(self, arr: np.ndarray, target_shape=(33, 48, 8)):

        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

        h, w, c = arr.shape
        th, tw, tc = target_shape

        # TRIM
        arr = arr[:min(h, th), :min(w, tw), :min(c, tc)]

        # PAD
        pad_h = max(0, th - arr.shape[0])
        pad_w = max(0, tw - arr.shape[1])
        pad_c = max(0, tc - arr.shape[2])

        arr = np.pad(
            arr,
            ((0, pad_h), (0, pad_w), (0, pad_c)),
            mode="constant",
            constant_values=0
        )

        return arr
    
    def feature_extractor(self):
        if self.feature_extractor_model is None:
            self.feature_extractor_model = tf.keras.models.load_model(CNN_MODEL_PATH)

        # Force graph building by running a dummy prediction. This ensures the model
        # is fully built and ready for calls, avoiding lazy initialization
        # issues that can arise when the model is first used.
        _ = self.feature_extractor_model.predict(np.zeros((1, 33, 48, 8)), verbose=0)

        extractor = Model(
            inputs=self.feature_extractor_model.layers[0].input,
            outputs=self.feature_extractor_model.layers[-3].output
        )
        # extract 64 features per epoch
        features = extractor.predict(self.stft_features, verbose=0)
        return features


    def run(self, raw_obj):
        # Preprocess EEG and create 6s epochs

        stft_list = []

        epochs = self.preprocessor.preprocess(raw_obj)

        for i, epoch_data in enumerate(epochs):
            feat = self.process_eeg_epoch(epoch_data)
            adj_feature = self.adjust_feature_shapes(feat)
            stft_list.append(adj_feature)

        self.stft_features = np.stack(stft_list, axis=0)  # shape (n_epochs, 33, 48, 8)
        print(f"Final stft feature shape: {self.stft_features.shape}")

        features = self.feature_extractor()

        if self.classifier_model is None:
            self.classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)
        
        predictions = self.classifier_model.predict(features)
        print(f"Final predictions: {predictions.shape}")

        seiz_epochs = int((predictions == 1).sum())
        total_epochs = len(predictions)

        return {
            "epoch_predictions": predictions.tolist(),
            "seizure_epochs": seiz_epochs,
            "total_epochs": total_epochs,
        }


if __name__ == "__main__":

    seiz_file = 'test_data/aaaaaalq_s001_t000.edf'
    non_seiz_file = 'test_data/aaaaaaug_s004_t001.edf'

    seiz_raw = mne.io.read_raw_edf(seiz_file, preload=True, verbose=False)
    non_seiz_raw = mne.io.read_raw_edf(non_seiz_file, preload=True, verbose=False)

    pipeline = EpilepsyPipeline()
    seiz_pred = pipeline.run(non_seiz_raw)











    

            

    
   