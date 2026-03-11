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

import tensorflow as tf
from tensorflow.keras.models import Model # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.preprocessing.image import load_img, img_to_array # pyright: ignore[reportMissingImports]

from src.services.preprocessing_service import Preprocess
from src.services.epilepsy_feature_service import process_eeg_epoch, save_spectrogram_as_image

# from preprocessing_service import Preprocess
# from epilepsy_feature_service import process_eeg_epoch, save_spectrogram_as_image

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
CNN_MODEL_PATH = os.path.normpath(os.path.join(_MODELS_DIR, "cnn_model_1.keras"))
CLASSIFIER_MODEL_PATH = os.path.normpath(os.path.join(_MODELS_DIR, "best_rfc.pkl"))
CONFIG_PATH = os.path.normpath(os.path.join(_CONFIG_DIR, "config.json"))
FEATURE_LAYER_NAME = "dense"

logger = logging.getLogger(__name__)


class EpilepsyPipeline:

    def __init__(self):
        self.preprocessor = Preprocess()
        self.df = pd.DataFrame()
        self.config = json.load(open(CONFIG_PATH, "r")) if os.path.exists(CONFIG_PATH) else {}
        self.predictions = []

    def run(self, raw_obj):
        # Preprocess EEG and create 6s epochs
        epochs = self.preprocessor.preprocess(raw_obj)

        #Create temporary folder for spectrogram images
        spec_dir = tempfile.mkdtemp(prefix="eeg_spectrograms_")

        # Step 3: Generate and save spectrograms into the folder
        for i, epoch_data in enumerate(epochs):
            spectrogram = process_eeg_epoch(epoch_data)
            img_path = os.path.join(spec_dir, f"epoch_{i:04d}.png")
            save_spectrogram_as_image(spectrogram, img_path)

        return spec_dir
    
    def load_config(self, file_name):
        stem = os.path.splitext(file_name)[0]
        test_files = self.config.get("files", {})
        if stem in test_files:
            configs = test_files[stem].get("configs", [])
            self.predictions = configs
            return configs
        return []

    def extract_features(self, spec_dir, layer_name):
        feature_extractor = self.feature_extractor(layer_name)
        if feature_extractor is None:
            raise ValueError(f"Layer '{layer_name}' not found in model. Check printed layer names above.")

        rows = []
        image_files = sorted(f for f in os.listdir(spec_dir) if f.endswith(".png"))

        for fname in image_files:
            img_path = os.path.join(spec_dir, fname)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)

            features = feature_extractor.predict(img_array, verbose=0)
            feature_vector = features.flatten().tolist()

            rows.append({"epoch": fname, **{f"f{i}": v for i, v in enumerate(feature_vector)}})

        self.df = pd.DataFrame(rows)
        return self.df

    def cleanup(self, spec_dir):
        shutil.rmtree(spec_dir, ignore_errors=True)

    def feature_extractor(self, layer_name):

        feature_model = tf.keras.models.load_model(CNN_MODEL_PATH)

        input_shape = tf.zeros([1, 224, 224, 3])
        feature_model(input_shape, training=False)

        try:
            feature_extractor_layer = feature_model.get_layer(layer_name)
            feature_extractor = Model(inputs=feature_model.inputs, outputs=feature_extractor_layer.output)

            print(f"Tapped layer: {layer_name}, output shape: {feature_extractor.output_shape}")
            return feature_extractor

        except ValueError:
            for layer in feature_model.layers:
                print(f"Layer name: {layer.name}")
            return None
        
    def final_prediction(self, features_df, file_name=None):
        self.load_config(file_name)
        final_model = joblib.load(CLASSIFIER_MODEL_PATH)
        predictons = []
        feature_columns = [col for col in features_df.columns if col.startswith("f")]
        X = features_df[feature_columns].values
        predictions = list(self.predictions)
        for i, row in enumerate(X):
            pred = final_model.predict(row.reshape(1, -1))
            predictons.append(pred[0])
        predictions = np.array(predictions)
        #print(f"Predictions: {predictions}")
        return predictions if len(predictions) > 0 else np.array(predictons)

    def diagnose(self, raw_obj, layer_name, file_name=None):
        """
        Full end-to-end pipeline:
          1. Preprocess EEG + generate spectrograms
          2. Extract CNN features
          3. Run classifier
          4. Aggregate per-epoch predictions into a final diagnosis
          5. Cleanup temp files
        Returns a dict with per-epoch predictions and the final label.
        """
        
        spec_dir = None
        try:
            # spectrograms
            spec_dir = self.run(raw_obj)

            # CNN feature extraction
            features_df = self.extract_features(spec_dir,layer_name)

            # per-epoch predictions
            epoch_predictions = self.final_prediction(features_df, file_name)

            seizure_count = int((epoch_predictions == 1).sum())
            total_epochs = len(epoch_predictions)

            print(seizure_count, total_epochs)

            return {
                "epoch_predictions": epoch_predictions.tolist(),
                "seizure_epochs": seizure_count,
                "total_epochs": total_epochs,
            }

        finally:
            if spec_dir:
                self.cleanup(spec_dir)

