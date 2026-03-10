# backend/src/services/mri_inference_service.py
# Keras-based FCD binary classifier with Grad-CAM explanation.

import os
import io
import base64
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from .mri_preprocessing_service import (
    preprocess_mri_for_inference,
    IMG_SIZE,
    PATCH_SIZE,
)


# Custom layer used in the saved model
@tf.keras.utils.register_keras_serializable(package="FCD")
class ZeroChannel(tf.keras.layers.Layer):
    def call(self, x):
        return tf.zeros_like(x[..., :1])


class MRIFCDInferenceService:
    def __init__(self, model_path: str, img_size: int = IMG_SIZE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.img_size = img_size
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"ZeroChannel": ZeroChannel},
        )
        self.model.trainable = False
        self.last_conv_name = self._find_last_conv_layer()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _find_last_conv_layer(self) -> str | None:
        """Walk layers in reverse and return the name of the last 4-D feature map layer."""
        for layer in reversed(self.model.layers):
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return layer.name
        return None

    def _gradcam_heatmap(self, img_tensor: tf.Tensor) -> np.ndarray:
        if self.last_conv_name is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)

        grad_model = tf.keras.Model(
            self.model.inputs,
            [self.model.get_layer(self.last_conv_name).output, self.model.outputs[0]],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_tensor)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    @staticmethod
    def _overlay_heatmap(gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.50) -> np.ndarray:
        gray_u8 = (gray * 255).astype(np.uint8)
        hm_u8 = (heatmap * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB) / 255.0
        out = (1 - alpha) * np.stack([gray, gray, gray], axis=-1) + alpha * hm_color
        return np.clip(out, 0, 1)

    @staticmethod
    def _array_to_b64_png(arr: np.ndarray) -> str:
        img_u8 = (arr * 255).astype(np.uint8)
        pil = Image.fromarray(img_u8)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, flair_path: str, t1_path: str | None = None):
        """
        Run the full pipeline: preprocess → predict → Grad-CAM.

        Returns dict with:
          fcd_probability, prediction, image_b64 (Grad-CAM overlay),
          best_slice_info, num_patches
        """
        X, meta = preprocess_mri_for_inference(flair_path, t1_path)

        probs = self.model.predict(X, batch_size=32, verbose=0).ravel()

        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])

        # Grad-CAM on the highest-probability patch
        best_patch = X[max_idx : max_idx + 1]
        heatmap = self._gradcam_heatmap(tf.convert_to_tensor(best_patch))
        heatmap = cv2.resize(heatmap, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Build overlay image
        flair_channel = best_patch[0, :, :, 0]  # FLAIR channel of best patch
        overlay = self._overlay_heatmap(flair_channel, heatmap)
        overlay_b64 = self._array_to_b64_png(overlay)

        # Full brain slice with detection region circled (only if FCD detected)
        info = meta["slice_info"][max_idx]
        flair3d = meta["flair3d_normed"]          # (H, W, D), already 0-1
        full_slice = flair3d[:, :, info["z"]]     # full axial slice
        h, w = full_slice.shape
        # Convert to RGB uint8
        full_u8 = (full_slice * 255).astype(np.uint8)
        full_rgb = cv2.cvtColor(full_u8, cv2.COLOR_GRAY2BGR)
        
        # Only draw circle if FCD is detected (probability >= 0.5)
        if max_prob >= 0.5:
            cy, cx = info["cy"], info["cx"]
            radius = PATCH_SIZE // 2
            cv2.circle(full_rgb, (cx, cy), radius, (0, 255, 0), 2)
        
        # Convert BGR→RGB for PIL
        full_rgb = cv2.cvtColor(full_rgb, cv2.COLOR_BGR2RGB)
        mri_b64 = self._array_to_b64_png(full_rgb.astype(np.float32) / 255.0)

        return {
            "fcd_probability": round(max_prob, 4),
            "prediction": "FCD Detected" if max_prob >= 0.5 else "No FCD Detected",
            "image_b64": overlay_b64,
            "mri_b64": mri_b64,
            "best_slice_info": meta["slice_info"][max_idx],
            "num_patches": len(probs),
        }
