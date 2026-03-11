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
    def _create_brain_mask(gray: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        """Create a binary mask that covers only the brain region.
        Uses Otsu thresholding + morphological cleanup for robustness."""
        gray_u8 = (gray * 255).astype(np.uint8)
        # Otsu threshold to separate brain from background
        _, mask = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # If Otsu gives poor results, fallback to simple threshold
        if mask.sum() < 100:
            mask = (gray > threshold).astype(np.uint8) * 255
        # Morphological close to fill small holes, then open to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # Optional: keep only the largest connected component (the brain)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # label 0 is background; find largest foreground component
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = ((labels == largest) * 255).astype(np.uint8)
        return mask  # uint8, 0 or 255

    @staticmethod
    def _overlay_heatmap(gray: np.ndarray, heatmap: np.ndarray,
                         alpha: float = 0.50, brain_mask: np.ndarray | None = None) -> np.ndarray:
        """Overlay Grad-CAM heatmap on a grayscale image.
        If brain_mask is provided, restrict the heatmap to brain-only pixels."""
        # Mask the heatmap to brain region only
        if brain_mask is not None:
            mask_f = (brain_mask / 255.0).astype(np.float32)
            heatmap = heatmap * mask_f

        gray_u8 = (gray * 255).astype(np.uint8)
        hm_u8 = (heatmap * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB) / 255.0

        gray_rgb = np.stack([gray, gray, gray], axis=-1)

        if brain_mask is not None:
            # Only blend inside the brain; outside stays pure grayscale
            mask_3ch = np.stack([mask_f, mask_f, mask_f], axis=-1)
            blended = (1 - alpha) * gray_rgb + alpha * hm_color
            out = np.where(mask_3ch > 0.5, blended, gray_rgb)
        else:
            out = (1 - alpha) * gray_rgb + alpha * hm_color

        return np.clip(out, 0, 1)

    @staticmethod
    def _create_fcd_overlay(gray: np.ndarray, heatmap: np.ndarray,
                            brain_mask: np.ndarray | None = None,
                            heatmap_thresh: float = 0.40,
                            alpha: float = 0.55) -> np.ndarray:
        """Create a B&W MRI image with a purple patch where FCD is detected.
        The FCD region is determined by thresholding the Grad-CAM heatmap."""
        # Build the B&W base image
        bw = np.stack([gray, gray, gray], axis=-1).copy()

        # Threshold heatmap to get FCD region
        fcd_region = (heatmap >= heatmap_thresh).astype(np.float32)

        # Restrict to brain mask if available
        if brain_mask is not None:
            mask_f = (brain_mask / 255.0).astype(np.float32)
            fcd_region = fcd_region * mask_f

        # Smooth the edges of the FCD region for a nicer look
        fcd_region_u8 = (fcd_region * 255).astype(np.uint8)
        fcd_region_u8 = cv2.GaussianBlur(fcd_region_u8, (5, 5), 0)
        fcd_region = fcd_region_u8.astype(np.float32) / 255.0

        # Purple color (R=0.6, G=0.0, B=0.8)
        purple = np.array([0.6, 0.0, 0.8], dtype=np.float32)

        # Create the purple overlay
        fcd_3ch = np.stack([fcd_region, fcd_region, fcd_region], axis=-1)
        purple_layer = np.ones_like(bw) * purple

        # Blend: where FCD is detected, mix purple with the B&W image
        out = bw * (1 - fcd_3ch * alpha) + purple_layer * (fcd_3ch * alpha)
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
          fcd_probability, prediction, image_b64 (MRI slice with aligned Grad-CAM overlay),
          mri_b64 (original MRI slice), best_slice_info, num_patches
        """
        X, meta = preprocess_mri_for_inference(flair_path, t1_path)

        probs = self.model.predict(X, batch_size=32, verbose=0).ravel()

        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])

        # Grad-CAM on the highest-probability patch
        best_patch = X[max_idx : max_idx + 1]
        heatmap = self._gradcam_heatmap(tf.convert_to_tensor(best_patch))
        heatmap = cv2.resize(heatmap, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Get full brain slice information
        info = meta["slice_info"][max_idx]
        flair3d = meta["flair3d_normed"]          # (H, W, D), already 0-1
        full_slice = flair3d[:, :, info["z"]]     # full axial slice
        h, w = full_slice.shape
        
        # Create full-sized heatmap aligned with MRI slice
        full_heatmap = np.zeros((h, w))
        cy, cx = info["cy"], info["cx"]
        patch_half = self.img_size // 2
        
        # Calculate patch boundaries on full slice
        y_start = max(0, cy - patch_half)
        y_end = min(h, cy + patch_half)
        x_start = max(0, cx - patch_half)
        x_end = min(w, cx + patch_half)
        
        # Calculate corresponding boundaries on heatmap
        hm_y_start = max(0, patch_half - cy) if cy < patch_half else 0
        hm_y_end = hm_y_start + (y_end - y_start)
        hm_x_start = max(0, patch_half - cx) if cx < patch_half else 0
        hm_x_end = hm_x_start + (x_end - x_start)
        
        # Place heatmap at correct location on full slice
        if (hm_y_end - hm_y_start > 0) and (hm_x_end - hm_x_start > 0):
            full_heatmap[y_start:y_end, x_start:x_end] = heatmap[hm_y_start:hm_y_end, hm_x_start:hm_x_end]

        # Create brain mask for this slice
        brain_mask = self._create_brain_mask(full_slice)

        # Create original MRI slice (grayscale converted to RGB)
        full_u8 = (full_slice * 255).astype(np.uint8)
        full_rgb = cv2.cvtColor(full_u8, cv2.COLOR_GRAY2BGR)
        full_rgb = cv2.cvtColor(full_rgb, cv2.COLOR_BGR2RGB)  # Convert to RGB
        mri_b64 = self._array_to_b64_png(full_rgb.astype(np.float32) / 255.0)

        # Create FCD detection image: B&W with purple patch
        fcd_overlay = self._create_fcd_overlay(full_slice, full_heatmap,
                                               brain_mask=brain_mask,
                                               heatmap_thresh=0.40, alpha=0.55)
        fcd_b64 = self._array_to_b64_png(fcd_overlay)

        # Create MRI slice with Grad-CAM overlay (brain-only heatmap)
        overlay_rgb = self._overlay_heatmap(full_slice, full_heatmap, alpha=0.45,
                                            brain_mask=brain_mask)
        
        # Add small marker at patch center for reference
        overlay_rgb_marked = overlay_rgb.copy()
        # Ensure coordinates are integers and within bounds
        cx_int = int(cx)
        cy_int = int(cy) 
        marker_size = 3
        
        # Convert from float array (0-1) to uint8 for cv2 operations
        overlay_uint8 = (overlay_rgb_marked * 255).astype(np.uint8)
        
        # Draw a small white cross at the patch center
        cv2.line(overlay_uint8, (cx_int-marker_size, cy_int), (cx_int+marker_size, cy_int), (255, 255, 255), 2)
        cv2.line(overlay_uint8, (cx_int, cy_int-marker_size), (cx_int, cy_int+marker_size), (255, 255, 255), 2)
        
        # Convert back to float array (0-1) for encoding
        overlay_rgb_marked = overlay_uint8.astype(np.float32) / 255.0
        
        overlay_b64 = self._array_to_b64_png(overlay_rgb_marked)

        return {
            "fcd_probability": round(max_prob, 4),
            "prediction": "FCD Detected" if max_prob >= 0.5 else "No FCD Detected",
            "image_b64": overlay_b64,  # MRI slice with Grad-CAM overlay and marker
            "mri_b64": mri_b64,        # Original MRI slice
            "fcd_b64": fcd_b64,        # B&W image with purple FCD patch
            "best_slice_info": meta["slice_info"][max_idx],
            "num_patches": len(probs),
        }
