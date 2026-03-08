# backend/src/services/mri_preprocessing_service.py
# Preprocessing pipeline matching the FCD patch-based training notebook.

import os
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import zoom

IMG_SIZE = 224
PATCH_SIZE = 96


def resolve_nii_path(p: str) -> str:
    """If p is file => return. If directory => return first .nii/.nii.gz inside."""
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        for fn in sorted(os.listdir(p)):
            if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                inner = os.path.join(p, fn)
                if os.path.isfile(inner):
                    return inner
    raise FileNotFoundError(f"Could not resolve nii path: {p}")


def load_nii(p: str):
    p = resolve_nii_path(p)
    img = nib.load(p)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine


def norm01(x: np.ndarray) -> np.ndarray:
    """Percentile-based [0,1] normalization (matches training notebook)."""
    x = np.nan_to_num(x)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return x.astype(np.float32)


def crop_patch(img2d: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    """Crop a square patch centred at (cy, cx) with zero-padding if needed."""
    h, w = img2d.shape
    half = size // 2
    y1 = max(0, cy - half); y2 = min(h, cy + half)
    x1 = max(0, cx - half); x2 = min(w, cx + half)
    patch = img2d[y1:y2, x1:x2]
    pad_y1 = max(0, half - cy)
    pad_x1 = max(0, half - cx)
    pad_y2 = max(0, (cy + half) - h)
    pad_x2 = max(0, (cx + half) - w)
    patch = np.pad(patch, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode="constant")
    return patch


def resize_to_model(x2d: np.ndarray) -> np.ndarray:
    return cv2.resize(x2d, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def make_2ch(flair2d: np.ndarray, t1w2d: np.ndarray) -> np.ndarray:
    """Stack FLAIR + T1w into a 2-channel image (H, W, 2)."""
    return np.stack([flair2d, t1w2d], axis=-1).astype(np.float32)


# ------------------------------------------------------------------
# Main inference preprocessing
# ------------------------------------------------------------------
def preprocess_mri_for_inference(
    flair_path: str,
    t1_path: str | None = None,
    patch_size: int = PATCH_SIZE,
    img_size: int = IMG_SIZE,
    num_slices: int = 16,
    patches_per_slice: int = 5,
):
    """
    Extract 2-channel (FLAIR+T1w) patches from axial slices.

    Returns
    -------
    X : np.ndarray  (N, 224, 224, 2)  – model-ready patches
    meta : dict     – bookkeeping needed for Grad-CAM overlay
    """
    flair3d, affine = load_nii(flair_path)

    if t1_path is not None:
        t1w3d, _ = load_nii(t1_path)
        if t1w3d.shape != flair3d.shape:
            factors = tuple(f / t for f, t in zip(flair3d.shape, t1w3d.shape))
            t1w3d = zoom(t1w3d, factors, order=1)
    else:
        t1w3d = flair3d.copy()

    flair3d = norm01(flair3d)
    t1w3d = norm01(t1w3d)

    h, w, d = flair3d.shape

    # Sample slices from the informative middle portion of the volume
    start = max(0, d // 4)
    end = min(d, 3 * d // 4)
    if end <= start:
        start, end = 0, d
    slice_indices = np.linspace(start, end - 1, min(num_slices, end - start), dtype=int)

    patches = []
    slice_info = []
    half = patch_size // 2

    for z in slice_indices:
        flair_slice = flair3d[:, :, z]
        t1w_slice = t1w3d[:, :, z]

        # Centre patch
        cy, cx = h // 2, w // 2
        _append_patch(patches, slice_info, flair_slice, t1w_slice,
                      cy, cx, patch_size, img_size, int(z), "center")

        # Quadrant patches (avoid going out-of-bounds for tiny volumes)
        offsets = [
            (h // 3, w // 3),
            (h // 3, 2 * w // 3),
            (2 * h // 3, w // 3),
            (2 * h // 3, 2 * w // 3),
        ]
        for oy, ox in offsets:
            if half <= oy < h - half and half <= ox < w - half:
                _append_patch(patches, slice_info, flair_slice, t1w_slice,
                              oy, ox, patch_size, img_size, int(z), "offset")

    X = np.stack(patches, axis=0).astype(np.float32)

    meta = {
        "affine": affine,
        "flair_shape": flair3d.shape,
        "slice_info": slice_info,
        "flair3d_normed": flair3d,
        "t1w3d_normed": t1w3d,
    }
    return X, meta


def _append_patch(patches, slice_info, flair_slice, t1w_slice,
                  cy, cx, patch_size, img_size, z, tag):
    f_patch = crop_patch(flair_slice, cy, cx, patch_size)
    t_patch = crop_patch(t1w_slice, cy, cx, patch_size)
    f_in = resize_to_model(f_patch)
    t_in = resize_to_model(t_patch)
    patches.append(make_2ch(f_in, t_in))
    slice_info.append({"z": z, "cy": cy, "cx": cx, "type": tag})
