# backend/src/services/epi_diagnosis/mri_preprocessing_service.py

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, binary_fill_holes
from skimage.transform import resize


# -----------------------------
# Core helpers (match your notebook)
# -----------------------------
def resolve_nii_path(p: str) -> str:
    """If p is file => return. If directory => return first .nii/.nii.gz inside."""
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        for fn in os.listdir(p):
            if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                inner = os.path.join(p, fn)
                if os.path.isfile(inner):
                    return inner
    raise FileNotFoundError(f"Could not resolve nii path: {p}")


def load_nii(p: str) -> np.ndarray:
    p = resolve_nii_path(p)
    img = nib.load(p)
    return img.get_fdata(dtype=np.float32)


def load_nii_with_affine(p: str):
    p = resolve_nii_path(p)
    img = nib.load(p)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def resample_to_ref(mov: np.ndarray, ref: np.ndarray, order=1) -> np.ndarray:
    """
    Shape-based resampling (same as your training code).
    NOTE: Not spacing-aware, but consistent with training.
    """
    zoom_factors = (
        ref.shape[0] / mov.shape[0],
        ref.shape[1] / mov.shape[1],
        ref.shape[2] / mov.shape[2],
    )
    return zoom(mov, zoom_factors, order=order)


def brain_mask_simple(vol: np.ndarray) -> np.ndarray:
    thr = float(vol.mean() + 0.2 * vol.std())
    m = (vol > thr).astype(np.uint8)
    for z in range(m.shape[2]):
        m[:, :, z] = binary_fill_holes(m[:, :, z]).astype(np.uint8)
    return m


def robust_zscore(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = vol[mask > 0]
    if vals.size < 50:
        return vol.astype(np.float32)
    med = float(np.median(vals))
    sd = float(np.std(vals) + 1e-6)
    out = (vol - med) / sd
    out *= mask
    return out.astype(np.float32)


def resize_2d(img2d: np.ndarray, size: int) -> np.ndarray:
    return resize(
        img2d, (size, size),
        order=1, mode="constant",
        preserve_range=True, anti_aliasing=True
    ).astype(np.float32)


def resize_mask_2d(msk2d: np.ndarray, size: int) -> np.ndarray:
    return resize(
        msk2d, (size, size),
        order=0, mode="constant",
        preserve_range=True, anti_aliasing=False
    ).astype(np.float32)


# -----------------------------
# Preprocessing pipeline for inference
# -----------------------------
def preprocess_mri_to_slices(
    flair_path: str,
    t1_path: str | None,
    img_size: int = 192,
    max_slices: int | None = None,
):
    """
    Returns:
      X: np.ndarray float32 shape (S, 2, img_size, img_size)  [FLAIR, T1]
      meta: dict with info to reconstruct mask to original FLAIR space
    """
    flair, affine, header = load_nii_with_affine(flair_path)

    if t1_path is None:
        # If user uploads only FLAIR, we duplicate it (works, but less accurate).
        t1r = flair.copy()
    else:
        t1 = load_nii(t1_path)
        t1r = resample_to_ref(t1, flair, order=1)

    bm = brain_mask_simple(flair)
    flair_n = robust_zscore(flair, bm)
    t1_n = robust_zscore(t1r, bm)

    Z = flair.shape[2]
    z_indices = list(range(Z))

    if max_slices is not None and Z > max_slices:
        step = Z / max_slices
        z_indices = [int(i * step) for i in range(max_slices)]

    Xs = []
    for zi in z_indices:
        f2 = resize_2d(flair_n[:, :, zi], img_size)
        t2 = resize_2d(t1_n[:, :, zi], img_size)
        Xs.append(np.stack([f2, t2], axis=0).astype(np.float32))

    X = np.stack(Xs, axis=0).astype(np.float32)

    meta = {
        "affine": affine,
        "header": header,
        "flair_shape": flair.shape,      # (H,W,Z)
        "z_indices": z_indices,          # which slices we actually used
        "img_size": img_size,
    }
    return X, meta


def reconstruct_3d_mask(pred_slices: np.ndarray, meta: dict, thr: float = 0.5) -> np.ndarray:
    """
    pred_slices: (S, img_size, img_size) float prob [0..1]
    returns: mask_3d in original FLAIR shape (H,W,Z) float32 {0,1}
    """
    H, W, Z = meta["flair_shape"]
    img_size = meta["img_size"]
    z_indices = meta["z_indices"]

    mask_3d = np.zeros((H, W, Z), dtype=np.float32)

    for s_idx, zi in enumerate(z_indices):
        prob2d = pred_slices[s_idx]
        bin2d = (prob2d > thr).astype(np.float32)
        # back to (H,W)
        up = resize_mask_2d(bin2d, H)  # gives (H,H) if H!=W this is imperfect
        # If your MRI is not square, do proper (H,W) resize:
        up = resize(
            bin2d, (H, W),
            order=0, mode="constant",
            preserve_range=True, anti_aliasing=False
        ).astype(np.float32)

        mask_3d[:, :, zi] = up

    return mask_3d


def save_mask_nifti(mask_3d: np.ndarray, meta: dict, out_path: str):
    img = nib.Nifti1Image(mask_3d.astype(np.float32), meta["affine"], header=meta["header"])
    nib.save(img, out_path)
    return out_path
