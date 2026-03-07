# backend/src/services/epi_diagnosis/mri_inference_service.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .mri_preprocessing_service import (
    preprocess_mri_to_slices,
    reconstruct_3d_mask,
    save_mask_nifti,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Model definition (must match training)
# -----------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class UNetLite(nn.Module):
    def __init__(self, in_ch=2, base=16):
        super().__init__()
        self.e1 = conv_block(in_ch, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.mid = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = conv_block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = conv_block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


# -----------------------------
# Service class: load once, reuse
# -----------------------------
class MRIFCDInferenceService:
    def __init__(self, model_pt_path: str, img_size: int = 192, base_ch: int = 16):
        self.model_pt_path = model_pt_path
        self.img_size = img_size
        self.base_ch = base_ch

        self.model = UNetLite(in_ch=2, base=self.base_ch).to(DEVICE)
        self._load_weights()
        self.model.eval()

    def _load_weights(self):
        if not os.path.exists(self.model_pt_path):
            raise FileNotFoundError(f"Model file not found: {self.model_pt_path}")

        ckpt = torch.load(self.model_pt_path, map_location=DEVICE)

        # Your saved file might be:
        # 1) {"model_state_dict": ...}  (recommended)
        # or 2) {"model": ...}          (from trainer checkpoint)
        # or 3) state_dict directly
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict):
            # might already be a state_dict
            sd = ckpt
        else:
            raise ValueError("Unknown checkpoint format")

        self.model.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def predict(
        self,
        flair_path: str,
        t1_path: str | None,
        thr: float = 0.5,
        max_slices: int | None = None,
        out_mask_path: str | None = None,
    ):
        """
        Returns:
          dict with mask_path + basic stats
        """
        X, meta = preprocess_mri_to_slices(
            flair_path=flair_path,
            t1_path=t1_path,
            img_size=self.img_size,
            max_slices=max_slices,
        )

        # X: (S,2,H,W)
        xb = torch.from_numpy(X).to(DEVICE)  # float32
        logits = self.model(xb)              # (S,1,H,W)
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # (S,H,W)

        mask_3d = reconstruct_3d_mask(probs, meta, thr=thr)

        # stats
        pos_vox = int((mask_3d > 0).sum())
        total_vox = int(mask_3d.size)

        saved_path = None
        if out_mask_path is not None:
            saved_path = save_mask_nifti(mask_3d, meta, out_mask_path)

        return {
            "mask_path": saved_path,
            "positive_voxels": pos_vox,
            "total_voxels": total_vox,
            "positive_ratio": float(pos_vox / max(total_vox, 1)),
            "threshold": float(thr),
            "used_slices": int(len(meta["z_indices"])),
        }
