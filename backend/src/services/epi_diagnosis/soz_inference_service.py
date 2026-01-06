# backend/src/services/epi_diagnosis/soz_inference_service.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class SOZInferenceService:
    """
    Import-safe loader.
    Heavy deps (torch, mne, torch_geometric, joblib) are imported inside _load_all().
    """

    def __init__(self, models_dir: Path, device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device

        self.cfg = None
        self.scaler = None
        self.template_nodes = None
        self.model = None

        self._load_all()

    def _load_all(self):
        # heavy imports INSIDE (prevents import crash on module import)
        import joblib
        import torch
        import pandas as pd
        import ast

        from torch_geometric.nn import GATv2Conv
        import torch.nn as nn
        import torch.nn.functional as F

        cfg_path = self.models_dir / "config.joblib"
        scaler_path = self.models_dir / "node_scaler.joblib"
        meta_path = self.models_dir / "META_graph_windows.csv"
        state_path = self.models_dir / "GATv2_best_state_dict.pt"

        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing: {cfg_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing: {scaler_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing: {meta_path}")
        if not state_path.exists():
            raise FileNotFoundError(f"Missing: {state_path}")

        self.cfg = joblib.load(cfg_path)
        self.scaler = joblib.load(scaler_path)

        meta = pd.read_csv(meta_path)
        logger.info(f"META columns: {list(meta.columns)}")

        def parse_nodes_field(x):
            if isinstance(x, list):
                return x
            if pd.isna(x):
                return []
            s = str(x)
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
            if "|" in s:
                return [t.strip() for t in s.split("|") if t.strip()]
            return [s.strip()]

        # --- robust: detect correct column for nodes/channels ---
        candidate_cols = [
            "nodes",
            "node_names",
            "channels",
            "channel_labels",
            "bipolar_channels",
            "kept_nodes",
            "onset_channel_labels",
        ]

        nodes_col = None
        for c in candidate_cols:
            if c in meta.columns:
                nodes_col = c
                break

        if nodes_col is None:
            logger.warning(
                "META file has no known nodes column. "
                f"Available columns: {list(meta.columns)}. Using fallback montage."
            )
            self.template_nodes = [
                "FP1-F7","F7-T7","T7-P7","P7-O1",
                "FP2-F8","F8-T8","T8-P8","P8-O2",
                "FP1-F3","F3-C3","C3-P3","P3-O1",
                "FP2-F4","F4-C4","C4-P4","P4-O2",
                "FZ-CZ","CZ-PZ"
            ]
        else:
            meta["nodes_list"] = meta[nodes_col].apply(parse_nodes_field)
            meta["n_nodes2"] = meta["nodes_list"].apply(len)

            common_len = int(meta["n_nodes2"].value_counts().index[0])
            cand = meta[meta["n_nodes2"] == common_len].copy()
            cand["nodes_str"] = cand["nodes_list"].astype(str)
            self.template_nodes = parse_nodes_field(cand["nodes_str"].value_counts().index[0])

            logger.info(f"Template nodes loaded from META column='{nodes_col}' | n={len(self.template_nodes)}")

        # ---- define model inside load ----
        class GATv2Node(nn.Module):
            def __init__(self, in_dim: int, hid: int = 32, heads: int = 4, drop: float = 0.25):
                super().__init__()
                self.c1 = GATv2Conv(in_dim, hid, heads=heads, dropout=drop, bias=True)
                self.c2 = GATv2Conv(hid * heads, hid, heads=1, dropout=drop, bias=True)
                self.lin = nn.Linear(hid, 2)
                self.drop = drop

            def forward(self, data):
                x, ei = data.x, data.edge_index
                x = F.elu(self.c1(x, ei))
                x = F.dropout(x, p=self.drop, training=self.training)
                x = F.elu(self.c2(x, ei))
                return self.lin(x)

        sd = torch.load(state_path, map_location="cpu")

        att = sd.get("c1.att", None)
        if att is None:
            raise RuntimeError("Checkpoint not GATv2: missing c1.att")

        heads = int(att.shape[1])
        hid = int(att.shape[2])

        # your feature dim = 5 bandpowers + 4 stats = 9
        in_dim = 9
        drop = float(self.cfg.get("dropout", 0.25)) if isinstance(self.cfg, dict) else 0.25

        model = GATv2Node(in_dim=in_dim, hid=hid, heads=heads, drop=drop)
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.model = model

        logger.info(f"Loaded SOZ model: GATv2 hid={hid} heads={heads} drop={drop}")

    def predict_from_edf_bytes(
        self,
        edf_bytes: bytes,
        filename: str,
        tmin: float = 0.0,
        window_sec: float = 10.0
    ) -> Dict[str, Any]:
        """
        Full EDF → features → graph → GATv2 inference pipeline.
        Returns top channels ranked by SOZ probability.
        Supports both standard 10-20 montage and fallback for non-standard EEG files.
        """
        import tempfile
        import os
        import mne
        import torch
        import numpy as np
        import re
        from torch_geometric.data import Data

        from src.services.epi_diagnosis.soz_preprocessing import (
            preprocess_scalp,
            build_node_signals,
            node_features_from_signals,
            corr_topk_edges,
        )

        temp_path = None
        try:
            # 1. Write bytes to temp file and load with MNE
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                tmp.write(edf_bytes)
                temp_path = tmp.name

            raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
            logger.info(f"Loaded EDF: {filename}, channels={len(raw.ch_names)}, sfreq={raw.info['sfreq']}")

            # 2. Preprocess (notch, bandpass, resample)
            raw = preprocess_scalp(raw)
            sfreq = raw.info["sfreq"]

            # 3. Crop to window
            tmax = tmin + window_sec
            if tmax > raw.times[-1]:
                tmax = raw.times[-1]
            raw = raw.crop(tmin=tmin, tmax=tmax)

            # 4. Try template bipolar montage first
            S, kept_nodes = build_node_signals(raw, self.template_nodes)
            fallback_mode = False

            # 5. Fallback: use raw EEG channels directly if no template match
            if S is None or len(kept_nodes) == 0:
                logger.warning(f"No template match. Using fallback mode with available EEG channels.")
                fallback_mode = True
                
                # Filter to only EEG-like channels (exclude temp, activity, signal strength, etc.)
                non_eeg_patterns = [
                    r"temp", r"activity", r"signal", r"strengt", r"ecg", r"ekg", 
                    r"emg", r"eog", r"resp", r"pulse", r"photic", r"trigger", r"marker"
                ]
                
                eeg_channels = []
                for ch in raw.ch_names:
                    ch_lower = ch.lower()
                    is_non_eeg = any(re.search(pat, ch_lower) for pat in non_eeg_patterns)
                    if not is_non_eeg:
                        eeg_channels.append(ch)
                
                if len(eeg_channels) == 0:
                    return {
                        "ok": False,
                        "error": f"No EEG channels found. Available channels: {raw.ch_names}"
                    }
                
                # Use available EEG channels directly
                raw_eeg = raw.copy().pick(eeg_channels)
                S = raw_eeg.get_data()
                S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
                kept_nodes = eeg_channels
                
                logger.info(f"Fallback mode: using {len(kept_nodes)} EEG channels directly: {kept_nodes}")

            logger.info(f"Built {len(kept_nodes)} node signals (fallback={fallback_mode})")

            # 6. Extract features (9 features per node)
            X = node_features_from_signals(S, sfreq, h_freq=40.0)

            # 7. Scale features
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # 8. Build graph edges
            edge_index, edge_weight = corr_topk_edges(S, top_k=min(8, len(kept_nodes) - 1))
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.float32)

            # 9. Create PyG Data object
            data = Data(
                x=X_tensor,
                edge_index=edge_index_tensor,
                edge_attr=edge_weight_tensor
            )

            # 10. Run inference
            self.model.eval()
            with torch.no_grad():
                logits = self.model(data)  # shape: [num_nodes, 2]
                probs = torch.softmax(logits, dim=1)[:, 1]  # prob of class 1 (SOZ)

            probs_np = probs.cpu().numpy()

            # 11. Rank channels by SOZ probability
            ranked_idx = np.argsort(probs_np)[::-1]
            top_channels = []
            for idx in ranked_idx:
                top_channels.append({
                    "channel": kept_nodes[idx],
                    "soz_probability": float(probs_np[idx]),
                    "prediction": int(probs_np[idx] > 0.5)
                })

            logger.info(f"SOZ inference complete. Top channel: {top_channels[0] if top_channels else 'none'}")

            return {
                "ok": True,
                "filename": filename,
                "template_nodes_count": len(self.template_nodes),
                "matched_nodes_count": len(kept_nodes),
                "fallback_mode": fallback_mode,
                "top_channels": top_channels,
                "topomap_png_base64": None
            }

        except Exception as e:
            logger.error(f"SOZ inference error: {e}", exc_info=True)
            return {
                "ok": False,
                "error": str(e)
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
