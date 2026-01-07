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

            # 12. Generate brain topomap with SOZ likelihood coloring
            topomap_b64 = self._generate_soz_topomap(kept_nodes, probs_np, fallback_mode)

            return {
                "ok": True,
                "filename": filename,
                "template_nodes_count": len(self.template_nodes),
                "matched_nodes_count": len(kept_nodes),
                "fallback_mode": fallback_mode,
                "top_channels": top_channels,
                "topomap_png_base64": topomap_b64
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

    def _generate_soz_topomap(self, channels: list, probabilities, fallback_mode: bool) -> str:
        """
        Generate a brain topomap image with SOZ likelihood coloring.
        Red = high likelihood (>0.75), Yellow = medium (0.5-0.75), Green = low (<0.5)
        Returns base64 encoded PNG string.
        """
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from io import BytesIO
        import base64

        try:
            # Standard 10-20 electrode positions (normalized coordinates)
            # Format: channel_name -> (x, y) where center is (0, 0)
            STANDARD_10_20_POS = {
                # Frontal
                "FP1": (-0.31, 0.95), "FP2": (0.31, 0.95), "FPZ": (0.0, 0.95),
                "F7": (-0.81, 0.59), "F3": (-0.39, 0.59), "FZ": (0.0, 0.59),
                "F4": (0.39, 0.59), "F8": (0.81, 0.59),
                # Temporal
                "T7": (-1.0, 0.0), "T3": (-1.0, 0.0),  # T7 = T3 in older naming
                "T8": (1.0, 0.0), "T4": (1.0, 0.0),    # T8 = T4 in older naming
                "T5": (-0.81, -0.59), "P7": (-0.81, -0.59),  # T5 = P7
                "T6": (0.81, -0.59), "P8": (0.81, -0.59),    # T6 = P8
                # Central
                "C3": (-0.5, 0.0), "CZ": (0.0, 0.0), "C4": (0.5, 0.0),
                # Parietal
                "P3": (-0.39, -0.59), "PZ": (0.0, -0.59), "P4": (0.39, -0.59),
                # Occipital
                "O1": (-0.31, -0.95), "O2": (0.31, -0.95), "OZ": (0.0, -0.95),
            }

            # For bipolar channels, use midpoint between two electrodes
            def get_bipolar_pos(label):
                parts = label.upper().replace(" ", "").split("-")
                if len(parts) == 2:
                    a, b = parts[0], parts[1]
                    if a in STANDARD_10_20_POS and b in STANDARD_10_20_POS:
                        ax, ay = STANDARD_10_20_POS[a]
                        bx, by = STANDARD_10_20_POS[b]
                        return ((ax + bx) / 2, (ay + by) / 2)
                # Try single electrode
                if parts[0] in STANDARD_10_20_POS:
                    return STANDARD_10_20_POS[parts[0]]
                return None

            # Extract channel name from various formats
            def normalize_channel_name(ch):
                ch = str(ch).upper().strip()
                # Remove common prefixes
                for prefix in ["EEG ", "EEG_", "EEG"]:
                    if ch.startswith(prefix):
                        ch = ch[len(prefix):]
                # Remove suffixes like -REF, -LE
                for suffix in ["-REF", "-LE", " REF", " LE"]:
                    if ch.endswith(suffix):
                        ch = ch[:-len(suffix)]
                return ch.strip()

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
            ax.set_aspect('equal')
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.axis('off')

            # Draw head outline
            head_circle = plt.Circle((0, 0), 1.15, fill=False, color='#2C3E50', linewidth=3)
            ax.add_patch(head_circle)

            # Draw nose
            nose_x = [0, -0.08, 0.08, 0]
            nose_y = [1.15, 1.28, 1.28, 1.15]
            ax.fill(nose_x, nose_y, color='#2C3E50')

            # Draw ears
            ear_left = plt.Circle((-1.2, 0), 0.08, fill=True, color='#2C3E50')
            ear_right = plt.Circle((1.2, 0), 0.08, fill=True, color='#2C3E50')
            ax.add_patch(ear_left)
            ax.add_patch(ear_right)

            # Custom colormap: Green -> Yellow -> Red
            colors_list = ['#22C55E', '#FACC15', '#EF4444']  # green, yellow, red
            cmap = LinearSegmentedColormap.from_list('soz_cmap', colors_list, N=256)

            # Plot channels with known positions
            plotted_positions = []
            plotted_probs = []
            plotted_labels = []

            for i, ch in enumerate(channels):
                prob = float(probabilities[i])
                ch_norm = normalize_channel_name(ch)
                
                # Try to get position
                pos = None
                if ch_norm in STANDARD_10_20_POS:
                    pos = STANDARD_10_20_POS[ch_norm]
                else:
                    pos = get_bipolar_pos(ch_norm)
                
                if pos is not None:
                    plotted_positions.append(pos)
                    plotted_probs.append(prob)
                    plotted_labels.append(ch)

            # If no standard positions found (fallback mode), arrange in a circle
            if len(plotted_positions) == 0 and len(channels) > 0:
                n_ch = len(channels)
                for i, ch in enumerate(channels):
                    prob = float(probabilities[i])
                    angle = (2 * np.pi * i / n_ch) - np.pi / 2  # Start from top
                    x = 0.7 * np.cos(angle)
                    y = 0.7 * np.sin(angle)
                    plotted_positions.append((x, y))
                    plotted_probs.append(prob)
                    plotted_labels.append(ch)

            # Plot electrodes
            for pos, prob, label in zip(plotted_positions, plotted_probs, plotted_labels):
                x, y = pos
                
                # Color based on probability
                color = cmap(prob)
                
                # Size based on probability (larger = higher likelihood)
                size = 800 + prob * 1200
                
                # Plot electrode
                ax.scatter(x, y, s=size, c=[color], edgecolors='#1E293B', linewidths=2, zorder=3)
                
                # Add label
                ax.annotate(
                    label, (x, y),
                    ha='center', va='center',
                    fontsize=8, fontweight='bold', color='#1E293B',
                    zorder=4
                )
                
                # Add probability percentage below
                ax.annotate(
                    f'{int(prob * 100)}%', (x, y - 0.12),
                    ha='center', va='top',
                    fontsize=7, color='#475569',
                    zorder=4
                )

            # Add title
            mode_text = "(Fallback Mode)" if fallback_mode else ""
            ax.set_title(f'SOZ Likelihood Brain Map {mode_text}', fontsize=14, fontweight='bold', color='#1E293B', pad=20)

            # Add legend
            legend_elements = [
                plt.scatter([], [], s=200, c='#EF4444', edgecolors='#1E293B', linewidths=1, label='High Risk (>75%)'),
                plt.scatter([], [], s=150, c='#FACC15', edgecolors='#1E293B', linewidths=1, label='Moderate (50-75%)'),
                plt.scatter([], [], s=100, c='#22C55E', edgecolors='#1E293B', linewidths=1, label='Low Risk (<50%)'),
            ]
            ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
                     ncol=3, frameon=False, fontsize=9)

            # Save to bytes
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close(fig)

            # Encode to base64
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            logger.info(f"Generated SOZ topomap with {len(plotted_positions)} electrodes")
            return img_b64

        except Exception as e:
            logger.error(f"Failed to generate topomap: {e}", exc_info=True)
            return None
