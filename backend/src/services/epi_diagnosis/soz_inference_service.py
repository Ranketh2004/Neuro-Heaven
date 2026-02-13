from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import tempfile
import os
import ast
import time

logger = logging.getLogger(__name__)


class SOZInferenceService:
    """
    EDF bytes -> preprocess -> graph -> GraphSAGE -> node probabilities
    """

    def __init__(self, models_dir: Path, device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device

        self.cfg: Dict[str, Any] = {}
        self.scaler = None
        self.template_nodes: List[str] = []
        self.model = None
        self.best_thr: float = 0.5

        self._load_all()

    def _load_all(self):
        import joblib
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv

        cfg_path = self.models_dir / "config.joblib"
        scaler_path = self.models_dir / "node_scaler.joblib"
        meta_path = self.models_dir / "META_graph_windows.csv"
        state_path = self.models_dir / "GraphSAGE_best_state_dict.pt"

        for p in (cfg_path, scaler_path, meta_path, state_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing required artifact: {p}")

        self.cfg = joblib.load(cfg_path) or {}
        self.scaler = joblib.load(scaler_path)
        self.best_thr = float(self.cfg.get("best_thr", 0.5))

        # -------- Template nodes from META --------
        meta = pd.read_csv(meta_path)

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

        if "nodes" not in meta.columns:
            raise RuntimeError(
                f"META_graph_windows.csv must contain a 'nodes' column. Found: {list(meta.columns)}"
            )

        meta["nodes_list"] = meta["nodes"].apply(parse_nodes_field)
        meta["n_nodes2"] = meta["nodes_list"].apply(len)

        common_len = int(meta["n_nodes2"].value_counts().index[0])
        cand = meta[meta["n_nodes2"] == common_len].copy()
        template = cand["nodes_list"].astype(str).value_counts().index[0]
        self.template_nodes = parse_nodes_field(template)

        logger.info(f"[SOZ] Template nodes loaded: n={len(self.template_nodes)}")

        # -------- GraphSAGE model definition --------
        class SAGENode(nn.Module):
            def __init__(self, in_dim: int, hid: int = 64, drop: float = 0.25):
                super().__init__()
                self.c1 = SAGEConv(in_dim, hid)
                self.c2 = SAGEConv(hid, hid)
                self.lin = nn.Linear(hid, 2)
                self.drop = drop

            def forward(self, data):
                x, ei = data.x, data.edge_index
                x = F.relu(self.c1(x, ei))
                x = F.dropout(x, p=self.drop, training=self.training)
                x = F.relu(self.c2(x, ei))
                return self.lin(x)

        # infer dims from config (fallbacks)
        # NOTE: your training features are 9 dims
        in_dim = int(self.cfg.get("in_dim", 9))
        hid = int(self.cfg.get("hid", 64))
        drop = float(self.cfg.get("dropout", 0.25))

        model = SAGENode(in_dim=in_dim, hid=hid, drop=drop)

        import torch
        sd = torch.load(state_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.model = model.to(self.device)

        logger.info(f"[SOZ] Loaded GraphSAGE on {self.device}: in_dim={in_dim}, hid={hid}, drop={drop}")

    def predict_from_edf_bytes(
        self,
        edf_bytes: bytes,
        filename: str,
        tmin: float = 0.0,
        window_sec: float = 10.0,
        top_k_return: int = 15,
        debug: bool = True,   # ✅ set False if you want a cleaner response
    ) -> Dict[str, Any]:
        import numpy as np
        import mne
        import torch
        from torch_geometric.data import Data

        from src.services.epi_diagnosis.soz_preprocessing import (
            preprocess_scalp,
            build_node_signals,
            node_features_from_signals,
            corr_topk_edges_with_attr,  # returns (edge_index, edge_attr)
        )

        temp_path: Optional[str] = None
        steps = {
            "write_temp": False,
            "load_edf": False,
            "preprocess": False,
            "crop": False,
            "build_nodes": False,
            "features": False,
            "scale": False,
            "edges": False,
            "inference": False,
            "rank": False,
        }
        timings_ms: Dict[str, float] = {}

        def _tick():
            return time.perf_counter()

        try:
            t0 = _tick()
            logger.info(f"[SOZ] START predict | file={filename} | tmin={tmin} | window={window_sec}s")

            # 1) Write EDF bytes -> temp file
            s = _tick()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                tmp.write(edf_bytes)
                temp_path = tmp.name
            steps["write_temp"] = True
            timings_ms["write_temp"] = (_tick() - s) * 1000.0
            logger.info(f"[SOZ] STEP write_temp OK | path={temp_path}")

            # 2) Load EDF
            s = _tick()
            raw = mne.io.read_raw_edf(temp_path, preload=False, verbose=False)
            steps["load_edf"] = True
            timings_ms["load_edf"] = (_tick() - s) * 1000.0
            logger.info(f"[SOZ] STEP load_edf OK | channels={len(raw.ch_names)} | sfreq={raw.info['sfreq']}")

            # 3) Preprocess
            s = _tick()
            notch = self.cfg.get("notch_freqs", [60, 120])
            l_freq = float(self.cfg.get("l_freq", 0.5))
            h_freq = float(self.cfg.get("h_freq", 40.0))
            target_sfreq = int(self.cfg.get("target_sfreq", 250))

            raw = preprocess_scalp(
                raw,
                notch_freqs=tuple(notch),
                l_freq=l_freq,
                h_freq=h_freq,
                target_sfreq=target_sfreq,
            )
            steps["preprocess"] = True
            timings_ms["preprocess"] = (_tick() - s) * 1000.0
            logger.info(f"[SOZ] STEP preprocess OK | sfreq={raw.info['sfreq']} | kept_channels={len(raw.ch_names)}")

            # 4) Crop window
            s = _tick()
            tmax = min(raw.times[-1], tmin + window_sec)
            if tmax - tmin < 1.0:
                return {"ok": False, "error": "Requested window is too short for inference."}

            seg = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
            steps["crop"] = True
            timings_ms["crop"] = (_tick() - s) * 1000.0
            logger.info(f"[SOZ] STEP crop OK | dur={(tmax - tmin):.2f}s")

            # 5) Build node signals aligned to training template
            s = _tick()
            S, kept_nodes = build_node_signals(seg, self.template_nodes)
            steps["build_nodes"] = True
            timings_ms["build_nodes"] = (_tick() - s) * 1000.0
            logger.info(f"[SOZ] STEP build_nodes OK | matched_nodes={len(kept_nodes)}")

            if S is None or len(kept_nodes) < int(self.cfg.get("min_nodes", 8)):
                return {
                    "ok": False,
                    "error": f"Could not build enough bipolar nodes from EDF. matched={len(kept_nodes)}",
                    "matched_nodes_count": len(kept_nodes),
                    "pipeline_steps": steps if debug else None,
                }

            sfreq = float(seg.info["sfreq"])

            # 6) Node features + scale
            s = _tick()
            X = node_features_from_signals(S, sfreq, h_freq=h_freq)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            steps["features"] = True
            timings_ms["features"] = (_tick() - s) * 1000.0
            logger.info(f"[SOZ] STEP features OK | X_shape={X.shape}")

            s = _tick()
            Xs = self.scaler.transform(X)
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            steps["scale"] = True
            timings_ms["scale"] = (_tick() - s) * 1000.0
            logger.info(f"[SOZ] STEP scale OK")

            # 7) Graph edges (GraphSAGE uses only edge_index)
            s = _tick()
            topk_edges = int(self.cfg.get("topk_edges", 8))
            edge_index_t, _edge_attr_t = corr_topk_edges_with_attr(S, top_k=topk_edges)

            # IMPORTANT: corr_topk_edges_with_attr already returns torch tensors in your training code style,
            # but your backend version might return numpy arrays.
            # So handle both:
            if hasattr(edge_index_t, "detach"):
                edge_index = edge_index_t
            else:
                edge_index = torch.tensor(edge_index_t, dtype=torch.long)

            steps["edges"] = True
            timings_ms["edges"] = (_tick() - s) * 1000.0
            logger.info(f"[SOZ] STEP edges OK | edge_index_shape={tuple(edge_index.shape)}")

            # 8) Build Data
            data = Data(
                x=torch.tensor(Xs, dtype=torch.float32, device=self.device),
                edge_index=edge_index.to(self.device),
            )

            # 9) Inference
            s = _tick()
            self.model.eval()
            with torch.no_grad():
                logits = self.model(data)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            steps["inference"] = True
            timings_ms["inference"] = (_tick() - s) * 1000.0
            logger.info(
                f"[SOZ] STEP inference OK | probs_min={float(np.min(probs)):.3f} probs_max={float(np.max(probs)):.3f}"
            )

            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

            # 10) Rank channels
            s = _tick()
            order = np.argsort(probs)[::-1]
            top_channels = []
            for idx in order[: min(top_k_return, len(order))]:
                p = float(probs[idx])
                top_channels.append(
                    {
                        "channel": kept_nodes[idx],
                        "soc_probability": p,
                        "above_threshold": bool(p >= self.best_thr),
                    }
                )
            steps["rank"] = True
            timings_ms["rank"] = (_tick() - s) * 1000.0

            total_ms = (_tick() - t0) * 1000.0
            logger.info(f"[SOZ] DONE predict | total_ms={total_ms:.1f} | top={top_channels[0] if top_channels else None}")

            resp = {
                "ok": True,
                "filename": filename,
                "tmin": float(tmin),
                "window_sec": float(window_sec),
                "threshold": float(self.best_thr),
                "template_nodes_count": len(self.template_nodes),
                "matched_nodes_count": len(kept_nodes),
                "top_channels": top_channels,
            }

            if debug:
                resp["pipeline_steps"] = steps
                resp["timings_ms"] = {k: round(v, 2) for k, v in timings_ms.items()}
                resp["debug"] = {
                    "raw_sfreq_after_preprocess": float(raw.info["sfreq"]),
                    "feature_shape": [int(X.shape[0]), int(X.shape[1])],
                    "topk_edges": int(topk_edges),
                }

            return resp

        except Exception as e:
            logger.exception("[SOZ] SOZ inference failed")
            return {
                "ok": False,
                "error": str(e),
                "pipeline_steps": steps if debug else None,
                "timings_ms": {k: round(v, 2) for k, v in timings_ms.items()} if debug else None,
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
