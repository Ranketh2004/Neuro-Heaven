import re
from typing import List, Tuple, Optional

import numpy as np
import mne


# ----------------------------
# Channel helpers (same logic as training)
# ----------------------------
NON_BRAIN_PATTERNS = [
    r"^EKG", r"^ECG", r"PHOTIC", r"^EMG", r"^RESP", r"PULSE", r"TRIG",
    r"^EOG", r"^TEMP", r"^MARK", r"^IBI"
]


def is_non_brain(ch: str) -> bool:
    up = str(ch).upper()
    return any(re.search(pat, up) for pat in NON_BRAIN_PATTERNS)


def norm_ref_name(ch: str) -> str:
    ch = str(ch).strip().upper()
    ch = re.sub(r"^EEG\s+", "", ch)
    ch = re.sub(r"-REF$", "", ch)
    ch = re.sub(r"-LE$", "", ch)
    ch = ch.replace(" ", "")
    return ch


def parse_bipolar(label: str) -> Tuple[str, Optional[str]]:
    s = str(label).strip().upper().replace(" ", "")
    if "-" in s:
        a, b = s.split("-", 1)
        return a, b
    return s, None


# ----------------------------
# Preprocess raw EDF (stable, training-aligned)
# ----------------------------
def preprocess_scalp(
    raw: mne.io.BaseRaw,
    notch_freqs=(60, 120),
    l_freq=0.5,
    h_freq=40.0,
    target_sfreq=250,
) -> mne.io.BaseRaw:
    if not raw.preload:
        raw.load_data()

    keep = [ch for ch in raw.ch_names if not is_non_brain(ch)]
    raw = raw.copy().pick(keep)

    raw.notch_filter(list(notch_freqs), verbose=False)
    raw.filter(l_freq, h_freq, verbose=False)

    if int(raw.info["sfreq"]) != int(target_sfreq):
        raw.resample(target_sfreq)

    return raw


# ----------------------------
# Features (9 dims per node, same as training)
# ----------------------------
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}


def hjorth_params(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x) + 1e-12
    var1 = np.var(dx) + 1e-12
    var2 = np.var(ddx) + 1e-12
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    return float(mobility), float(complexity)


def line_length(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(np.abs(np.diff(x))))


def bandpower_log(sig: np.ndarray, sfreq: float, h_freq: float = 40.0):
    sig = np.nan_to_num(np.asarray(sig, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    psd, freqs = mne.time_frequency.psd_array_welch(
        sig[np.newaxis, :], sfreq=sfreq, fmin=1, fmax=h_freq, n_fft=1024, verbose=False
    )
    psd = np.nan_to_num(psd[0], nan=0.0, posinf=0.0, neginf=0.0)
    out = []
    for (f1, f2) in BANDS.values():
        idx = (freqs >= f1) & (freqs <= f2)
        m = psd[idx].mean() if idx.any() else 0.0
        out.append(float(np.log10(m + 1e-12)))
    return out


def node_features_from_signals(S: np.ndarray, sfreq: float, h_freq: float = 40.0) -> np.ndarray:
    feats = []
    for sig in S:
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        bp = bandpower_log(sig, sfreq, h_freq=h_freq)
        v = float(np.var(sig))
        ll = line_length(sig)
        hm, hc = hjorth_params(sig)
        row = np.nan_to_num(bp + [v, ll, hm, hc], nan=0.0, posinf=0.0, neginf=0.0).tolist()
        feats.append(row)
    X = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# Edges (correlation top-k) WITH edge_attr for GATv2(edge_dim=1)
# ----------------------------
def corr_topk_edges_with_attr(S: np.ndarray, top_k: int = 8):
    """
    Returns:
      edge_index: np.ndarray shape [2, E] int64
      edge_attr : np.ndarray shape [E, 1] float32
    """
    S = np.nan_to_num(np.asarray(S, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    C = np.corrcoef(S)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)

    n = C.shape[0]
    edges, weights = [], []
    for i in range(n):
        js = np.argsort(np.abs(C[i]))[::-1][:min(top_k, n - 1)]
        for j in js:
            edges.append([i, j])
            weights.append(float(abs(C[i, j])))

    # undirected
    edges_ud = edges + [[j, i] for (i, j) in edges]
    weights_ud = weights + weights

    # self loops
    for i in range(n):
        edges_ud.append([i, i])
        weights_ud.append(1.0)

    edge_index = np.array(edges_ud, dtype=np.int64).T  # [2, E]
    edge_attr = np.array(weights_ud, dtype=np.float32).reshape(-1, 1)  # [E,1]
    return edge_index, edge_attr


# ----------------------------
# Build node signals using TEMPLATE nodes (bipolar montage list)
# ----------------------------
def build_node_signals(raw_ref: mne.io.BaseRaw, node_labels: List[str]):
    ref_names = [norm_ref_name(c) for c in raw_ref.ch_names]
    idx_ref = {ref_names[i]: i for i in range(len(ref_names))}
    data_ref = raw_ref.get_data()

    S_list, kept = [], []
    for lab in node_labels:
        a, b = parse_bipolar(lab)
        if b is not None:
            if a in idx_ref and b in idx_ref:
                sig = data_ref[idx_ref[a]] - data_ref[idx_ref[b]]
                S_list.append(sig)
                kept.append(lab)
        else:
            if a in idx_ref:
                S_list.append(data_ref[idx_ref[a]])
                kept.append(lab)

    if len(S_list) == 0:
        return None, []
    S = np.asarray(S_list, dtype=np.float32)
    return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0), kept
