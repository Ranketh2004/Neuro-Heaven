<!--
  NeuroGraphAI — Multimodal AI for Epilepsy
  README.md (HTML + GitHub-safe CSS-ish formatting)
-->

<div align="center">

  <h1>🧠 NeuroGraphAI</h1>
  <p><b>Multimodal AI System for Epilepsy Diagnosis & Treatment Response</b></p>

  <!-- Badges -->
  <p>
    <img alt="Made with Python" src="https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white">
    <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
    <img alt="PyTorch" src="https://img.shields.io/badge/Model-PyTorch-EE4C2C?logo=pytorch&logoColor=white">
    <img alt="XGBoost" src="https://img.shields.io/badge/Tabular-XGBoost-FF6600?logo=apache-spark&logoColor=white">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-000000?logo=open-source-initiative&logoColor=white">
  </p>

  <!-- Quick Nav -->
  <p>
    <a href="#overview">Overview</a> •
    <a href="#modules">Modules</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#datasets">Datasets</a> •
    <a href="#ui">Clinician UI</a> •
    <a href="#team">Team</a>
  </p>

  <!-- Tagline -->
  <blockquote>
    <i>“Where brainwaves meet intelligence — empowering neurologists with interpretable AI.”</i>
  </blockquote>

  <!-- Hero -->
  <img alt="NeuroGraphAI Header" src="https://img.shields.io/badge/-Epilepsy%20AI%20%7C%20EEG%20%7C%20MRI%20%7C%20GNN%20%7C%20XAI-673AB7?labelColor=1F1F1F&style=for-the-badge">
</div>

---

<a id="overview"></a>
<h2>📘 Overview</h2>

<b>NeuroGraphAI</b> is an end-to-end, multimodal clinical decision-support system for epilepsy.  
It combines four components into one workflow:
<ul>
  <li><b>EEG-based diagnosis</b> using deep embeddings + case-based reasoning (CBR)</li>
  <li><b>MRI lesion detection</b> via CNNs with Grad-CAM explainability</li>
  <li><b>SOZ localization</b> (Seizure Onset Zone) from non-invasive interictal EEG using GNNs</li>
  <li><b>ASM response prediction</b> (anti-seizure medication) with XGBoost + SHAP</li>
</ul>

The system outputs <b>interpretable visuals</b> (saliency/Grad-CAM, attention heatmaps, SHAP plots) and a <b>unified clinician report</b> with confidence scores.

---

<a id="modules"></a>
<h2>🧩 Modules</h2>

<table>
  <thead>
    <tr>
      <th>Module</th>
      <th>Focus</th>
      <th>Core Methods</th>
      <th>Outputs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1) EEG Diagnosis</b></td>
      <td>Epilepsy vs non-epilepsy from EEG</td>
      <td>Deep embeddings · FAISS ANN · CBR</td>
      <td>Case-matches · Confidence score · Similarity viz</td>
    </tr>
    <tr>
      <td><b>2) MRI Lesion Detection</b></td>
      <td>Hippocampal sclerosis, FCD, asymmetry</td>
      <td>CNN/UNet · Grad-CAM</td>
      <td>Lesion maps · Region report</td>
    </tr>
    <tr>
      <td><b>3) SOZ Localization</b></td>
      <td>SOZ from interictal scalp EEG</td>
      <td>Graph construction · GNN (GAT/GCN)</td>
      <td>Attention heatmaps · Electrode ranking</td>
    </tr>
    <tr>
      <td><b>4) ASM Response Prediction</b></td>
      <td>One-year seizure freedom likelihood</td>
      <td>XGBoost/TabTransformer · SHAP</td>
      <td>Probability · Feature attributions</td>
    </tr>
  </tbody>
</table>

<details>
  <summary><b>Key Features</b> (click to expand)</summary>
  <ul>
    <li>Multimodal fusion (EEG + MRI + clinical metadata)</li>
    <li>Real-time FAISS retrieval for CBR explanations</li>
    <li>Clinician-friendly interpretability (Grad-CAM, SHAP, attention)</li>
    <li>Low-resource deployability (scalable API, modular components)</li>
  </ul>
</details>

---

<a id="architecture"></a>
<a id="architecture"></a>

## 🧬 Architecture

```mermaid
flowchart TD
  A["EEG: raw / interictal"] --> B["Preprocess & Embeddings"]
  B --> C["FAISS Case Retrieval"]
  C --> D["EEG Diagnosis (CBR)"]

  E["MRI: T1 / T2 / FLAIR"] --> F["CNN-based Lesion Detector"]

  G["Clinical Metadata"] --> H["ASM Response (XGBoost / Transformer)"]

  I["Graph Builder (EEG → connectivity)"] --> J["GNN SOZ Localization"]

  D --> K["Multimodal API (FastAPI)"]
  F --> K
  J --> K
  H --> K

  K --> L["Clinician Dashboard & Reports"]
```

<a id="quickstart"></a>
<h2>⚡ Quickstart</h2> <details open> <summary><b>1) Clone & Environment</b></summary>
git clone https://github.com/<your-org>/neurographai.git
cd neurographai

# (Option A) Conda
conda create -n neurographai python=3.10 -y
conda activate neurographai

# (Option B) venv
python -m venv .venv && source .venv/bin/activate
</details> <details> <summary><b>2) Install (core + extras)</b></summary>

pip install -U pip wheel
pip install -r requirements.txt

# optional accelerators
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
</details> <details> <summary><b>3) Configure Datasets</b></summary>

# put files like:
data/
  eeg/           # EEG (e.g., TUH EEG)
  mri/           # MRI (T1/T2/FLAIR)
  clinical/      # CSV/Parquet clinical metadata
</details> <details> <summary><b>4) Run API & UI</b></summary>

# start API
uvicorn app.main:api --host 0.0.0.0 --port 8000 --reload

# start dashboard (example)
streamlit run ui/Dashboard.py
</details>

<a id="datasets"></a>
<h2>📦 Datasets (examples)</h2> <ul> <li><b>EEG:</b> TUH EEG / CHB-MIT</li> <li><b>MRI:</b> MELD, open neuroimaging cohorts (T1/T2/FLAIR)</li> <li><b>Clinical metadata:</b> curated CSV/EMR exports (de-identified)</li> </ul>
<i>Note:</i> Use only ethically sourced, de-identified data. Comply with institutional approvals.

<a id="ui"></a>
<h2>🖥️ Clinician UI (Highlights)</h2> <ul> <li><b>Case Explorer:</b> nearest-neighbor EEG matches with similarity rationale</li> <li><b>MRI Panel:</b> lesion probability maps + Grad-CAM overlay</li> <li><b>SOZ Map:</b> electrode/region attention heatmaps from GNN</li> <li><b>ASM Card:</b> seizure-freedom probability with SHAP explanations</li> <li><b>Report Export:</b> PDF with findings, confidence, and references</li> </ul>

<h2>🧠 Training & Evaluation</h2> <table> <tr> <td><b>EEG Diagnosis</b></td> <td>AUROC, F1, Precision@K (case retrieval), latency</td> </tr> <tr> <td><b>MRI Lesion</b></td> <td>Dice/IoU, sensitivity (lesion-wise), Grad-CAM sanity checks</td> </tr> <tr> <td><b>SOZ Localization</b></td> <td>Electrode-level AUC, top-K accuracy, clinical concordance</td> </tr> <tr> <td><b>ASM Response</b></td> <td>AUROC, PR-AUC, calibration (ECE), SHAP stability</td> </tr> </table>

<h2>🔐 Ethics & Privacy</h2> <ul> <li>De-identification and encryption for all PHI</li> <li>HIPAA/GDPR-aligned workflows</li> <li>Model cards and dataset statements for transparency</li> </ul>

<a id="contribute"></a>
<h2>🤝 Contributing</h2>
Fork & create a feature branch
Follow <code>black</code>/<code>ruff</code> formatting
Add tests & docs
Open a PR with a clear description and screenshots
<a id="team"></a>
<h2>🧑‍🔬 Research Team</h2> <ul> <li><b>P.A.S.R. Gunathilaka</b> — EEG Diagnosis</li> <li><b>H.V.D. Himsara</b> — MRI Lesion Detection</li> <li><b>B.G.S. Navodya</b> — ASM Response Prediction</li> <li><b>R.A.D.S. Ranaweera</b> — SOZ Localization</li> <li>Supervisors: <b>Prof. Samantha Thelijjagoda</b>, <b>Mr. Samadhi Rathnayake</b></li> </ul>


<h2>📜 Citation</h2>
If you use this project, please cite:
<pre> Gunathilaka P.A.S.R., Himsara H.V.D., Navodya B.G.S., Ranaweera R.A.D.S. (2025). NeuroGraphAI: A Multimodal AI System for Epilepsy Diagnosis & Treatment Response. Department of Computer Science, SLIIT. </pre>
<div align="center"> <sub>© 2025 NeuroGraphAI — MIT License</sub><br> <sub>“Revolutionizing epilepsy care — one neural connection at a time.” ⚡</sub> </div> 


