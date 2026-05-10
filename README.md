# ZAN-Multimodal-Neuroimaging

# ZAN Multimodal Neuroimaging
 
> **Functional and electrophysiological biomarkers for ZAN classification using resting-state fMRI, EEG, and multimodal fusion.**
 
This repository contains three independent machine learning pipelines applied to the same dataset of 51 subjects (27 Healthy Controls, 24 ZAN), progressively building toward a multimodal fusion model. Each notebook is self-contained and reproducible on Google Colab.
 
---
 
## Repository Structure
 
```
ZAN-Multimodal-Neuroimaging/
│
├── 01_fMRI/
│   └── ZAN_fMRI_Classification.ipynb       ✅ Complete
│
├── 02_EEG/
│   └── ZAN_EEG_Classification.ipynb        🔄 In Progress
│
├── 03_Fusion/
│   └── ZAN_fMRI_EEG_Fusion.ipynb           🔄 In Progress
│
└── README.md
```
 
---
 
## Dataset
 
| Property | Value |
|---|---|
| Total subjects | 51 (after QC) |
| Healthy Controls (HC) | 27 |
| ZAN patients | 24 |
| Age HC | 61.1 ± 4.6 years |
| Age ZAN | 61.7 ± 6.7 years |
| Sex HC | 13M / 16F |
| Sex ZAN | 13M / 12F |
| Age group match | t = −0.399, p = 0.691 ✅ |
| Sex group match | χ² = 0.064, p = 0.800 ✅ |
 
**Data is not included in this repository.** Expected directory structure on Google Drive:
 
```
MyDrive/
├── ZAN_Research/
│   ├── participants.tsv          ← subject metadata (participant_id, group, age, sex)
│   └── features_v2/              ← auto-created by notebooks
│
└── Fmri/
    └── Func/
        ├── sub-01_task-rest_bold.nii
        ├── sub-02_task-rest_bold.nii
        └── ...
```
 
---
 
## Notebook 1 — fMRI Functional Connectivity
 
**File:** `01_fMRI/ZAN_fMRI_Classification.ipynb`
 
### Pipeline
 
```
Raw fMRI (.nii)
    ↓
AAL3v2 Atlas (166 ROIs) — NiftiLabelsMasker
Bandpass: 0.01–0.1 Hz | TR: 2.0s | z-score | detrend
    ↓
Timeseries extraction (240 timepoints × 166 ROIs)
    ↓
Tangent Space Functional Connectivity
    ↓
Full brain: 166×166 → 13,695 edges
Targeted:    60×60  →  1,770 edges
(Thalamus · Frontal · Temporal · Parietal · Insula · Cingulum)
    ↓
10-Fold CV Benchmark          LOOCV (Primary)
SVM · MLP · Transformer       LASSO · Ridge · Linear SVM
    ↓
XAI Biomarker Extraction (LASSO weights)
    ↓
Verification: Permutation · Bootstrap · Calibration · Clinical Stats
```
 
### Results
 
**10-Fold Cross-Validation**
 
| Model | Accuracy | AUC | F1 |
|---|---|---|---|
| SVM-RBF (166 ROIs, Top 300) | 0.613 ± 0.197 | 0.556 ± 0.251 | 0.513 ± 0.251 |
| MLP (166 ROIs, Top 300) | 0.573 ± 0.233 | 0.606 ± 0.252 | 0.533 ± 0.273 |
| BrainNetTransformer (166×166) | 0.533 ± 0.242 | 0.572 ± 0.337 | 0.417 ± 0.309 |
| Ensemble SVM+Trans (166 ROIs) | 0.557 ± 0.298 | 0.589 ± 0.312 | 0.467 ± 0.355 |
| SVM-RBF (60 ROIs, Top 300) | 0.633 ± 0.205 | 0.628 ± 0.249 | 0.548 ± 0.316 |
| MLP (60 ROIs, Top 300) | 0.570 ± 0.190 | 0.611 ± 0.262 | 0.542 ± 0.250 |
| BrainNetTransformer (60×60) | 0.573 ± 0.250 | 0.528 ± 0.362 | 0.433 ± 0.389 |
| Ensemble SVM+Trans (60 ROIs) | 0.613 ± 0.197 | 0.583 ± 0.318 | 0.513 ± 0.322 |
 
**LOOCV — Primary Results**
 
| Model | Accuracy | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Linear SVM (Top 30) | 0.667 | 0.732 | 0.653 | 0.667 | 0.667 |
| Ridge / L2 (Top 30) | 0.725 | 0.804 | 0.708 | 0.708 | 0.741 |
| **★ LASSO / L1 (Top 30)** | **0.824** | **0.900** | **0.808** | **0.792** | **0.852** |
 
**LASSO LOOCV — 95% Bootstrap Confidence Intervals (n=2000)**
 
| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.824 | 0.725 – 0.922 |
| AUC | 0.900 | 0.801 – 0.979 |
| F1-Score | 0.808 | 0.667 – 0.917 |
| Sensitivity | 0.792 | 0.619 – 0.950 |
| Specificity | 0.852 | 0.708 – 0.966 |
 
Permutation test (100 shuffles, LOOCV): **p < 0.05** — result is significantly above chance.
 
### Top Biomarkers (LASSO, 21 surviving edges)
 
| Weight | Connection | Direction |
|---|---|---|
| −1.234 | Thal_Re_L ↔ Thal_PuA_R | Hypo-connectivity in ZAN |
| +0.879 | Frontal_Sup_Medial_R ↔ Thal_MDl_L | Hyper-connectivity in ZAN |
| +0.723 | Parietal_Sup_R ↔ Temporal_Pole_Sup_L | Hyper-connectivity in ZAN |
| −0.688 | Frontal_Mid_2_R ↔ Thal_PuA_R | Hypo-connectivity in ZAN |
| +0.558 | Frontal_Inf_Tri_R ↔ Parietal_Inf_L | Hyper-connectivity in ZAN |
| +0.463 | Insula_L ↔ Temporal_Mid_R | Hyper-connectivity in ZAN |
 
Predominant pattern: **thalamo-frontal and thalamo-temporal dysconnectivity**, consistent with known ZAN pathophysiology.
 
### Cell-by-Cell Guide
 
| Cell | Description |
|---|---|
| 1 | Install dependencies |
| 2 | Imports, drive mount, SSL patch |
| 3 | Load participants, filter to available subjects |
| 4 | Timeseries extraction — AAL3v2, 166 ROIs |
| 5 | Tangent space FC — full 166 ROIs |
| 6 | Targeted 60-ROI FC |
| 7 | BrainNetTransformer architecture + helper functions |
| 8 | 10-Fold CV — full 166 ROIs |
| 9 | 10-Fold CV — targeted 60 ROIs |
| 10 | LOOCV — LASSO · Ridge · Linear SVM |
| 11 | XAI biomarker extraction |
| 12 | Publication figures (ROC, confusion matrix, XAI bar chart) |
| 13 | Verification: permutation, bootstrap, calibration, clinical stats |
| 14 | Results summary table |
| 15 | Demographic table (age t-test, sex chi-square) |
| 16 | Multiple comparison correction (FDR, Bonferroni) |
| 17 | Bootstrap 95% CIs on all clinical metrics |
 
---
 
## Notebook 2 — EEG Classification *(In Progress)*
 
**File:** `02_EEG/ZAN_EEG_Classification.ipynb`
 
Planned pipeline:
- EEG preprocessing (bandpass, artifact rejection, epoching)
- Feature extraction: power spectral density, connectivity (coherence / PLV), microstate analysis
- Same classification framework: SVM · MLP · LOOCV · LASSO
- XAI: electrode-level and frequency-band biomarkers
*Results to be updated upon completion.*
 
---
 
## Notebook 3 — fMRI + EEG Fusion *(In Progress)*
 
**File:** `03_Fusion/ZAN_fMRI_EEG_Fusion.ipynb`
 
Planned pipeline:
- Feature-level fusion: concatenate fMRI connectome edges + EEG features
- Decision-level fusion: ensemble of fMRI LASSO + EEG LASSO predictions
- Comparison of fusion strategies vs unimodal baselines
- Joint XAI: which modality contributes more to each prediction
*Results to be updated upon completion.*
 
---
 
## Requirements
 
```
Python        3.12
nilearn       0.13.1
nibabel       5.4.2
scikit-learn  1.6.1
tensorflow    2.x
numpy         2.0.2
pandas        2.2.2
matplotlib    3.10.0
seaborn       0.13.2
scipy
statsmodels
```
 
All notebooks run on **Google Colab**. GPU recommended for Cells 8–9 (BrainNetTransformer). All other cells run on CPU.
 
```python
# Runtime → Change runtime type → T4 GPU
```
 
---
 
## Reproducibility Notes
 
- Random seed fixed at `42` across all models and cross-validation splits
- Tangent space FC computed using `nilearn.connectome.ConnectivityMeasure(kind='tangent')`
- Feature selection (`SelectKBest`, ANOVA F-test, k=30) applied **within each LOOCV fold** to prevent data leakage
- Subjects with < 166 AAL3v2 ROIs (sub-17: 165 ROIs, sub-29: 163 ROIs, sub-45: 163 ROIs) excluded from FC computation
- Subject sub-51 excluded due to insufficient scan data
---
 
## Citation
 
If you use this code or results, please cite:
 
```
[Your Name et al.] (2025).
Multimodal Neuroimaging Biomarkers for ZAN Classification:
fMRI Functional Connectivity, EEG, and Fusion.
[Journal / Conference]. [DOI]
```
 
---
 
## License
 
MIT License — free to use with attribution.
 
---
 
*This repository is actively maintained. EEG and fusion notebooks will be added upon completion.*
 
