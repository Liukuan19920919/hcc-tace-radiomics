# HCC TACE Radiomics: Interpretable ML for TACE Response Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Data: WAW-TACE](https://img.shields.io/badge/data-WAW--TACE%20(CC%20BY%204.0)-orange.svg)](https://zenodo.org/records/12741586)

Fully reproducible machine learning pipeline for predicting transarterial chemoembolization (TACE) treatment response and survival in hepatocellular carcinoma (HCC) using baseline multiphase CT radiomics with SHAP interpretability.

## Paper

**"Interpretable Machine Learning Based on Baseline Multiphase CT Radiomics for Predicting Transarterial Chemoembolization Response and Survival in Hepatocellular Carcinoma: A Fully Reproducible Analysis Using a Public Dataset"**

*Submitted to European Radiology (2026)*

## Key Results

| Model | AUC (5-fold CV) |
|-------|----------------|
| **Random Forest (Radiomics)** | **0.831 ± 0.051** |
| Logistic Regression (Radiomics) | 0.771 ± 0.019 |
| Clinical Baseline | 0.746 ± 0.066 |

- Radiomics-based risk stratification significantly discriminates **overall survival** (log-rank P < 0.0001) and **progression-free survival** (P = 0.039)
- **SHAP analysis** identifies liver tumor texture, skeletal morphology (vertebrae, ribs), and clinical etiology as key predictors
- Portal venous phase significantly outperforms arterial, non-contrast, and delayed phases

## Data

All analyses use the publicly available **WAW-TACE dataset**:

> Bartnik K, Bartczak T, Krzyziński M, et al. WAW-TACE: A Hepatocellular Carcinoma Multiphase CT Dataset with Segmentations, Radiomics Features, and Clinical Data. *Radiol Artif Intell*. 2024.

- **233 treatment-naïve HCC patients** treated with TACE
- **Multiphasic CT** (non-contrast, arterial, portal venous, delayed)
- **3,339 PyRadiomics features** from 104 anatomical regions
- **Clinical data**: demographics, lab values, OS, PFS, LR-TR response
- **Download**: [Zenodo (DOI: 10.5281/zenodo.12741586)](https://zenodo.org/records/12741586) — CC BY 4.0

## Installation

```bash
git clone https://github.com/KuanLiu-hbu/hcc-tace-radiomics.git
cd hcc-tace-radiomics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
hcc-tace-radiomics/
├── src/
│   ├── data/load_data.py          # DICOM/NIfTI loading, clinical data parsing
│   ├── features/extract_qvmf.py   # QVMF extraction (skeletonization, radius bins)
│   ├── modeling/train_models.py   # LASSO + LR/RF/SVM + SHAP pipeline
│   └── visualization/plots.py     # Publication-quality figures
├── scripts/
│   ├── download_all.sh            # Auto-download datasets
│   ├── zenodo_download.py         # Zenodo API downloader
│   └── run_pipeline.py            # End-to-end pipeline runner
└── outputs/
    └── er_submission_ready.md     # Manuscript
```

## Quick Start

```python
from src.modeling.train_models import run_full_pipeline
import pandas as pd

# Load your feature DataFrame with 'response' column
df = pd.read_csv("your_features.csv")

# Run complete pipeline: LASSO → CV → SHAP
results = run_full_pipeline(df, response_col='response', output_dir='outputs')
```

## Features

- **Two-stage feature selection**: Univariate screening (Mann-Whitney U) + LASSO (L1-regularized LR)
- **Multiple classifiers**: Logistic Regression, Random Forest, SVM with 5-fold stratified CV
- **Class imbalance handling**: SMOTE oversampling on training folds
- **SHAP interpretability**: Summary plots, bar plots, waterfall plots, correlation networks
- **Survival analysis**: Kaplan-Meier stratification, Cox proportional hazards
- **Phase comparison**: Model performance across all 4 CT contrast phases

## Citation

If you use this code or the WAW-TACE dataset, please cite:

```bibtex
@article{liu2026radiomics,
  title={Interpretable Machine Learning Based on Baseline Multiphase CT Radiomics 
         for Predicting TACE Response and Survival in HCC},
  author={Liu, Kuan},
  journal={European Radiology (under review)},
  year={2026}
}

@article{bartnik2024waw,
  title={WAW-TACE: A Hepatocellular Carcinoma Multiphase CT Dataset with 
         Segmentations, Radiomics Features, and Clinical Data},
  author={Bartnik, Krzysztof and Bartczak, Tomasz and Krzyzi{\'n}ski, Mateusz and others},
  journal={Radiology: Artificial Intelligence},
  year={2024},
  doi={10.1148/ryai.240296}
}
```

## License

MIT License. The WAW-TACE dataset is distributed under CC BY 4.0.

## Contact

Kuan Liu (刘宽) — KuanLiu@hbu.edu.cn  
Department of Radiation Oncology, Affiliated Hospital of Hebei University
