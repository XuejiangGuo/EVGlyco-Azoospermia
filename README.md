# EVGlycoAzoospermia

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936656.svg)](https://doi.org/10.5281/zenodo.21936656)

## Title

Site-specific N-glycoproteomic analysis of extracellular vesicles from seminal plasma reveals the molecular typing of azoospermia

## Overview

This repository contains the analysis code, partial/example data, and representative output figures for the EVGlyco azoospermia study. The code is organized by analysis module so that each script can be run from its own directory with the accompanying input files.

The folder names use the following abbreviations:

- `DC`: discovery cohort
- `VC`: validation cohort
- `OA`: obstructive azoospermia
- `NOA`: non-obstructive azoospermia

## Repository structure

| Folder | Main script | Purpose |
|---|---|---|
| `ANOVA-cluster/` | `ANOVA_Mfuzz.R` | ANOVA filtering, Mfuzz clustering, and heatmap output |
| `volcano/` | `volcano.R` | Differential glycopeptide volcano plots |
| `PCA/` | `PCA.R` | PCA visualization by sample type and batch |
| `UMAP/` | `UMAP.R` | UMAP visualization by sample type and batch |
| `DC-OA-NOA-SHAP/` | `SHAP.py` | Discovery cohort OA vs NOA random forest and SHAP analysis |
| `VC-OA-NOA-SHAP/` | `SHAP.py` | Validation cohort OA vs NOA random forest and SHAP analysis |
| `DC-OA-NOA-eight-ML/` | `DC-OA-NOA-eight-ML.py` | Discovery cohort OA vs NOA comparison of eight machine-learning models |
| `VC-OA-NOA-eight-ML/` | `VC-OA-NOA-eight-ML.py` | Validation cohort OA vs NOA comparison of eight machine-learning models |
| `DC-OA-NOA-AUC-single/` | `AUC_single_Cohen's d.py` | Discovery cohort OA vs NOA single-feature ROC analysis |
| `DC-OA-NOA-AUC-panel/` | `AUC_panel_Cohen's d.py` | Discovery cohort OA vs NOA panel ROC analysis |
| `VC-OA-NOA-AUC-single-panel/` | `AUC_single_Cohen's d.py` | Validation cohort OA vs NOA single-feature and panel ROC analysis |
| `DC-NOAsub-AUC-single/` | `AUC-subNOA-single-Cohen's d.py` | Discovery cohort NOA subtype single-feature ROC analysis |
| `DC-NOAsub-AUC-panel/` | `AUC-NOAsub-panel-Cohen's d.py` | Discovery cohort NOA subtype panel ROC analysis |
| `VC-NOAsub-AUC-single/` | `AUC-subNOA-single-Cohen's d.py` | Validation cohort NOA subtype single-feature ROC analysis |
| `VC-NOAsub-AUC-panel/` | `AUC-NOAsub-panel-Cohen's d.py` | Validation cohort NOA subtype panel ROC analysis |
| `docs/` | `file_manifest.tsv` | File list and size manifest |

## Data

Each analysis folder includes partial/example data needed by the corresponding script. This repository is not a full raw-data archive; it is prepared as a runnable code-and-example-data package for reviewer assessment. The scripts use local relative paths, so run each script from inside its own folder.

The repository includes representative generated PDF figures. Some scripts also write JPG outputs when run.

## Install dependencies

Python scripts were prepared for Python 3.12. Install the required Python packages with:

```bash
pip install -r requirements.txt
```

R scripts were prepared for R 4.3. Install the required R packages with:

```r
install.packages(c(
  "dplyr", "ggplot2", "ggrepel", "ggpubr", "ggsci", "ggvenn",
  "openxlsx", "patchwork", "RColorBrewer", "scales", "stringr",
  "umap", "DescTools", "dendextend", "circlize"
))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
BiocManager::install(c("Mfuzz", "ComplexHeatmap"))
```

## Run examples

Run Python analyses from the target folder:

```bash
cd DC-OA-NOA-SHAP
python SHAP.py
```

```bash
cd VC-OA-NOA-eight-ML
python VC-OA-NOA-eight-ML.py
```

Run R analyses from the target folder:

```bash
cd PCA
Rscript PCA.R
```

```bash
cd ANOVA-cluster
Rscript ANOVA_Mfuzz.R
```

## Notes for reviewer assessment

The scripts intentionally keep their original folder-level input files so that they can be run directly without rewriting paths. Generated output figures are stored either in the same folder or in local `fig/` subfolders.

## Citation

Please cite the version of the analysis code archived on Zenodo:

> Guo, X. & Ni, J. (2026). *Site-specific N-glycoproteomic analysis of extracellular vesicles from seminal plasma reveals the molecular typing of azoospermia* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21936656

## License

This repository is released under the MIT License. See `LICENSE`.
