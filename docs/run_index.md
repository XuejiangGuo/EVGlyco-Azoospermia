# Run index

Run each script from its own directory. The scripts use relative paths to local CSV or XLSX files.

## R scripts

```bash
cd ANOVA-cluster
Rscript ANOVA_Mfuzz.R

cd ../volcano
Rscript volcano.R

cd ../PCA
Rscript PCA.R

cd ../UMAP
Rscript UMAP.R
```

## Python scripts

```bash
cd DC-OA-NOA-SHAP
python SHAP.py

cd ../VC-OA-NOA-SHAP
python SHAP.py

cd ../DC-OA-NOA-eight-ML
python DC-OA-NOA-eight-ML.py

cd ../VC-OA-NOA-eight-ML
python VC-OA-NOA-eight-ML.py

cd ../DC-OA-NOA-AUC-single
python "AUC_single_Cohen's d.py"

cd ../DC-OA-NOA-AUC-panel
python "AUC_panel_Cohen's d.py"

cd ../VC-OA-NOA-AUC-single-panel
python "AUC_single_Cohen's d.py"

cd ../DC-NOAsub-AUC-single
python "AUC-subNOA-single-Cohen's d.py"

cd ../DC-NOAsub-AUC-panel
python "AUC-NOAsub-panel-Cohen's d.py"

cd ../VC-NOAsub-AUC-single
python "AUC-subNOA-single-Cohen's d.py"

cd ../VC-NOAsub-AUC-panel
python "AUC-NOAsub-panel-Cohen's d.py"
```
