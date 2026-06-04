# Fuzzy TOPSIS for MSME Digital Readiness Selection
Supporting code and data for the paper submitted to ICCSCI 2026.

## Files
- `fuzzy_topsis_final.py` — Full pipeline: preprocessing → Fuzzy TOPSIS → K-Means → Bootstrap → Ablation → Borderline → Empirical Proxy Validation
- `UMKM_TKM_Dataset_Bersih__1_.xlsx` — Cleaned dataset (Kriteria_Encoded sheet)
- `Research_Data_Assets_FuzzyTOPSIS.xlsx` — All computation matrices

## How to Run
pip install numpy pandas scipy scikit-learn openpyxl
python fuzzy_topsis_final.py

## Dataset
478 MSMEs from the Kemnaker TKM Programme, 2024.
