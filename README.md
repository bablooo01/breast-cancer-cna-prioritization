# Explainable Multi-Omics CNA Prioritization in Breast Cancer

An integrated machine learning framework that combines **CNA**, **RNA-seq**, and **DNA methylation** data to prioritize functionally important copy number alterations in breast cancer.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.42+-purple)

---

## 🧬 What This Project Does

Breast cancer genomes have thousands of copy number alterations (CNAs), but only some actually drive cancer. This framework:

- Integrates **3 omics layers**: CNA + RNA-seq + Methylation
- Engineers **42 features** from TCGA-BRCA data
- Trains **4 ML models**: Random Forest, XGBoost, LightGBM, Ensemble
- Generates a **Functional Importance Score (FIS)** for each gene
- Explains predictions using **SHAP**

---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| **Best Model** | Ensemble |
| **AUC-ROC** | 0.600 |
| **Accuracy** | 64.1% |
| **CV AUC-ROC** | 0.597 ± 0.007 |
| **Samples** | 779 |
| **Genes Analyzed** | 16,162 |
| **Methylation Probes** | 122,401 |
| **ClinVar Validation** | 99.93% |

---

## 🧬 Top 10 Prioritized Genes

| Rank | Gene | FIS | Function |
|------|------|-----|----------|
| 1 | **KRT81** | 0.842 | Keratin 81 |
| 2 | **NDUFA1** | 0.766 | Mitochondrial complex I |
| 3 | **HSPG2** | 0.753 | Heparan sulfate proteoglycan |
| 4 | **HTATSF1** | 0.751 | RNA processing |
| 5 | **WDR45** | 0.749 | Autophagy |
| 6 | **COX7B** | 0.746 | Cytochrome c oxidase |
| 7 | **HBS1L** | 0.740 | Translation factor |
| 8 | **DKC1** | 0.736 | Telomere maintenance |
| 9 | **SEL1L3** | 0.731 | ERAD |
| 10 | **MORC4** | 0.730 | Chromatin remodeling |

---

## 📁 Project Structure
