# 🧬 Explainable Multi-Omics Machine Learning Framework for Functional Prioritization of Copy Number Alterations in Breast Cancer

## 📌 Overview

Breast cancer contains numerous **Copy Number Alterations (CNAs)**, including genomic amplifications and deletions. However, not every CNA has the same biological importance. Identifying which alterations are potentially functionally important is therefore a challenging problem.

This project develops an **explainable multi-omics machine learning framework** to prioritize potentially important CNA-associated genes in breast cancer.

The framework integrates three complementary molecular layers:

- 🧬 **Copy Number Alterations (CNA)** — identifies genomic gains and losses
- 🧪 **RNA-seq Gene Expression** — evaluates changes in gene activity
- 🧫 **DNA Methylation** — captures epigenetic regulation

These data are integrated at the gene level, transformed into biologically meaningful features, and used to train machine-learning models.

The framework produces a **Functional Importance Score (FIS)** for each gene and uses **SHAP (SHapley Additive exPlanations)** to explain which features influence the model's predictions.

---

## 🎯 Objective

The main objective is to build a computational framework that can:

1. Integrate CNA, gene-expression, and DNA-methylation data.
2. Extract meaningful features from multiple molecular layers.
3. Identify patterns associated with potentially functionally important genes.
4. Rank genes using a **Functional Importance Score (FIS)**.
5. Explain model predictions using SHAP.
6. Provide an interpretable list of candidate genes for further investigation.

---

## 🔬 Methodology

The overall workflow is:

```text
TCGA-BRCA Multi-Omics Data
          │
          ├── CNA
          ├── RNA-seq
          └── DNA Methylation
                    │
                    ▼
          Data Preprocessing
                    │
                    ▼
        Gene ID Normalization
              using GENCODE
                    │
                    ▼
        Sample & Gene Integration
                    │
                    ▼
          Feature Engineering
                    │
                    ▼
       ┌────────────┼────────────┐
       ▼            ▼            ▼
 Random Forest   XGBoost     LightGBM
       └────────────┼────────────┘
                    ▼
             Ensemble Model
                    │
                    ▼
       Functional Importance Score
                    │
                    ▼
           SHAP Explainability
                    │
                    ▼
          Prioritized Candidate Genes
```

---

## 📊 Dataset

The project uses publicly available **TCGA-BRCA** molecular data along with external annotation and reference information.

| Dataset | Purpose |
|---|---|
| **GISTIC2 CNA** | Copy number alteration information |
| **HiSeqV2 RNA-seq** | Gene-expression information |
| **HumanMethylation450** | DNA-methylation information |
| **Clinical Data** | Patient and survival information |
| **GENCODE v50** | Gene annotation and identifier mapping |
| **ClinVar** | Reference information for functional/pathogenic labeling |

After preprocessing and integration:

- **16,162 common genes**
- **779 common samples**
- **3 integrated omics layers**
- **42 engineered features**

---

## 🧠 Feature Engineering

A total of **42 features** were generated from the integrated multi-omics data.

### CNA Features — 13

These capture the magnitude, frequency, and distribution of copy number alterations.

Examples:

- Mean CNA
- CNA standard deviation
- Amplification frequency
- Deletion frequency
- CNA variance
- CNA magnitude
- CNA alteration frequency
- CNA skewness
- CNA kurtosis

### Expression Features — 11

These describe the distribution and variability of gene-expression levels.

Examples:

- Mean expression
- Maximum expression
- Expression variance
- Expression range
- Coefficient of variation
- Median expression
- Expression skewness
- Expression kurtosis

### Methylation Features — 11

These capture methylation levels and their variability across samples.

Examples:

- Mean methylation
- Methylation standard deviation
- High/low methylation frequency
- Methylation range
- Methylation variance-related measures
- Methylation skewness
- Methylation kurtosis

### Cross-Omics Interaction Features — 4

These features capture relationships between different molecular layers:

- `cna_expr_correlation`
- `cna_expr_consistency`
- `cna_methyl_correlation`
- `cna_expr_discrepancy`

These are particularly important because they allow the framework to determine whether a CNA is associated with corresponding changes in gene expression or methylation.

### Biological Features — 3

- Protein-coding status
- Gene length
- Known oncogene indicator

---

## 🤖 Machine Learning

Three tree-based machine-learning models were trained and evaluated:

### Random Forest

```text
n_estimators = 300
max_depth = 20
```

### XGBoost

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
```

### LightGBM

```text
n_estimators = 300
num_leaves = 50
learning_rate = 0.05
```

A **soft-voting ensemble** was also developed by combining the predictions of the individual models.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 66.2% | 35.9% | 29.9% | 32.6% | 0.599 |
| XGBoost | 62.0% | 33.8% | 40.8% | 37.0% | 0.597 |
| LightGBM | 62.0% | 33.1% | 38.5% | 35.6% | 0.588 |
| **Ensemble** | **64.1%** | **35.3%** | **37.4%** | **36.3%** | **0.600** |

### Cross-Validation

| Metric | Result |
|---|---:|
| CV Accuracy | **65.1% ± 0.5%** |
| CV AUC-ROC | **59.7% ± 0.7%** |

### Precision@K

| K | Precision@K |
|---:|---:|
| 10 | 60.0% |
| 20 | 60.0% |
| 50 | 42.0% |
| 100 | 40.0% |
| 200 | 42.0% |
| 500 | 37.0% |

The relatively high Precision@10 and Precision@20 indicate that the highest-ranked candidates contain a larger proportion of positive-labelled genes.

---

## 🏆 Functional Importance Score

The framework generates a **Functional Importance Score (FIS)** for each analyzed gene.

```text
FIS = Model-derived functional importance probability

Range:
0 ─────────────────────────────── 1
Low Priority                  High Priority
```

A higher FIS indicates that the model considers the gene more likely to be functionally important based on its integrated multi-omics characteristics.

The FIS enables genes to be ranked according to their predicted functional relevance.

---

## 🔍 Explainable AI with SHAP

Machine-learning predictions can be difficult to interpret. To address this, the framework uses **SHAP** to determine which features contribute most strongly to the predictions.

### Top SHAP Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `expr_max` | 0.0528 |
| 2 | `expr_mean` | 0.0499 |
| 3 | `cna_methyl_correlation` | 0.0432 |
| 4 | `expr_range` | 0.0432 |
| 5 | `cna_expr_correlation` | 0.0427 |
| 6 | `methyl_std` | 0.0426 |
| 7 | `expr_cv` | 0.0420 |
| 8 | `methyl_range` | 0.0409 |
| 9 | `expr_variance` | 0.0402 |
| 10 | `cna_expr_consistency` | 0.0395 |

### Key Observation

Expression-related features such as `expr_max` and `expr_mean` were among the strongest contributors to model predictions.

Cross-omics features such as `cna_expr_correlation` and `cna_methyl_correlation` were also highly influential.

This demonstrates that the model is not relying only on the presence of a CNA, but also considers how the CNA relates to **gene expression and DNA methylation**.

---

## 🧬 Top Prioritized Genes

The highest-ranked genes according to their Functional Importance Score include:

| Rank | Gene | FIS |
|---:|---|---:|
| 1 | **KRT81** | 0.842 |
| 2 | **NDUFA1** | 0.766 |
| 3 | **HSPG2** | 0.753 |
| 4 | **HTATSF1** | 0.751 |
| 5 | **WDR45** | 0.749 |
| 6 | **COX7B** | 0.746 |
| 7 | **HBS1L** | 0.740 |
| 8 | **DKC1** | 0.736 |
| 9 | **SEL1L3** | 0.731 |
| 10 | **MORC4** | 0.730 |
| 11 | **LANCL1** | 0.723 |
| 12 | **DLC1** | 0.722 |
| 13 | **CD99L2** | 0.720 |
| 14 | **FBXO28** | 0.719 |
| 15 | **GCNT3** | 0.718 |
| 16 | **IL10RA** | 0.718 |
| 17 | **ETS1** | 0.716 |
| 18 | **SYT7** | 0.706 |
| 19 | **RPS24** | 0.705 |
| 20 | **COL6A2** | 0.698 |

The ranked list contains both known cancer-associated genes and candidate genes that may require further biological investigation.

---

## 🧪 ClinVar Validation

ClinVar-derived information was used as a reference for evaluating the positive label set.

```text
Positive labels : 4,419
ClinVar overlap : 4,416
Overlap         : 99.93%
```

This provides a reference-based assessment of the labels used for model development.

---

## 📊 Visualizations

The project generates visualizations for model evaluation, explainability, prioritization, and clinical analysis.

### Model Evaluation
- ROC curves
- Confusion matrices
- Cross-validation results

### Explainability
- SHAP summary plot
- SHAP feature-importance plot

### Gene Prioritization
- FIS distribution
- Top-ranked gene visualization

### Clinical Analysis
- Kaplan-Meier survival curve
- Survival by risk group

### Multi-Omics Analysis
- Gene-expression heatmaps
- CNA-expression relationships
- Methylation-related analysis

---

## 🖥️ Interactive Dashboard

A **Streamlit dashboard** is included to provide an interactive way of exploring the model outputs.

Run the dashboard using:

```bash
streamlit run app.py
```

The dashboard can be used to explore:

- Functional Importance Scores
- Ranked genes
- Model predictions
- SHAP feature importance
- Multi-omics features
- Generated visualizations

---

## 📁 Project Structure

```text
breast-cancer-cna-prioritization/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   ├── clinical_validator.py
│   ├── feature_selector.py
│   └── hyperparameter_tuner.py
│
├── results/
│   ├── figures/
│   ├── rankings/
│   └── models/
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-cna-prioritization.git
cd breast-cancer-cna-prioritization
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python main.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **LightGBM**
- **SHAP**
- **Matplotlib**
- **Seaborn**
- **Streamlit**
- **Lifelines**
- **GENCODE**
- **TCGA-BRCA**
- **ClinVar**

---

## 📌 Key Takeaway

This project demonstrates how **multi-omics data and explainable machine learning can be combined to prioritize potentially functionally important CNA-associated genes in breast cancer**.

Rather than relying solely on CNA recurrence, the framework considers **genomic alterations, gene-expression patterns, DNA methylation, and cross-omics relationships**, producing an interpretable ranking through the Functional Importance Score and SHAP analysis.
