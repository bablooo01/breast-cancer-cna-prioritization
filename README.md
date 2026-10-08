# 🧬 Explainable Multi-Omics ML for Breast Cancer CNA Prioritization

An **explainable multi-omics machine learning framework** for prioritizing potentially functionally important Copy Number Alterations (CNAs) in breast cancer.

The project integrates **CNA, RNA-seq gene expression, and DNA methylation** data from TCGA-BRCA and uses machine learning with SHAP explainability to generate a **Functional Importance Score (FIS)** for candidate genes.

## 🔬 Key Features

- Multi-omics integration of **CNA + RNA-seq + DNA methylation**
- **42 engineered biological and statistical features**
- Random Forest, XGBoost, and LightGBM models
- Soft-voting ensemble model
- Functional Importance Score (**FIS**) for gene prioritization
- SHAP-based model explainability
- ClinVar-based reference validation
- Interactive Streamlit dashboard

## 📊 Results

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---:|---:|---:|
| Random Forest | 66.2% | 32.6% | 0.599 |
| XGBoost | 62.0% | 37.0% | 0.597 |
| LightGBM | 62.0% | 35.6% | 0.588 |
| **Ensemble** | **64.1%** | **36.3%** | **0.600** |

**Dataset:** 779 common TCGA-BRCA samples  
**Common genes:** 16,162  
**Features:** 42

### 🧠 Top SHAP Features

`expr_max` · `expr_mean` · `cna_methyl_correlation` · `expr_range` · `cna_expr_correlation`

### 🏆 Top Prioritized Genes

**KRT81, NDUFA1, HSPG2, HTATSF1, WDR45, COX7B, HBS1L, DKC1, MORC4, DLC1**

## 🛠️ Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · SHAP · Matplotlib · Streamlit · Lifelines**

## 📁 Project Structure

```text
breast-cancer-cna-prioritization/
├── data/
├── src/
├── results/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## 🚀 Run

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-cna-prioritization.git
cd breast-cancer-cna-prioritization
pip install -r requirements.txt
python main.py
```

To launch the dashboard:

```bash
streamlit run app.py
```

## 🔮 Future Work

- External validation using METABRIC
- Full methylation dataset integration
- Pathway-level analysis
- Deep-learning approaches
- Experimental validation of prioritized candidates

## 👩‍💻 Author

**Sri Raaghavi R.K.**  
B.Tech CSE – Artificial Intelligence
