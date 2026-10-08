import pandas as pd

# Load processed data
cna = pd.read_csv('data/processed/cna_matrix.csv', index_col=0)
expr = pd.read_csv('data/processed/expression_matrix.csv', index_col=0)

print(f"CNA genes: {len(cna)}")
print(f"Expression genes: {len(expr)}")

# Check for duplicates
cna_dup = cna.index.duplicated().sum()
expr_dup = expr.index.duplicated().sum()
print(f"CNA duplicates: {cna_dup}")
print(f"Expression duplicates: {expr_dup}")

# Check common genes
common = set(cna.index) & set(expr.index)
print(f"Common genes: {len(common)}")

# Check why features has 24265 rows
features = pd.read_csv('data/processed/multi_omics_features.csv', index_col=0)
print(f"Features rows: {len(features)}")
print(f"Features index first 10: {features.index[:10].tolist()}")