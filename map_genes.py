import pandas as pd

# Load FIS results
fis_df = pd.read_csv('results/rankings/functional_importance_scores.csv')

# Load GENCODE for mapping
gencode = pd.read_csv('data/processed/gene_annotation.csv')

# Create mapping from gene_id to gene_name
gene_map = {}
for _, row in gencode.iterrows():
    gene_id = row['gene_id'].split('.')[0]
    gene_name = row['gene_name']
    if gene_id not in gene_map:
        gene_map[gene_id] = gene_name

# Map genes
def get_symbol(ensembl_id):
    base_id = ensembl_id.split('.')[0]
    return gene_map.get(base_id, base_id)

fis_df['gene_symbol'] = fis_df['gene'].apply(get_symbol)

# Verify mapping worked
print(f"Total rows: {len(fis_df)}")
print(f"Rows with symbols: {fis_df['gene_symbol'].notna().sum()}")
print(f"Unique symbols: {fis_df['gene_symbol'].nunique()}")

# Show top 20 with corrected symbols
print("\nTop 20 genes with symbols:")
print(fis_df[['gene', 'gene_symbol', 'FIS']].head(20))

# Save
fis_df.to_csv('results/rankings/functional_importance_scores_with_symbols.csv', index=False)
print("\nSaved to results/rankings/functional_importance_scores_with_symbols.csv")