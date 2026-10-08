import pandas as pd

# Check what symbols exist for your top genes
gencode = pd.read_csv('data/processed/gene_annotation.csv')

# Create gene map
gene_map = {}
for _, row in gencode.iterrows():
    gene_id = row['gene_id'].split('.')[0]
    gene_name = row['gene_name']
    if gene_id not in gene_map:
        gene_map[gene_id] = gene_name

top_genes = [
    'ENSG00000180182.13',
    'ENSG00000068323.19',
    'ENSG00000126903.18',
    'ENSG00000205609.15',
    'ENSG00000124486.17',
    'ENSG00000232119.10',
    'ENSG00000078808.22',
    'ENSG00000150938.12',
    'ENSG00000131171.15',
    'ENSG00000157514.20'
]

print("Gene Mapping Verification:")
print("-" * 40)
for gene in top_genes:
    base_id = gene.split('.')[0]
    symbol = gene_map.get(base_id, "NOT FOUND")
    print(f"{gene:25s} -> {symbol}")