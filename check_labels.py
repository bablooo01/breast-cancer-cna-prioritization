import pandas as pd

# Load labels
labels = pd.read_csv('data/processed/labels.csv', index_col=0)
print(f"Labels shape: {labels.shape}")
print(f"Labels columns: {labels.columns.tolist()}")

# Fix column name if needed
if labels.columns.tolist() == ['0']:
    labels = labels.rename(columns={'0': 'label'})
elif 'label' not in labels.columns:
    if len(labels.columns) == 0:
        labels = pd.DataFrame({'label': labels.values}, index=labels.index)
    else:
        labels = pd.DataFrame({'label': labels.iloc[:, 0].values}, index=labels.index)

print(f"Labels after fix: {labels.columns.tolist()}")
print(f"Label distribution: {labels['label'].value_counts().to_dict()}")

# Load GENCODE for mapping
gencode = pd.read_csv('data/processed/gene_annotation.csv')
gene_map = {}
for _, row in gencode.iterrows():
    gene_id = row['gene_id'].split('.')[0]
    gene_name = row['gene_name']
    if gene_id not in gene_map:
        gene_map[gene_id] = gene_name

# Load ClinVar
clinvar = pd.read_csv('data/processed/clinvar_genes.csv', low_memory=False)
pathogenic = clinvar[
    clinvar['clinical_significance'].str.contains('Pathogenic', na=False)
]['gene_symbol'].unique()
print(f"\nPathogenic genes in ClinVar: {len(pathogenic)}")

# Check positive labels
positive_genes = labels[labels['label'] == 1].index.tolist()
print(f"\nPositive labels: {len(positive_genes)}")

# Count how many positive labels are in ClinVar
count_in_clinvar = 0
clinvar_matched_genes = []
for gene in positive_genes:
    base_id = gene.split('.')[0]
    symbol = gene_map.get(base_id, '')
    if symbol in pathogenic:
        count_in_clinvar += 1
        clinvar_matched_genes.append(symbol)

print(f"Positive labels in ClinVar: {count_in_clinvar}")
if len(positive_genes) > 0:
    print(f"Percentage: {count_in_clinvar/len(positive_genes)*100:.2f}%")
else:
    print("No positive labels found")

if count_in_clinvar > 0:
    print(f"\nSample ClinVar-matched genes: {clinvar_matched_genes[:10]}")