import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
RANKINGS_DIR = RESULTS_DIR / 'rankings'
MODELS_DIR = RESULTS_DIR / 'models'

# Create directories
for dir_path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, RANKINGS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths - update these to match your exact file names
CNA_FILE = RAW_DIR / 'TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz'
EXPRESSION_FILE = RAW_DIR / 'TCGA.BRCA.sampleMap_HiSeqV2.gz'
METHYLATION_FILE = RAW_DIR / 'TCGA.BRCA.sampleMap_HumanMethylation450.gz'
CLINICAL_FILE = RAW_DIR / 'TCGA.BRCA.sampleMap_BRCA_clinicalMatrix'
GENCODE_FILE = RAW_DIR / 'gencode.v50.basic.annotation.gtf.gz'
CLINVAR_FILE = RAW_DIR / 'clinvar.vcf.gz'

# Output files
CNA_PROCESSED = PROCESSED_DIR / 'cna_matrix.csv'
EXPRESSION_PROCESSED = PROCESSED_DIR / 'expression_matrix.csv'
METHYLATION_PROCESSED = PROCESSED_DIR / 'methylation_matrix.csv'
CLINICAL_PROCESSED = PROCESSED_DIR / 'clinical_data.csv'
GENCODE_PROCESSED = PROCESSED_DIR / 'gene_annotation.csv'
CLINVAR_PROCESSED = PROCESSED_DIR / 'clinvar_genes.csv'
MULTI_OMICS_FEATURES = PROCESSED_DIR / 'multi_omics_features.csv'
LABELS_FILE = PROCESSED_DIR / 'labels.csv'
FINAL_DATASET = PROCESSED_DIR / 'final_dataset.csv'

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
CNA_THRESHOLD_AMP = 0.3
CNA_THRESHOLD_DEL = -0.3
METHYLATION_HIGH = 0.7
METHYLATION_LOW = 0.3