import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src import config
from src.data_loader import DataLoader

class Preprocessor:
    def __init__(self):
        self.loader = DataLoader()
        self.cna = None
        self.expression = None
        self.methylation = None
        self.clinical = None
        self.gencode = None
        self.clinvar = None
        
    def load_all_data(self):
        """Load all raw data"""
        self.cna = self.loader.load_cna()
        self.expression = self.loader.load_expression()
        
        # Load methylation - will try max possible
        try:
            self.methylation = self.loader.load_methylation()
        except Exception as e:
            print(f"Methylation loading issue: {e}")
            print("Proceeding without methylation data...")
            self.methylation = None
            
        self.clinical = self.loader.load_clinical()
        self.gencode = self.loader.load_gencode()
        self.clinvar = self.loader.load_clinvar()
        
    def standardize_sample_ids(self, df, id_column=None):
        """Standardize TCGA sample IDs to format TCGA-XX-XXXX"""
        if id_column:
            ids = df[id_column]
        else:
            ids = df.index
            
        pattern = r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})'
        standardized = []
        for idx in ids:
            match = re.search(pattern, str(idx))
            if match:
                standardized.append(match.group(1))
            else:
                standardized.append(str(idx))
        
        if id_column:
            df[id_column] = standardized
            return df
        else:
            df.index = standardized
            return df
    
    def clean_cna(self):
        """Clean and prepare CNA data"""
        print("Cleaning CNA data...")
        
        self.cna = self.cna.dropna(how='all')
        self.cna = self.cna.dropna(thresh=int(0.5 * len(self.cna)), axis=1)
        
        self.cna.columns = self.standardize_sample_ids(
            pd.DataFrame({'id': self.cna.columns}), 'id'
        )['id'].values
        
        if self.cna.columns.duplicated().any():
            self.cna = self.cna.loc[:, ~self.cna.columns.duplicated()]
        
        imputer = SimpleImputer(strategy='median')
        self.cna = pd.DataFrame(
            imputer.fit_transform(self.cna),
            index=self.cna.index,
            columns=self.cna.columns
        )
        
        print(f"Cleaned CNA shape: {self.cna.shape}")
        return self.cna
    
    def clean_expression(self):
        """Clean and prepare expression data"""
        print("Cleaning expression data...")
        
        if self.expression.columns.duplicated().any():
            dup_count = self.expression.columns.duplicated().sum()
            print(f"Found {dup_count} duplicate columns. Removing...")
            self.expression = self.expression.loc[:, ~self.expression.columns.duplicated()]
        
        self.expression = self.expression.dropna(how='all')
        self.expression = self.expression.dropna(thresh=int(0.5 * len(self.expression)), axis=1)
        
        self.expression.columns = self.standardize_sample_ids(
            pd.DataFrame({'id': self.expression.columns}), 'id'
        )['id'].values
        
        if self.expression.columns.duplicated().any():
            self.expression = self.expression.loc[:, ~self.expression.columns.duplicated()]
        
        imputer = SimpleImputer(strategy='median')
        self.expression = pd.DataFrame(
            imputer.fit_transform(self.expression),
            index=self.expression.index,
            columns=self.expression.columns
        )
        
        if self.expression.max().max() > 50:
            self.expression = np.log2(self.expression + 1)
        
        print(f"Cleaned expression shape: {self.expression.shape}")
        return self.expression
    
    def clean_methylation(self):
        """Clean and prepare methylation data"""
        print("Cleaning methylation data...")
        
        if self.methylation is None:
            print("Methylation data not loaded, skipping...")
            return None
            
        self.methylation = self.methylation.dropna(how='all')
        self.methylation = self.methylation.dropna(thresh=int(0.5 * len(self.methylation)), axis=1)
        
        self.methylation.columns = self.standardize_sample_ids(
            pd.DataFrame({'id': self.methylation.columns}), 'id'
        )['id'].values
        
        if self.methylation.columns.duplicated().any():
            self.methylation = self.methylation.loc[:, ~self.methylation.columns.duplicated()]
        
        imputer = SimpleImputer(strategy='median')
        self.methylation = pd.DataFrame(
            imputer.fit_transform(self.methylation),
            index=self.methylation.index,
            columns=self.methylation.columns
        )
        
        print(f"Cleaned methylation shape: {self.methylation.shape}")
        return self.methylation
    
    def clean_clinical(self):
        """Clean and prepare clinical data"""
        print("Cleaning clinical data...")
        
        self.clinical.index = self.standardize_sample_ids(
            pd.DataFrame({'id': self.clinical.index}), 'id'
        )['id'].values
        
        self.clinical = self.clinical.dropna(thresh=int(0.5 * len(self.clinical)), axis=1)
        
        print(f"Cleaned clinical shape: {self.clinical.shape}")
        return self.clinical
    
    def find_common_samples(self):
        """Find samples common across all platforms"""
        sample_sets = [set(self.cna.columns), set(self.expression.columns)]
        
        if self.methylation is not None and not self.methylation.empty:
            sample_sets.append(set(self.methylation.columns))
        
        common_samples = set.intersection(*sample_sets)
        print(f"Common samples: {len(common_samples)}")
        
        return list(common_samples)
    
    def harmonize_gene_names(self):
        """Harmonize gene names across platforms using GENCODE"""
        print("Harmonizing gene names...")
        
        protein_coding = self.gencode[self.gencode['gene_type'] == 'protein_coding']
        gene_map = dict(zip(protein_coding['gene_name'], protein_coding['gene_id']))
        
        cna_genes = set(self.cna.index) & set(gene_map.keys())
        expr_genes = set(self.expression.index) & set(gene_map.keys())
        
        print(f"Protein-coding genes in CNA: {len(cna_genes)}")
        print(f"Protein-coding genes in Expression: {len(expr_genes)}")
        
        common_genes = cna_genes & expr_genes
        print(f"Common genes across platforms: {len(common_genes)}")
        
        self.cna = self.cna.loc[list(common_genes)]
        self.expression = self.expression.loc[list(common_genes)]
        
        self.cna.index = [gene_map[g] for g in self.cna.index]
        self.expression.index = [gene_map[g] for g in self.expression.index]
        
        return common_genes
    
    def save_processed_data(self):
        """Save processed data to CSV"""
        print("Saving processed data...")
        
        self.cna.to_csv(config.CNA_PROCESSED)
        self.expression.to_csv(config.EXPRESSION_PROCESSED)
        self.clinical.to_csv(config.CLINICAL_PROCESSED)
        self.gencode.to_csv(config.GENCODE_PROCESSED)
        
        if self.methylation is not None and not self.methylation.empty:
            self.methylation.to_csv(config.METHYLATION_PROCESSED)
        
        if self.clinvar is not None:
            self.clinvar.to_csv(config.CLINVAR_PROCESSED)
        
        print("All processed data saved successfully")
    
    def run_full_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        self.load_all_data()
        
        self.clean_cna()
        self.clean_expression()
        
        if self.methylation is not None and not self.methylation.empty:
            self.clean_methylation()
        else:
            print("Methylation not available, proceeding without it")
        
        self.clean_clinical()
        
        common_samples = self.find_common_samples()
        
        self.cna = self.cna[common_samples]
        self.expression = self.expression[common_samples]
        
        if self.methylation is not None and not self.methylation.empty:
            self.methylation = self.methylation[common_samples]
        
        self.harmonize_gene_names()
        self.save_processed_data()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return {
            'samples': len(common_samples),
            'genes': len(self.cna),
            'cna_shape': self.cna.shape,
            'expression_shape': self.expression.shape
        }