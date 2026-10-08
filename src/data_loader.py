import pandas as pd
import numpy as np
import gzip
import re
from pathlib import Path
from tqdm import tqdm
from src import config

class DataLoader:
    def __init__(self):
        self.cna = None
        self.expression = None
        self.methylation = None
        self.clinical = None
        self.gencode = None
        self.clinvar = None
        
    def load_cna(self, file_path=None):
        """Load GISTIC2 CNA data"""
        if file_path is None:
            file_path = config.CNA_FILE
            
        print("Loading CNA data...")
        self.cna = pd.read_csv(file_path, sep='\t', index_col=0)
        print(f"CNA shape: {self.cna.shape}")
        print(f"CNA samples: {self.cna.columns[:5].tolist()}")
        return self.cna
    
    def load_expression(self, file_path=None):
        """Load RNA-seq expression data"""
        if file_path is None:
            file_path = config.EXPRESSION_FILE
            
        print("Loading expression data...")
        self.expression = pd.read_csv(file_path, sep='\t', index_col=0)
        print(f"Expression shape: {self.expression.shape}")
        print(f"Expression samples: {self.expression.columns[:5].tolist()}")
        return self.expression
    
    def load_methylation(self, file_path=None):
        """Load methylation data - properly handles probe IDs as strings"""
        if file_path is None:
            file_path = config.METHYLATION_FILE
            
        print("Loading methylation data...")
        
        # Try loading with proper string handling for first column
        probe_amounts = [150000, 100000, 80000, 50000, 30000]
        
        for nrows in probe_amounts:
            try:
                print(f"Trying {nrows:,} probes...")
                
                # Read with proper data types - first column as string
                df = pd.read_csv(
                    file_path,
                    sep='\t',
                    header=0,
                    low_memory=False,
                    nrows=nrows,
                    dtype={0: str}  # First column (probe IDs) as string
                )
                
                # Set first column as index
                df.index = df.iloc[:, 0].astype(str)
                df = df.iloc[:, 1:]  # Remove first column
                
                # Convert all remaining columns to float32
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove rows with >50% missing values
                df = df.dropna(thresh=int(0.5 * len(df.columns)))
                
                print(f"✓ Loaded {len(df):,} probes after cleaning")
                self.methylation = df
                print(f"Methylation shape: {self.methylation.shape}")
                print(f"Samples: {self.methylation.columns[:5].tolist()}")
                return self.methylation
                
            except MemoryError:
                print(f"✗ {nrows:,} probes caused memory error, trying smaller...")
                continue
            except Exception as e:
                print(f"✗ Failed at {nrows:,} probes: {e}")
                continue
        
        # If all fail, try fallback
        print("All methods failed. Trying fallback with 30,000 probes...")
        try:
            df = pd.read_csv(
                file_path,
                sep='\t',
                low_memory=False,
                nrows=30000,
                header=0
            )
            df.index = df.iloc[:, 0].astype(str)
            df = df.iloc[:, 1:]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(thresh=int(0.5 * len(df.columns)))
            self.methylation = df
            print(f"Methylation loaded (fallback): {self.methylation.shape}")
            return self.methylation
        except Exception as e:
            print(f"All loading methods failed: {e}")
            self.methylation = None
            return None
    
    def load_clinical(self, file_path=None):
        """Load clinical data"""
        if file_path is None:
            file_path = config.CLINICAL_FILE
            
        print("Loading clinical data...")
        self.clinical = pd.read_csv(file_path, sep='\t', index_col=0)
        print(f"Clinical shape: {self.clinical.shape}")
        print(f"Clinical columns: {self.clinical.columns[:10].tolist()}")
        return self.clinical
    
    def load_gencode(self, file_path=None):
        """Load GENCODE annotation"""
        if file_path is None:
            file_path = config.GENCODE_FILE
            
        print("Loading GENCODE annotation...")
        
        genes = []
        with gzip.open(file_path, 'rt') as f:
            for line in tqdm(f, desc="Parsing GENCODE"):
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 9 and parts[2] == 'gene':
                    attributes = parts[8]
                    gene_id = re.search(r'gene_id "([^"]+)"', attributes)
                    gene_name = re.search(r'gene_name "([^"]+)"', attributes)
                    gene_type = re.search(r'gene_type "([^"]+)"', attributes)
                    
                    if gene_id and gene_name:
                        genes.append({
                            'gene_id': gene_id.group(1),
                            'gene_name': gene_name.group(1),
                            'gene_type': gene_type.group(1) if gene_type else 'unknown',
                            'chromosome': parts[0],
                            'start': int(parts[3]),
                            'end': int(parts[4]),
                            'strand': parts[6]
                        })
        
        self.gencode = pd.DataFrame(genes)
        print(f"GENCODE shape: {self.gencode.shape}")
        print(f"Unique gene types: {self.gencode['gene_type'].nunique()}")
        return self.gencode
    
    def load_clinvar(self, file_path=None):
        """Load ClinVar VCF and extract gene information"""
        if file_path is None:
            file_path = config.CLINVAR_FILE
            
        print("Loading ClinVar data...")
        clinvar_genes = []
        
        with gzip.open(file_path, 'rt') as f:
            for line in tqdm(f, desc="Parsing ClinVar"):
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 8:
                    info = parts[7]
                    gene_info = re.search(r'GENEINFO=([^;]+)', info)
                    clnsig = re.search(r'CLNSIG=([^;]+)', info)
                    
                    if gene_info:
                        gene_string = gene_info.group(1)
                        genes = re.findall(r'([^:]+):\d+', gene_string)
                        
                        for gene in genes:
                            clinvar_genes.append({
                                'gene_symbol': gene,
                                'clinical_significance': clnsig.group(1) if clnsig else 'unknown',
                                'variant_id': parts[2],
                                'chromosome': parts[0],
                                'position': parts[1]
                            })
        
        self.clinvar = pd.DataFrame(clinvar_genes)
        print(f"ClinVar records: {len(self.clinvar)}")
        print(f"Unique genes: {self.clinvar['gene_symbol'].nunique()}")
        return self.clinvar
    
    def inspect_all(self):
        """Quick inspection of all loaded data"""
        data_dict = {
            'CNA': self.cna,
            'Expression': self.expression,
            'Methylation': self.methylation,
            'Clinical': self.clinical,
            'GENCODE': self.gencode,
            'ClinVar': self.clinvar
        }
        
        print("\n" + "="*60)
        print("DATA INSPECTION SUMMARY")
        print("="*60)
        
        for name, data in data_dict.items():
            if data is not None:
                print(f"\n{name}:")
                print(f"  Shape: {data.shape}")
                print(f"  Missing values: {data.isnull().sum().sum()}")
                if hasattr(data, 'columns'):
                    print(f"  First 3 columns: {data.columns[:3].tolist()}")
                if hasattr(data, 'index'):
                    print(f"  First 3 indices: {data.index[:3].tolist()}")