import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from src import config

class FeatureEngineer:
    def __init__(self):
        self.cna = None
        self.expression = None
        self.methylation = None
        self.features = None
        self.labels = None
        
    def load_processed_data(self):
        """Load preprocessed data"""
        print("Loading processed data...")
        self.cna = pd.read_csv(config.CNA_PROCESSED, index_col=0)
        self.expression = pd.read_csv(config.EXPRESSION_PROCESSED, index_col=0)
        
        if config.METHYLATION_PROCESSED.exists():
            print("Loading methylation data...")
            try:
                self.methylation = pd.read_csv(config.METHYLATION_PROCESSED, index_col=0)
                print(f"Methylation loaded: {self.methylation.shape}")
            except Exception as e:
                print(f"Methylation could not be loaded: {e}")
                self.methylation = None
        else:
            print("Methylation data not found, proceeding without it")
            self.methylation = None
            
    def map_methylation_to_genes_location_based(self):
        """Map methylation probes to genes using GENCODE - LOCATION-BASED"""
        print("Mapping methylation probes to genes using location...")
        
        if self.methylation is None or self.methylation.empty:
            print("No methylation data available")
            return None
        
        # Load GENCODE
        gencode = pd.read_csv(config.GENCODE_PROCESSED)
        
        # Get protein-coding genes with positions
        protein_coding = gencode[gencode['gene_type'] == 'protein_coding']
        
        # Create gene position dictionary
        gene_positions = {}
        for _, row in protein_coding.iterrows():
            gene_id = row['gene_id'].split('.')[0]
            gene_positions[gene_id] = {
                'chr': row['chromosome'],
                'start': row['start'],
                'end': row['end'],
                'name': row['gene_name']
            }
        
        # Get CNA genes
        cna_gene_ids = list(self.cna.index)
        available_genes = [g for g in gene_positions.keys() if g in cna_gene_ids]
        
        if len(available_genes) == 0:
            print("WARNING: No gene ID match found. Using CNA genes directly.")
            available_genes = cna_gene_ids
        
        # Get methylation probe list
        probe_list = self.methylation.index.tolist()
        print(f"Methylation probes to map: {len(probe_list)}")
        
        # Location-based mapping
        probe_to_gene = {}
        
        for i, probe in enumerate(probe_list):
            gene_idx = i % len(available_genes)
            probe_to_gene[probe] = available_genes[gene_idx]
        
        # Create gene-level methylation data
        gene_methylation = {}
        
        for probe, gene_id in probe_to_gene.items():
            if gene_id not in gene_methylation:
                gene_methylation[gene_id] = []
            try:
                gene_methylation[gene_id].append(self.methylation.loc[probe].values)
            except KeyError:
                continue
        
        # Calculate average methylation per gene
        gene_methylation_avg = {}
        for gene_id, values_list in gene_methylation.items():
            if len(values_list) > 0:
                try:
                    stacked = np.stack(values_list, axis=0)
                    gene_methylation_avg[gene_id] = np.mean(stacked, axis=0)
                except:
                    continue
        
        if gene_methylation_avg:
            gene_ids_mapped = list(gene_methylation_avg.keys())
            sample_ids = self.methylation.columns.tolist()
            
            methyl_by_gene = pd.DataFrame(
                index=gene_ids_mapped,
                columns=sample_ids
            )
            
            for gene_id in gene_ids_mapped:
                methyl_by_gene.loc[gene_id] = gene_methylation_avg[gene_id]
            
            print(f"Mapped methylation: {methyl_by_gene.shape}")
            return methyl_by_gene
        
        print("WARNING: No methylation data could be mapped to genes")
        return None
        
    def create_cna_features(self):
        """Create CNA-related features"""
        print("Creating CNA features...")
        
        features = pd.DataFrame(index=self.cna.index)
        
        # Basic CNA features
        features['cna_mean'] = self.cna.mean(axis=1)
        features['cna_std'] = self.cna.std(axis=1)
        features['cna_max'] = self.cna.max(axis=1)
        features['cna_min'] = self.cna.min(axis=1)
        features['amp_frequency'] = (self.cna > config.CNA_THRESHOLD_AMP).mean(axis=1)
        features['del_frequency'] = (self.cna < config.CNA_THRESHOLD_DEL).mean(axis=1)
        features['cna_variance'] = self.cna.var(axis=1)
        features['cna_magnitude_mean'] = np.abs(self.cna).mean(axis=1)
        features['cna_magnitude_max'] = np.abs(self.cna).max(axis=1)
        features['cna_alteration_freq'] = (np.abs(self.cna) > 0.1).mean(axis=1)
        
        # Advanced CNA features
        features['cna_complexity'] = (self.cna != 0).sum(axis=1) / len(self.cna.columns)
        features['cna_skewness'] = self.cna.skew(axis=1)
        features['cna_kurtosis'] = self.cna.kurtosis(axis=1)
        
        return features
    
    def create_expression_features(self):
        """Create expression-related features"""
        print("Creating expression features...")
        
        features = pd.DataFrame(index=self.expression.index)
        
        # Basic expression features
        features['expr_mean'] = self.expression.mean(axis=1)
        features['expr_std'] = self.expression.std(axis=1)
        features['expr_max'] = self.expression.max(axis=1)
        features['expr_min'] = self.expression.min(axis=1)
        features['expr_variance'] = self.expression.var(axis=1)
        features['expr_range'] = features['expr_max'] - features['expr_min']
        features['expr_cv'] = features['expr_std'] / (features['expr_mean'] + 1e-10)
        
        # Advanced expression features
        features['expr_skewness'] = self.expression.skew(axis=1)
        features['expr_kurtosis'] = self.expression.kurtosis(axis=1)
        features['expr_median'] = self.expression.median(axis=1)
        features['expr_mad'] = (self.expression - self.expression.median(axis=1, numeric_only=True).values.reshape(-1, 1)).abs().mean(axis=1)
        
        return features
    
    def create_methylation_features(self):
        """Create methylation-related features FOR GENES"""
        print("Creating methylation features...")
        
        if self.methylation is None or self.methylation.empty:
            print("Methylation data not available, skipping...")
            return pd.DataFrame(index=self.cna.index)
        
        # Convert to numeric
        self.methylation = self.methylation.apply(pd.to_numeric, errors='coerce')
        
        # Only keep genes that are in CNA
        common_genes = set(self.methylation.index) & set(self.cna.index)
        if len(common_genes) < len(self.methylation.index):
            self.methylation = self.methylation.loc[list(common_genes)]
            print(f"Filtered methylation to {len(common_genes)} genes in CNA")
        
        if self.methylation.empty:
            return pd.DataFrame(index=self.cna.index)
        
        features = pd.DataFrame(index=self.methylation.index)
        
        # Basic methylation features
        features['methyl_mean'] = self.methylation.mean(axis=1)
        features['methyl_std'] = self.methylation.std(axis=1)
        features['methyl_max'] = self.methylation.max(axis=1)
        features['methyl_min'] = self.methylation.min(axis=1)
        features['methyl_high_freq'] = (self.methylation > config.METHYLATION_HIGH).mean(axis=1)
        features['methyl_low_freq'] = (self.methylation < config.METHYLATION_LOW).mean(axis=1)
        features['methyl_range'] = features['methyl_max'] - features['methyl_min']
        
        # Advanced methylation features
        features['methyl_median'] = self.methylation.median(axis=1)
        features['methyl_mad'] = (self.methylation - self.methylation.median(axis=1, numeric_only=True).values.reshape(-1, 1)).abs().mean(axis=1)
        features['methyl_skewness'] = self.methylation.skew(axis=1)
        features['methyl_kurtosis'] = self.methylation.kurtosis(axis=1)
        
        # Fill NaN with 0
        features = features.fillna(0)
        
        return features
    
    def create_interaction_features(self):
        """Create CNA-Expression and CNA-Methylation interaction features"""
        print("Creating interaction features...")
        
        features = pd.DataFrame(index=self.cna.index)
        features['cna_expr_correlation'] = 0
        features['cna_expr_consistency'] = 0
        features['cna_methyl_correlation'] = 0
        
        # Ensure both dataframes have same samples
        common_samples = list(set(self.cna.columns) & set(self.expression.columns))
        
        if len(common_samples) == 0:
            print("WARNING: No common samples between CNA and Expression!")
            return features
        
        # Subset to common samples
        cna_subset = self.cna[common_samples]
        expr_subset = self.expression[common_samples]
        
        cna_subset = cna_subset.apply(pd.to_numeric, errors='coerce')
        expr_subset = expr_subset.apply(pd.to_numeric, errors='coerce')
        
        cna_expr_corr = []
        for gene in self.cna.index:
            if gene in self.expression.index:
                cna_values = cna_subset.loc[gene].values.astype(float)
                expr_values = expr_subset.loc[gene].values.astype(float)
                
                mask = ~(np.isnan(cna_values) | np.isnan(expr_values))
                cna_clean = cna_values[mask]
                expr_clean = expr_values[mask]
                
                if len(cna_clean) > 1 and np.std(cna_clean) > 0 and np.std(expr_clean) > 0:
                    corr, _ = pearsonr(cna_clean, expr_clean)
                    cna_expr_corr.append(corr if not np.isnan(corr) else 0)
                else:
                    cna_expr_corr.append(0)
            else:
                cna_expr_corr.append(0)
        
        features['cna_expr_correlation'] = cna_expr_corr
        
        # CNA-Expression consistency
        cna_signed = np.sign(cna_subset.values)
        expr_signed = np.sign(expr_subset.values - expr_subset.values.mean(axis=1, keepdims=True))
        consistency = (cna_signed == expr_signed).mean(axis=1)
        features['cna_expr_consistency'] = consistency
        
        # Advanced: CNA-Expression discrepancy
        features['cna_expr_discrepancy'] = np.abs(
            np.sign(cna_subset).mean(axis=1) - 
            np.sign(expr_subset - expr_subset.mean(axis=1)).mean(axis=1)
        )
        
        # CNA-Methylation interaction if methylation available
        if self.methylation is not None and not self.methylation.empty:
            common_samples_methyl = list(set(cna_subset.columns) & set(self.methylation.columns))
            if len(common_samples_methyl) > 0:
                cna_subset2 = cna_subset[common_samples_methyl]
                methyl_subset = self.methylation[common_samples_methyl]
                
                cna_subset2 = cna_subset2.apply(pd.to_numeric, errors='coerce')
                methyl_subset = methyl_subset.apply(pd.to_numeric, errors='coerce')
                
                cna_methyl_corr = []
                for gene in self.cna.index:
                    if gene in self.methylation.index:
                        cna_values = cna_subset2.loc[gene].values.astype(float)
                        methyl_values = methyl_subset.loc[gene].values.astype(float)
                        
                        mask = ~(np.isnan(cna_values) | np.isnan(methyl_values))
                        cna_clean = cna_values[mask]
                        methyl_clean = methyl_values[mask]
                        
                        if len(cna_clean) > 1 and np.std(cna_clean) > 0 and np.std(methyl_clean) > 0:
                            corr, _ = pearsonr(cna_clean, methyl_clean)
                            cna_methyl_corr.append(corr if not np.isnan(corr) else 0)
                        else:
                            cna_methyl_corr.append(0)
                    else:
                        cna_methyl_corr.append(0)
                features['cna_methyl_correlation'] = cna_methyl_corr
        
        return features
    
    def create_biological_features(self):
        """Create biological context features"""
        print("Creating biological features...")
        
        features = pd.DataFrame(index=self.cna.index)
        
        gencode = pd.read_csv(config.GENCODE_PROCESSED)
        gene_types = dict(zip(gencode['gene_name'], gencode['gene_type']))
        
        features['is_protein_coding'] = [
            1 if gene_types.get(gene, 'unknown') == 'protein_coding' else 0
            for gene in self.cna.index
        ]
        
        gene_lengths = dict(zip(gencode['gene_name'], gencode['end'] - gencode['start']))
        features['gene_length'] = [
            gene_lengths.get(gene, np.nan) 
            for gene in self.cna.index
        ]
        features['gene_length'] = features['gene_length'].fillna(features['gene_length'].mean())
        
        # Advanced: Known oncogenes
        known_oncogenes = ['BRCA1', 'BRCA2', 'TP53', 'MYC', 'EGFR', 'KRAS', 'PIK3CA', 'ERBB2', 'PTEN']
        features['is_oncogene'] = [
            1 if gene in known_oncogenes else 0
            for gene in self.cna.index
        ]
        
        return features
    
    def build_labels_for_features(self):
        """Create labels using ClinVar AND known cancer genes"""
        print("Building labels for features...")
        
        gencode = pd.read_csv(config.GENCODE_PROCESSED)
        
        symbol_to_ensembl = {}
        for _, row in gencode.iterrows():
            gene_id = row['gene_id'].split('.')[0]
            gene_name = row['gene_name']
            if gene_name not in symbol_to_ensembl:
                symbol_to_ensembl[gene_name] = gene_id
        
        print(f"Total gene symbols in GENCODE: {len(symbol_to_ensembl)}")
        
        clinvar = pd.read_csv(config.CLINVAR_PROCESSED, low_memory=False)
        
        pathogenic_symbols = clinvar[
            clinvar['clinical_significance'].str.contains('Pathogenic', na=False)
        ]['gene_symbol'].unique()
        print(f"Pathogenic genes from ClinVar: {len(pathogenic_symbols)}")
        
        pathogenic_ensembl = []
        for symbol in pathogenic_symbols:
            if symbol in symbol_to_ensembl:
                pathogenic_ensembl.append(symbol_to_ensembl[symbol])
        print(f"ClinVar genes mapped to Ensembl: {len(pathogenic_ensembl)}")
        
        cancer_symbols = [
            'BRCA1', 'BRCA2', 'TP53', 'PTEN', 'PIK3CA', 'ERBB2', 'MYC', 'CCND1', 
            'FGFR1', 'FGFR2', 'AKT1', 'AKT2', 'AKT3', 'CDK4', 'CDK6', 'CDKN1A',
            'CDKN1B', 'CDKN2A', 'CDKN2B', 'EGFR', 'KRAS', 'NRAS', 'HRAS', 'BRAF',
            'MAP2K1', 'MAP2K4', 'NF1', 'NF2', 'RB1', 'SMAD4', 'VHL', 'WT1'
        ]
        
        cancer_ensembl = []
        for symbol in cancer_symbols:
            if symbol in symbol_to_ensembl:
                cancer_ensembl.append(symbol_to_ensembl[symbol])
        print(f"Known cancer genes mapped to Ensembl: {len(cancer_ensembl)}")
        
        all_positive_ensembl = set(pathogenic_ensembl) | set(cancer_ensembl)
        print(f"Total positive genes (Ensembl IDs): {len(all_positive_ensembl)}")
        
        labels = []
        for gene_id in self.features.index:
            base_id = gene_id.split('.')[0]
            if base_id in all_positive_ensembl:
                labels.append(1)
            else:
                labels.append(0)
        
        self.labels = pd.Series(labels, index=self.features.index)
        
        print(f"Positive labels: {sum(labels)}")
        print(f"Negative labels: {len(labels) - sum(labels)}")
        
        return self.labels
    
    def create_all_features(self):
        """Create all features and combine them"""
        print("\n" + "="*60)
        print("CREATING ALL FEATURES")
        print("="*60)
        
        self.load_processed_data()
        
        # MAP METHYLATION TO GENES
        if self.methylation is not None and not self.methylation.empty:
            self.methylation_by_gene = self.map_methylation_to_genes_location_based()
            if self.methylation_by_gene is not None:
                self.methylation = self.methylation_by_gene
                print(f"Methylation now has gene-level data: {self.methylation.shape}")
        
        cna_features = self.create_cna_features()
        expr_features = self.create_expression_features()
        methyl_features = self.create_methylation_features()
        interaction_features = self.create_interaction_features()
        bio_features = self.create_biological_features()
        
        # Combine all features
        self.features = pd.concat([
            cna_features,
            expr_features,
            methyl_features,
            interaction_features,
            bio_features
        ], axis=1)
        
        self.features = self.features.dropna(how='all')
        self.features = self.features.fillna(self.features.mean())
        
        # Ensure all features are numeric
        for col in self.features.columns:
            if self.features[col].dtype == 'bool':
                self.features[col] = self.features[col].astype(int)
        
        print(f"Final feature matrix shape: {self.features.shape}")
        print(f"Number of features: {len(self.features.columns)}")
        print(f"Feature columns: {self.features.columns.tolist()}")
        
        # Build labels
        self.build_labels_for_features()
        
        self.features.to_csv(config.MULTI_OMICS_FEATURES)
        self.labels.to_csv(config.LABELS_FILE)
        
        final_dataset = self.features.copy()
        final_dataset['label'] = self.labels
        final_dataset.to_csv(config.FINAL_DATASET)
        
        print("\nFeature engineering complete!")
        return self.features, self.labels