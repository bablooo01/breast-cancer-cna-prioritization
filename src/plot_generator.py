import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap
import pickle
from src import config

class PlotGenerator:
    def __init__(self):
        self.fis_df = None
        self.model = None
        self.shap_values = None
        
    def load_data(self):
        """Load results for plotting"""
        print("Loading results...")
        self.fis_df = pd.read_csv(
            config.RANKINGS_DIR / 'functional_importance_scores_with_symbols.csv'
        )
        
        try:
            with open(config.MODELS_DIR / 'best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except:
            print("Model not found")
        
        try:
            with open(config.MODELS_DIR / 'shap_values.pkl', 'rb') as f:
                self.shap_values = pickle.load(f)
        except:
            print("SHAP values not found")
    
    def plot_fis_distribution(self):
        """Plot FIS score distribution"""
        print("\nPlotting FIS distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.fis_df['FIS'], bins=50, color='steelblue', alpha=0.7)
        axes[0].axvline(self.fis_df['FIS'].median(), color='red', linestyle='--', label=f'Median: {self.fis_df["FIS"].median():.3f}')
        axes[0].set_xlabel('FIS Score')
        axes[0].set_ylabel('Number of Genes')
        axes[0].set_title('Distribution of FIS Scores')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by label
        data_by_label = [self.fis_df[self.fis_df['true_label'] == 0]['FIS'],
                        self.fis_df[self.fis_df['true_label'] == 1]['FIS']]
        axes[1].boxplot(data_by_label, labels=['Unknown', 'Cancer Gene'])
        axes[1].set_xlabel('Gene Label')
        axes[1].set_ylabel('FIS Score')
        axes[1].set_title('FIS Scores by Gene Label')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / 'fis_distribution.png', dpi=300)
        plt.close()
        print(f"Saved to {config.FIGURES_DIR / 'fis_distribution.png'}")
    
    def plot_top_genes(self, n=20):
        """Plot top N genes"""
        print(f"\nPlotting top {n} genes...")
        
        top_genes = self.fis_df.head(n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#FF6B6B' if label == 0 else '#51CF66' for label in top_genes['true_label']]
        ax.barh(top_genes['gene_symbol'], top_genes['FIS'], color=colors)
        
        ax.set_xlabel('FIS Score')
        ax.set_ylabel('Gene Symbol')
        ax.set_title(f'Top {n} Prioritized Genes')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#51CF66', label='Cancer Gene'),
            Patch(facecolor='#FF6B6B', label='Unknown')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / 'top_genes.png', dpi=300)
        plt.close()
        print(f"Saved to {config.FIGURES_DIR / 'top_genes.png'}")
    
    def plot_feature_importance(self):
        """Plot feature importance from SHAP"""
        print("\nPlotting feature importance...")
        
        if self.shap_values is not None:
            # SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, show=False)
            plt.tight_layout()
            plt.savefig(config.FIGURES_DIR / 'shap_summary_final.png', dpi=300)
            plt.close()
            print(f"Saved to {config.FIGURES_DIR / 'shap_summary_final.png'}")
        else:
            print("SHAP values not available")
    
    def generate_all_plots(self):
        """Generate all publication-ready plots"""
        print("\n" + "="*60)
        print("GENERATING PUBLICATION-READY PLOTS")
        print("="*60)
        
        self.load_data()
        self.plot_fis_distribution()
        self.plot_top_genes(20)
        self.plot_feature_importance()
        
        print("\nAll plots generated!")