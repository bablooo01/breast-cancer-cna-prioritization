import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src import config

def main():
    print("="*70)
    print(" FINAL COMPLETE BREAST CANCER CNA PRIORITIZATION")
    print(" MAX METHYLATION PROBES + 42 FEATURES + SHAP")
    print("="*70)
    
    print("\n[1/4] Loading and Preprocessing Data...")
    preprocessor = Preprocessor()
    preprocessor.run_full_preprocessing()
    
    print("\n[2/4] Feature Engineering (42 Features)...")
    engineer = FeatureEngineer()
    features, labels = engineer.create_all_features()
    
    print("\n[3/4] Training Models (WITH SHAP)...")
    trainer = ModelTrainer()
    results = trainer.run_full_pipeline()
    
    print("\n[4/4] Results Summary...")
    print(f"\nBest model: {results['best_model']}")
    print(f"AUC-ROC: {results['results'][results['best_model']]['metrics']['auc_roc']:.4f}")
    print("\nTop 10 genes with highest FIS:")
    print(results['fis'].head(10)[['gene', 'FIS']])
    
    print("\n" + "="*70)
    print(" COMPLETE!")
    print(" Features: 42")
    print(f" Methylation Probes: Check processed data")
    print(" SHAP: Included (sampled for memory efficiency)")
    print("="*70)

if __name__ == "__main__":
    main()