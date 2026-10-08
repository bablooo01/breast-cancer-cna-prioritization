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
    print(" BREAST CANCER CNA PRIORITIZATION FRAMEWORK")
    print("="*70)
    
    print("\n[1/4] Loading and Preprocessing Data...")
    preprocessor = Preprocessor()
    preprocessor.run_full_preprocessing()
    
    print("\n[2/4] Feature Engineering...")
    engineer = FeatureEngineer()
    features, labels = engineer.create_all_features()
    
    print("\n[3/4] Training Models...")
    trainer = ModelTrainer()
    results = trainer.run_full_pipeline()
    
    print("\n[4/4] Results Summary...")
    print(f"\nBest model: {results['best_model']}")
    print("\nTop 10 genes with highest FIS:")
    print(results['fis'].head(10)[['gene', 'FIS']])
    
    print("\n" + "="*70)
    print(" COMPLETE! Check results/ folder for outputs")
    print("="*70)

if __name__ == "__main__":
    main()