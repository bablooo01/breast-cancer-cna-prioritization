import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_selector import FeatureSelector
from src.hyperparameter_tuner import HyperparameterTuner
from src.clinical_validator import ClinicalValidator
from src.plot_generator import PlotGenerator
from src import config

def run_feature_selection():
    print("\n" + "="*60)
    print("STEP 1: FEATURE SELECTION")
    print("="*60)
    
    selector = FeatureSelector()
    selector.load_data()
    features, _ = selector.select_by_importance(threshold='median')
    selector.save_selected_features()
    
    return features

def run_hyperparameter_tuning():
    print("\n" + "="*60)
    print("STEP 2: HYPERPARAMETER TUNING")
    print("="*60)
    
    tuner = HyperparameterTuner()
    tuner.load_data()
    
    best_rf = tuner.tune_random_forest()
    tuner.save_best_model(best_rf, 'RandomForest')
    
    return best_rf

def run_clinical_validation():
    print("\n" + "="*60)
    print("STEP 3: CLINICAL VALIDATION")
    print("="*60)
    
    validator = ClinicalValidator()
    validator.run_validation()

def run_plots():
    print("\n" + "="*60)
    print("STEP 4: GENERATE FIGURES")
    print("="*60)
    
    plotter = PlotGenerator()
    plotter.generate_all_plots()

def main():
    print("="*70)
    print(" PUBLICATION PIPELINE")
    print("="*70)
    
    # Step 1: Feature Selection
    run_feature_selection()
    
    # Step 2: Hyperparameter Tuning (uncomment if needed)
    # run_hyperparameter_tuning()
    
    # Step 3: Clinical Validation
    run_clinical_validation()
    
    # Step 4: Generate Figures
    run_plots()
    
    print("\n" + "="*70)
    print(" PUBLICATION PIPELINE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()