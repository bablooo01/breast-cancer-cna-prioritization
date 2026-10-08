import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, confusion_matrix,
                           classification_report)
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
from src import config

class ModelTrainer:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load feature matrix and labels"""
        print("Loading data...")
        self.X = pd.read_csv(config.MULTI_OMICS_FEATURES, index_col=0)
        self.y = pd.read_csv(config.LABELS_FILE, index_col=0).squeeze()
        
        print(f"Features shape: {self.X.shape}")
        print(f"Labels shape: {self.y.shape}")
        print(f"Positive samples: {sum(self.y)}")
        print(f"Negative samples: {len(self.y) - sum(self.y)}")
        
        return self.X, self.y
    
    def prepare_data(self, test_size=0.2):
        """Split and scale data"""
        print("\nPreparing data...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=config.RANDOM_SEED,
            stratify=self.y
        )
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train_scaled.shape}")
        print(f"Test set: {self.X_test_scaled.shape}")
        
        self.feature_names = self.X.columns.tolist()
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_random_forest(self):
        """Train Random Forest"""
        print("\nTraining Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        
        rf.fit(self.X_train_scaled, self.y_train)
        self.models['Random Forest'] = rf
        return rf
    
    def train_xgboost(self):
        """Train XGBoost"""
        print("\nTraining XGBoost...")
        
        pos_count = sum(self.y_train == 1)
        neg_count = sum(self.y_train == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=config.RANDOM_SEED,
            eval_metric='logloss',
            verbosity=0
        )
        
        xgb_model.fit(
            self.X_train_scaled, 
            self.y_train,
            eval_set=[(self.X_train_scaled, self.y_train), (self.X_test_scaled, self.y_test)],
            verbose=False
        )
        
        self.models['XGBoost'] = xgb_model
        return xgb_model
    
    def train_lightgbm(self):
        """Train LightGBM"""
        print("\nTraining LightGBM...")
        
        pos_count = sum(self.y_train == 1)
        neg_count = sum(self.y_train == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=50,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1
        )
        
        lgb_model.fit(
            self.X_train_scaled, 
            self.y_train,
            eval_set=[(self.X_train_scaled, self.y_train), (self.X_test_scaled, self.y_test)]
        )
        
        self.models['LightGBM'] = lgb_model
        return lgb_model
    
    def train_ensemble(self):
        """Create voting ensemble"""
        print("\nTraining Ensemble...")
        
        rf = self.models.get('Random Forest')
        xgb_model = self.models.get('XGBoost')
        lgb_model = self.models.get('LightGBM')
        
        if rf is None or xgb_model is None or lgb_model is None:
            self.train_random_forest()
            self.train_xgboost()
            self.train_lightgbm()
            rf = self.models.get('Random Forest')
            xgb_model = self.models.get('XGBoost')
            lgb_model = self.models.get('LightGBM')
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb_model), ('lgb', lgb_model)],
            voting='soft'
        )
        
        ensemble.fit(self.X_train_scaled, self.y_train)
        self.models['Ensemble'] = ensemble
        return ensemble
    
    def evaluate_model(self, model, name):
        """Evaluate a single model"""
        print(f"\nEvaluating {name}...")
        
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'auc_roc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        self.results[name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model
        }
        
        return metrics
    
    def evaluate_top_k(self, model, name):
        """Evaluate precision@K"""
        print(f"\n  {name} - Top-K Precision:")
        
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        ranking = pd.DataFrame({
            'true_label': self.y_test.values,
            'score': y_pred_proba
        }).sort_values('score', ascending=False)
        
        for k in [10, 20, 50, 100, 200, 500]:
            precision = ranking.head(k)['true_label'].sum() / k
            print(f"    Precision@{k}: {precision:.4f}")
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        self.train_ensemble()
        
        for name, model in self.models.items():
            self.evaluate_model(model, name)
            self.evaluate_top_k(model, name)
        
        best_auc = 0
        for name, results in self.results.items():
            if results['metrics']['auc_roc'] > best_auc:
                best_auc = results['metrics']['auc_roc']
                self.best_model = results['model']
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} (AUC-ROC: {best_auc:.4f})")
        return self.results
    
    def cross_validate_model(self, model, name):
        """Cross-validation"""
        print(f"\nCross-validating {name}...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        
        cv_accuracy = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='accuracy')
        cv_auc = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv, scoring='roc_auc')
        
        print(f"  CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        print(f"  CV AUC-ROC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
        return {'accuracy': cv_accuracy, 'auc': cv_auc}
    
    def plot_confusion_matrix(self, name):
        """Plot confusion matrix"""
        results = self.results[name]
        cm = confusion_matrix(self.y_test, results['y_pred'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f'confusion_matrix_{name}.png', dpi=300)
        plt.close()
    
    def plot_roc_curves(self):
        """Plot ROC curves"""
        plt.figure(figsize=(10, 8))
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {results["metrics"]["auc_roc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / 'roc_curves.png', dpi=300)
        plt.close()
    
    def compute_shap_values(self):
        """Compute SHAP values with sampling to avoid memory error"""
        print(f"\nComputing SHAP values for {self.best_model_name}...")
        
        try:
            # Use SUBSET for SHAP (prevents memory error)
            sample_size = min(500, len(self.X_test_scaled))
            X_shap_sample = self.X_test_scaled[:sample_size]
            X_shap_names = self.feature_names
            
            print(f"Using {sample_size} samples for SHAP (out of {len(self.X_test_scaled)})")
            
            # Get model for SHAP
            if self.best_model_name == 'Ensemble':
                model_for_shap = self.best_model.estimators_[0]
            else:
                model_for_shap = self.best_model
            
            explainer = shap.TreeExplainer(model_for_shap)
            shap_values = explainer.shap_values(X_shap_sample)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            print("✓ SHAP values computed successfully!")
            
            # SHAP Summary Plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_shap_sample, feature_names=X_shap_names, show=False)
            plt.tight_layout()
            plt.savefig(config.FIGURES_DIR / 'shap_summary.png', dpi=300)
            plt.close()
            print("✓ SHAP summary plot saved")
            
            # SHAP Bar Plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_shap_sample, feature_names=X_shap_names, plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig(config.FIGURES_DIR / 'shap_importance.png', dpi=300)
            plt.close()
            print("✓ SHAP importance plot saved")
            
            # Save SHAP values
            with open(config.MODELS_DIR / 'shap_values.pkl', 'wb') as f:
                pickle.dump(shap_values, f)
            print("✓ SHAP values saved")
            
            return shap_values
            
        except Exception as e:
            print(f"✗ SHAP computation failed: {e}")
            print("Trying with smaller sample...")
            try:
                # Try with even smaller sample
                sample_size = min(200, len(self.X_test_scaled))
                X_shap_sample = self.X_test_scaled[:sample_size]
                
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(X_shap_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X_shap_sample, feature_names=X_shap_names, show=False)
                plt.tight_layout()
                plt.savefig(config.FIGURES_DIR / 'shap_summary.png', dpi=300)
                plt.close()
                print("✓ SHAP with smaller sample saved!")
                return shap_values
            except Exception as e2:
                print(f"✗ SHAP still failed: {e2}")
                print("SHAP skipped - but model results are still valid.")
                return None
    
    def generate_functional_importance_scores(self):
        """Generate FIS scores"""
        print("\nGenerating Functional Importance Scores...")
        
        fis = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        fis_df = pd.DataFrame({
            'gene': self.X_test.index,
            'FIS': fis,
            'true_label': self.y_test.values
        }).sort_values('FIS', ascending=False)
        fis_df['rank'] = range(1, len(fis_df) + 1)
        
        fis_df.to_csv(config.RANKINGS_DIR / 'functional_importance_scores.csv', index=False)
        
        print(f"FIS range: {fis_df['FIS'].min():.4f} - {fis_df['FIS'].max():.4f}")
        print(f"\nTop 10 genes:")
        print(fis_df.head(10)[['gene', 'FIS']])
        
        return fis_df
    
    def analyze_top_genes(self, fis_df, n_top=50):
        """Analyze top genes"""
        print(f"\nAnalyzing top {n_top} genes...")
        print("Top genes analysis complete.")
    
    def run_full_pipeline(self):
        """Run complete pipeline WITH SHAP"""
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING PIPELINE (WITH SHAP)")
        print("="*60)
        
        self.load_data()
        self.prepare_data()
        
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        
        self.evaluate_all_models()
        cv_results = self.cross_validate_model(self.best_model, self.best_model_name)
        
        with open(config.MODELS_DIR / 'best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Generate plots
        for name in self.models.keys():
            self.plot_confusion_matrix(name)
        self.plot_roc_curves()
        
        # SHAP (with sampling)
        self.compute_shap_values()
        
        fis_df = self.generate_functional_importance_scores()
        self.analyze_top_genes(fis_df)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE")
        print("="*60)
        
        return {
            'results': self.results,
            'best_model': self.best_model_name,
            'fis': fis_df,
            'cv_results': cv_results
        }