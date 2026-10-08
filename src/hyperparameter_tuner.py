import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src import config
import pickle

class HyperparameterTuner:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_models = {}
        
    def load_data(self):
        """Load feature matrix and labels"""
        print("Loading data for tuning...")
        self.X = pd.read_csv(config.MULTI_OMICS_FEATURES, index_col=0)
        self.y = pd.read_csv(config.LABELS_FILE, index_col=0).squeeze()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=config.RANDOM_SEED,
            stratify=self.y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def tune_random_forest(self):
        """Tune Random Forest - FAST VERSION"""
        print("\n" + "="*60)
        print("TUNING RANDOM FOREST (FAST)")
        print("="*60)
        
        # SMALLER GRID FOR SPEED
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        rf = RandomForestClassifier(
            class_weight='balanced',
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.RANDOM_SEED)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV AUC-ROC: {grid_search.best_score_:.4f}")
        
        y_pred_proba = grid_search.best_estimator_.predict_proba(self.X_test)[:, 1]
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f"Test AUC-ROC: {test_auc:.4f}")
        
        self.best_models['RandomForest'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def tune_xgboost(self):
        """Tune XGBoost - FAST VERSION"""
        print("\n" + "="*60)
        print("TUNING XGBOOST (FAST)")
        print("="*60)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        pos_count = sum(self.y_train == 1)
        neg_count = sum(self.y_train == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_weight,
            random_state=config.RANDOM_SEED,
            eval_metric='logloss',
            verbosity=0
        )
        
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.RANDOM_SEED)
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV AUC-ROC: {grid_search.best_score_:.4f}")
        
        y_pred_proba = grid_search.best_estimator_.predict_proba(self.X_test)[:, 1]
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f"Test AUC-ROC: {test_auc:.4f}")
        
        self.best_models['XGBoost'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def tune_lightgbm(self):
        """Tune LightGBM - FAST VERSION"""
        print("\n" + "="*60)
        print("TUNING LIGHTGBM (FAST)")
        print("="*60)
        
        param_grid = {
            'n_estimators': [100, 200],
            'num_leaves': [31, 50],
            'max_depth': [6, 8],
            'learning_rate': [0.05, 0.1]
        }
        
        pos_count = sum(self.y_train == 1)
        neg_count = sum(self.y_train == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        lgb_model = lgb.LGBMClassifier(
            scale_pos_weight=scale_weight,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1
        )
        
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.RANDOM_SEED)
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV AUC-ROC: {grid_search.best_score_:.4f}")
        
        y_pred_proba = grid_search.best_estimator_.predict_proba(self.X_test)[:, 1]
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f"Test AUC-ROC: {test_auc:.4f}")
        
        self.best_models['LightGBM'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def create_ensemble(self):
        """Create stacking ensemble"""
        print("\n" + "="*60)
        print("CREATING STACKING ENSEMBLE")
        print("="*60)
        
        estimators = [
            ('rf', self.best_models['RandomForest']),
            ('xgb', self.best_models['XGBoost']),
            ('lgb', self.best_models['LightGBM'])
        ]
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000),
            cv=2,
            n_jobs=-1
        )
        
        ensemble.fit(self.X_train, self.y_train)
        
        y_pred_proba = ensemble.predict_proba(self.X_test)[:, 1]
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f"Ensemble Test AUC-ROC: {test_auc:.4f}")
        
        self.best_models['Ensemble'] = ensemble
        return ensemble
    
    def save_best_model(self, model, name):
        """Save best model"""
        with open(config.MODELS_DIR / f'{name}_tuned.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved tuned {name} to {config.MODELS_DIR / f'{name}_tuned.pkl'}")
    
    def run_tuning(self):
        """Run complete tuning pipeline - FAST VERSION"""
        print("\n" + "="*60)
        print("STARTING HYPERPARAMETER TUNING (FAST)")
        print("="*60)
        
        self.load_data()
        
        # Tune all models
        rf = self.tune_random_forest()
        self.save_best_model(rf, 'RandomForest')
        
        xgb_model = self.tune_xgboost()
        self.save_best_model(xgb_model, 'XGBoost')
        
        lgb_model = self.tune_lightgbm()
        self.save_best_model(lgb_model, 'LightGBM')
        
        ensemble = self.create_ensemble()
        self.save_best_model(ensemble, 'Ensemble')
        
        print("\n" + "="*60)
        print("TUNING SUMMARY")
        print("="*60)
        for name, model in self.best_models.items():
            y_pred = model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_pred)
            print(f"{name:15s}: AUC-ROC = {auc:.4f}")
        
        return self.best_models