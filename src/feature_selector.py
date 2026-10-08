import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LassoCV
from src import config

class FeatureSelector:
    def __init__(self):
        self.X = None
        self.y = None
        self.selected_features = None
        
    def load_data(self):
        """Load feature matrix and labels"""
        print("Loading data for feature selection...")
        self.X = pd.read_csv(config.MULTI_OMICS_FEATURES, index_col=0)
        self.y = pd.read_csv(config.LABELS_FILE, index_col=0).squeeze()
        print(f"Features shape: {self.X.shape}")
        print(f"Labels shape: {self.y.shape}")
        return self.X, self.y
    
    def select_by_importance(self, threshold='median'):
        """Select features using Random Forest importance"""
        print(f"\nSelecting features with threshold='{threshold}'...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        rf.fit(self.X, self.y)
        
        # Get importances
        importances = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features by importance:")
        print(importances.head(10))
        
        # Select features
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        X_selected = selector.transform(self.X)
        selected_features = self.X.columns[selector.get_support()].tolist()
        
        print(f"\nSelected {len(selected_features)} features from {len(self.X.columns)}")
        print(f"Selected features: {selected_features}")
        
        self.selected_features = selected_features
        return selected_features, X_selected
    
    def select_by_rfe(self, n_features=15):
        """Select features using Recursive Feature Elimination"""
        print(f"\nSelecting {n_features} features using RFE...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        
        selector = RFE(rf, n_features_to_select=n_features, step=1)
        selector.fit(self.X, self.y)
        
        selected_features = self.X.columns[selector.support_].tolist()
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        self.selected_features = selected_features
        return selected_features
    
    def select_by_lasso(self):
        """Select features using Lasso (L1 regularization)"""
        print("\nSelecting features using Lasso...")
        
        lasso = LassoCV(cv=5, random_state=config.RANDOM_SEED)
        lasso.fit(self.X, self.y)
        
        selected_features = self.X.columns[lasso.coef_ != 0].tolist()
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        self.selected_features = selected_features
        return selected_features
    
    def save_selected_features(self):
        """Save selected features to file"""
        if self.selected_features:
            pd.Series(self.selected_features).to_csv(
                config.PROCESSED_DIR / 'selected_features.csv',
                index=False,
                header=['feature']
            )
            print(f"Saved selected features to {config.PROCESSED_DIR / 'selected_features.csv'}")