#!/usr/bin/env python3
"""
Model Training Module for Wastewater Quality Prediction
This module trains and saves ML models for predicting outlet water quality parameters.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_clean_data, prepare_model_data, get_sample_data
import warnings
warnings.filterwarnings('ignore')

class WastewaterQualityModel:
    """Wastewater quality prediction model."""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = None
        self.target_columns = None
        self.feature_importance = None
        
    def train(self, X, y, feature_columns, target_columns):
        """Train the model."""
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # Initialize model
        if self.model_type == 'random_forest':
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model = MultiOutputRegressor(base_model)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Inverse transform predictions for evaluation
        y_train_orig = self.scaler_y.inverse_transform(y_train)
        y_test_orig = self.scaler_y.inverse_transform(y_test)
        y_pred_train_orig = self.scaler_y.inverse_transform(y_pred_train)
        y_pred_test_orig = self.scaler_y.inverse_transform(y_pred_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train_orig, y_pred_train_orig, multioutput='uniform_average')
        test_r2 = r2_score(y_test_orig, y_pred_test_orig, multioutput='uniform_average')
        train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig, multioutput='uniform_average')
        test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig, multioutput='uniform_average')
        
        print(f"Model Performance:")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Train MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Calculate feature importance
        if hasattr(self.model.estimators_[0], 'feature_importances_'):
            # Average feature importance across all output targets
            importances = np.mean([estimator.feature_importances_ 
                                 for estimator in self.model.estimators_], axis=0)
            self.feature_importance = dict(zip(feature_columns, importances))
            
            print("\nFeature Importance:")
            for feature, importance in sorted(self.feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance:.4f}")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2, 
            'train_mae': train_mae,
            'test_mae': test_mae,
            'X_test': X_test,
            'y_test_orig': y_test_orig,
            'y_pred_test_orig': y_pred_test_orig
        }
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale input
        X_scaled = self.scaler_X.transform(X)
        
        # Make prediction
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Convert to DataFrame with proper column names
        return pd.DataFrame(y_pred, columns=self.target_columns)
    
    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        model_instance = cls()
        model_instance.model = model_data['model']
        model_instance.scaler_X = model_data['scaler_X']
        model_instance.scaler_y = model_data['scaler_y']
        model_instance.feature_columns = model_data['feature_columns']
        model_instance.target_columns = model_data['target_columns']
        model_instance.feature_importance = model_data.get('feature_importance')
        
        return model_instance

def plot_feature_importance(feature_importance, save_path='feature_importance.png'):
    """Plot feature importance."""
    if feature_importance is None:
        print("No feature importance data available.")
        return
    
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features, palette='viridis')
    plt.title('Feature Importance for Wastewater Quality Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")

def plot_actual_vs_predicted(y_test, y_pred, target_columns, save_path='actual_vs_predicted.png'):
    """Plot actual vs predicted values."""
    n_targets = len(target_columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for i, target in enumerate(target_columns):
        axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.6)
        axes[i].plot([y_test[:, i].min(), y_test[:, i].max()], 
                    [y_test[:, i].min(), y_test[:, i].max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{target}')
        
        # Calculate R²
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Actual vs predicted plot saved to {save_path}")

def main():
    """Main training function."""
    try:
        # Load data
        print("Loading data...")
        data = load_and_clean_data()
    except Exception as e:
        print(f"Error loading Excel data: {e}")
        print("Using sample data instead...")
        data = get_sample_data()
    
    # Prepare data
    X, y, feature_columns, target_columns = prepare_model_data(data)
    
    # Train model
    model = WastewaterQualityModel('random_forest')
    results = model.train(X, y, feature_columns, target_columns)
    
    # Save model
    model.save_model('wastewater_model.pkl')
    
    # Create visualizations
    plot_feature_importance(model.feature_importance)
    plot_actual_vs_predicted(results['y_test_orig'], results['y_pred_test_orig'], target_columns)
    
    print("\nTraining completed successfully!")
    
    return model, results

if __name__ == "__main__":
    model, results = main()
