"""
Machine Learning Model for Polymarket Price Prediction
Trains and evaluates models on prepared market data
"""

import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketPredictor:
    """Train and evaluate ML models for Polymarket price prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
    def load_data(self, filepath: str = "polymarket_data/processed_training_data.csv") -> pd.DataFrame:
        """Load processed training data"""
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} samples from {filepath}")
        return df
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable"""
        
        # Exclude non-feature columns
        exclude_cols = ['target_binary', 'target_price_change', 'timestamp', 
                       'condition_id', 'open', 'high', 'low', 'close']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['target_binary'].copy()
        
        # Handle any remaining NaN or inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        self.feature_names = feature_cols
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, 
                   val_size: float = 0.1) -> Dict:
        """Split data into train/val/test sets"""
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate validation set from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def scale_features(self, data_splits: Dict) -> Dict:
        """Scale features using StandardScaler"""
        
        # Fit scaler on training data only
        self.scaler.fit(data_splits['X_train'])
        
        # Transform all splits
        data_splits['X_train_scaled'] = pd.DataFrame(
            self.scaler.transform(data_splits['X_train']),
            columns=data_splits['X_train'].columns,
            index=data_splits['X_train'].index
        )
        
        data_splits['X_val_scaled'] = pd.DataFrame(
            self.scaler.transform(data_splits['X_val']),
            columns=data_splits['X_val'].columns,
            index=data_splits['X_val'].index
        )
        
        data_splits['X_test_scaled'] = pd.DataFrame(
            self.scaler.transform(data_splits['X_test']),
            columns=data_splits['X_test'].columns,
            index=data_splits['X_test'].index
        )
        
        logger.info("Features scaled using StandardScaler")
        
        return data_splits
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model
    
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        """Train Gradient Boosting model"""
        logger.info("Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")
        
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> Dict:
        """Evaluate model performance"""
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        logger.info(f"\n{dataset_name} Set Performance:")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def compare_models(self, data_splits: Dict) -> Dict:
        """Train and compare multiple models"""
        
        X_train = data_splits['X_train_scaled']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val_scaled']
        y_val = data_splits['y_val']
        
        models = {
            'Random Forest': self.train_random_forest(X_train, y_train),
            'Gradient Boosting': self.train_gradient_boosting(X_train, y_train),
            'Logistic Regression': self.train_logistic_regression(X_train, y_train)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating {name}")
            logger.info(f"{'='*50}")
            
            train_metrics = self.evaluate_model(model, X_train, y_train, "Train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "Validation")
            
            results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
        
        return results
    
    def select_best_model(self, results: Dict) -> Tuple[str, object]:
        """Select best model based on validation F1 score"""
        
        best_model_name = None
        best_f1 = 0
        
        for name, result in results.items():
            val_f1 = result['val_metrics']['f1']
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_name = name
        
        logger.info(f"\nBest model: {best_model_name} (Validation F1: {best_f1:.4f})")
        
        return best_model_name, results[best_model_name]['model']
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = "polymarket_data/feature_importance.png"):
        """Plot top feature importances"""
        if self.feature_importance is None:
            logger.warning("No feature importance available")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, model, X: pd.DataFrame, y: pd.Series, 
                             save_path: str = "polymarket_data/confusion_matrix.png"):
        """Plot confusion matrix"""
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def save_model(self, model, model_name: str = "best_model", 
                   save_dir: str = "polymarket_data"):
        """Save trained model and scaler"""
        
        model_path = f"{save_dir}/{model_name}.pkl"
        scaler_path = f"{save_dir}/scaler.pkl"
        features_path = f"{save_dir}/feature_names.json"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Feature names saved to {features_path}")
    
    def load_model(self, model_path: str = "polymarket_data/best_model.pkl",
                   scaler_path: str = "polymarket_data/scaler.pkl",
                   features_path: str = "polymarket_data/feature_names.json"):
        """Load trained model and scaler"""
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        
        logger.info("Model loaded successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        X_scaled = self.scaler.transform(X[self.feature_names])
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities


def main():
    """Main training pipeline"""
    
    predictor = PolymarketPredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Prepare features and target
    X, y = predictor.prepare_features_and_target(df)
    
    # Split data
    data_splits = predictor.split_data(X, y, test_size=0.2, val_size=0.1)
    
    # Scale features
    data_splits = predictor.scale_features(data_splits)
    
    # Train and compare models
    logger.info("\n" + "="*70)
    logger.info("Training and comparing models...")
    logger.info("="*70)
    
    results = predictor.compare_models(data_splits)
    
    # Select best model
    best_model_name, best_model = predictor.select_best_model(results)
    
    # Final evaluation on test set
    logger.info("\n" + "="*70)
    logger.info("Final evaluation on test set")
    logger.info("="*70)
    
    test_metrics = predictor.evaluate_model(
        best_model,
        data_splits['X_test_scaled'],
        data_splits['y_test'],
        "Test"
    )
    
    # Plot feature importance (if Random Forest was best)
    if best_model_name == 'Random Forest':
        predictor.plot_feature_importance()
    
    # Plot confusion matrix
    predictor.plot_confusion_matrix(
        best_model,
        data_splits['X_test_scaled'],
        data_splits['y_test']
    )
    
    # Save the best model
    predictor.model = best_model
    predictor.save_model(best_model, model_name="best_model")
    
    logger.info("\n" + "="*70)
    logger.info("Training complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
