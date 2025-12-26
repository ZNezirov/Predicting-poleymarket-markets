"""
Comprehensive Model Evaluation and Testing
Detailed analysis of model performance and predictions
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and testing"""
    
    def __init__(self, model_path="polymarket_data/best_model.pkl",
                 scaler_path="polymarket_data/scaler.pkl",
                 data_path="polymarket_data/processed_training_data.csv"):
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.data = pd.read_csv(data_path)
        
        logger.info("✓ Model, scaler, and data loaded")
    
    def prepare_test_data(self, test_size=0.2):
        """Prepare test set from data"""
        
        exclude_cols = ['target_binary', 'target_price_change', 'timestamp', 
                       'condition_id', 'open', 'high', 'low', 'close']
        
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        X = self.data[feature_cols].copy()
        y = self.data['target_binary'].copy()
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            
            # Additional metrics
            'true_positives': int(((y_pred == 1) & (y_true == 1)).sum()),
            'true_negatives': int(((y_pred == 0) & (y_true == 0)).sum()),
            'false_positives': int(((y_pred == 1) & (y_true == 0)).sum()),
            'false_negatives': int(((y_pred == 0) & (y_true == 1)).sum()),
        }
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Evaluate model on test set"""
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_all_metrics(y_test, y_pred, y_pred_proba)
        
        return y_pred, y_pred_proba, metrics
    
    def print_metrics_report(self, metrics, dataset_name="Test"):
        """Print formatted metrics report"""
        
        print("\n" + "="*70)
        print(f"{dataset_name} Set Evaluation Results")
        print("="*70)
        
        print(f"\n Overall Performance:")
        print(f"  Accuracy:     {metrics['accuracy']:.2%}")
        print(f"  Precision:    {metrics['precision']:.2%}")
        print(f"  Recall:       {metrics['recall']:.2%}")
        print(f"  F1 Score:     {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:      {metrics['roc_auc']:.4f}")
        print(f"  Specificity:  {metrics['specificity']:.2%}")
        
        print(f"\n Confusion Matrix Breakdown:")
        print(f"  True Positives:  {metrics['true_positives']:4d}  (Correctly predicted UP)")
        print(f"  True Negatives:  {metrics['true_negatives']:4d}  (Correctly predicted DOWN)")
        print(f"  False Positives: {metrics['false_positives']:4d}  (Predicted UP, was DOWN)")
        print(f"  False Negatives: {metrics['false_negatives']:4d}  (Predicted DOWN, was UP)")
        
        total = sum([metrics['true_positives'], metrics['true_negatives'], 
                     metrics['false_positives'], metrics['false_negatives']])
        
        print(f"\n Interpretation:")
        if metrics['accuracy'] > 0.65:
            print(f"   EXCELLENT: Model accuracy is {metrics['accuracy']:.1%}")
        elif metrics['accuracy'] > 0.60:
            print(f"  ✓ GOOD: Model accuracy is {metrics['accuracy']:.1%}")
        elif metrics['accuracy'] > 0.55:
            print(f"  → MODERATE: Model accuracy is {metrics['accuracy']:.1%}")
        else:
            print(f"   LOW: Model accuracy is {metrics['accuracy']:.1%}")
        
        if metrics['precision'] > 0.70:
            print(f"   When predicting UP, correct {metrics['precision']:.1%} of the time")
        else:
            print(f"  → When predicting UP, correct {metrics['precision']:.1%} of the time")
        
        if metrics['recall'] > 0.65:
            print(f"   Catches {metrics['recall']:.1%} of actual UP movements")
        else:
            print(f"  Catches {metrics['recall']:.1%} of actual UP movements")
        
        print("="*70)
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path="polymarket_data/confusion_matrix_detailed.png"):
        """Plot detailed confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Predicted DOWN', 'Predicted UP'],
                   yticklabels=['Actual DOWN', 'Actual UP'])
        
        plt.title('Confusion Matrix - Model Predictions', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy in title
        accuracy = accuracy_score(y_true, y_pred)
        plt.suptitle(f'Overall Accuracy: {accuracy:.2%}', y=0.98, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path="polymarket_data/roc_curve.png"):
        """Plot ROC curve"""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Model Performance', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ ROC curve saved to {save_path}")
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, 
                                   save_path="polymarket_data/precision_recall_curve.png"):
        """Plot Precision-Recall curve"""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_f1_idx = np.argmax(f1_scores)
        plt.scatter(recall[best_f1_idx], precision[best_f1_idx], 
                   color='red', s=100, zorder=5,
                   label=f'Best F1 = {f1_scores[best_f1_idx]:.3f}')
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Precision-Recall curve saved to {save_path}")
        plt.close()
    
    def plot_prediction_distribution(self, y_true, y_pred_proba, 
                                    save_path="polymarket_data/prediction_distribution.png"):
        """Plot distribution of prediction probabilities"""
        
        plt.figure(figsize=(12, 6))
        
        proba_class_0 = y_pred_proba[y_true == 0]
        proba_class_1 = y_pred_proba[y_true == 1]
        
        plt.hist(proba_class_0, bins=50, alpha=0.6, label='Actual DOWN', color='red')
        plt.hist(proba_class_1, bins=50, alpha=0.6, label='Actual UP', color='green')
        
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, 
                   label='Decision Threshold (0.5)')
        
        plt.xlabel('Predicted Probability of UP', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Probabilities', fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Prediction distribution saved to {save_path}")
        plt.close()
    
    def analyze_by_confidence(self, y_true, y_pred, y_pred_proba):
        """Analyze performance by prediction confidence"""
        
        print("\n" + "="*70)
        print("Performance by Confidence Level")
        print("="*70)
        
        confidence_levels = {
            'Very High (>80%)': (y_pred_proba > 0.8) | (y_pred_proba < 0.2),
            'High (70-80%)': ((y_pred_proba >= 0.7) & (y_pred_proba <= 0.8)) | 
                            ((y_pred_proba >= 0.2) & (y_pred_proba <= 0.3)),
            'Medium (60-70%)': ((y_pred_proba >= 0.6) & (y_pred_proba < 0.7)) | 
                              ((y_pred_proba > 0.3) & (y_pred_proba <= 0.4)),
            'Low (50-60%)': (y_pred_proba >= 0.5) & (y_pred_proba < 0.6) | 
                           (y_pred_proba > 0.4) & (y_pred_proba <= 0.5)
        }
        
        for level, mask in confidence_levels.items():
            if mask.sum() > 0:
                acc = accuracy_score(y_true[mask], y_pred[mask])
                count = mask.sum()
                print(f"\n{level}:")
                print(f"  Predictions: {count}")
                print(f"  Accuracy:    {acc:.2%}")
    
    def test_on_recent_data(self, X_test, y_test, y_pred, y_pred_proba, n_recent=50):
        """Test on most recent data points"""
        
        print("\n" + "="*70)
        print(f"Performance on Most Recent {n_recent} Predictions")
        print("="*70)
        
        # Take last n samples
        y_true_recent = y_test.iloc[-n_recent:]
        y_pred_recent = y_pred[-n_recent:]
        y_proba_recent = y_pred_proba[-n_recent:]
        
        metrics = self.calculate_all_metrics(y_true_recent, y_pred_recent, y_proba_recent)
        
        print(f"\nAccuracy: {metrics['accuracy']:.2%}")
        print(f"Correct predictions: {metrics['true_positives'] + metrics['true_negatives']}/{n_recent}")
    
    def save_predictions_analysis(self, X_test, y_test, y_pred, y_pred_proba,
                                 filepath="polymarket_data/predictions_analysis.csv"):
        """Save detailed predictions for analysis"""
        
        results_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
            'probability': y_pred_proba,
            'correct': (y_test.values == y_pred).astype(int),
            'confidence': np.abs(y_pred_proba - 0.5) * 2  # 0 to 1 scale
        })
        
        results_df.to_csv(filepath, index=False)
        logger.info(f"✓ Predictions analysis saved to {filepath}")
        
        return results_df


def main():
    """Run comprehensive model evaluation"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    try:
        evaluator = ModelEvaluator()
        
        X_train, X_test, y_train, y_test, feature_cols = evaluator.prepare_test_data()
        
        y_pred, y_pred_proba, metrics = evaluator.evaluate_on_test_set(X_test, y_test)
        
        evaluator.print_metrics_report(metrics, "Test")
        
        evaluator.analyze_by_confidence(y_test, y_pred, y_pred_proba)
        
        evaluator.test_on_recent_data(X_test, y_test, y_pred, y_pred_proba, n_recent=50)
        
        # Create all visualizations
        print("\n" + "="*70)
        print("Generating Visualizations...")
        print("="*70)
        
        evaluator.plot_confusion_matrix(y_test, y_pred)
        evaluator.plot_roc_curve(y_test, y_pred_proba)
        evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
        evaluator.plot_prediction_distribution(y_test, y_pred_proba)
        
        
        results_df = evaluator.save_predictions_analysis(X_test, y_test, y_pred, y_pred_proba)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print("\nGenerated Files:")
        print("   polymarket_data/confusion_matrix_detailed.png")
        print("   polymarket_data/roc_curve.png")
        print("   polymarket_data/precision_recall_curve.png")
        print("   polymarket_data/prediction_distribution.png")
        print("   polymarket_data/predictions_analysis.csv")
        
        print("\n✅ Model evaluation complete!")
        
        #-- model summary
        print("\n" + "="*70)
        print("QUICK SUMMARY")
        print("="*70)
        print(f"✓ Accuracy:  {metrics['accuracy']:.1%}")
        print(f"✓ Precision: {metrics['precision']:.1%}")
        print(f"✓ Recall:    {metrics['recall']:.1%}")
        print(f"✓ F1 Score:  {metrics['f1_score']:.3f}")
        print(f"✓ ROC AUC:   {metrics['roc_auc']:.3f}")
        print("="*70)
        
    except FileNotFoundError as e:
        print("\n ERROR: Could not find required files.")
        print("Make sure you've run these scripts first:")
        print("  1. python collect_data_simple.py")
        print("  2. python feature_engineering.py")
        print("  3. python train_model.py")
        print(f"\nMissing file: {e}")
    
    except Exception as e:
        print(f"\n ERROR: {e}")
        raise


if __name__ == "__main__":
    main()