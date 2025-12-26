"""
Complete End-to-End Example
Demonstrates the entire workflow from data collection to prediction
"""

import os
import logging
from collect_polymarket_data import PolymarketDataCollector
from feature_engineering import PolymarketFeatureEngineering
from train_model import PolymarketPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run complete pipeline"""
    
    logger.info("="*80)
    logger.info("POLYMARKET PREDICTION PROJECT - COMPLETE PIPELINE")
    logger.info("="*80)
    
    # Create data directory
    os.makedirs("polymarket_data", exist_ok=True)
    
    # ========================================
    # STEP 1: DATA COLLECTION
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: COLLECTING DATA")
    logger.info("="*80)
    
    collector = PolymarketDataCollector()
    
    # Collect data for 10 markets (adjust as needed)
    datasets = collector.collect_multiple_markets(
        num_markets=10, 
        resample_freq='1H'
    )
    
    if not datasets:
        logger.error("Failed to collect data. Please check API connectivity.")
        return
    
    # Save collected data
    collector.save_datasets(datasets)
    
    logger.info(f"✓ Collected data for {len(datasets)} markets")
    
    # ========================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: ENGINEERING FEATURES")
    logger.info("="*80)
    
    engineer = PolymarketFeatureEngineering()
    
    # Prepare training data with 6-hour prediction horizon
    training_df = engineer.prepare_training_data(
        datasets, 
        prediction_horizon=6
    )
    
    if training_df.empty:
        logger.error("Failed to create training data.")
        return
    
    # Save processed data
    engineer.save_processed_data(training_df)
    
    logger.info(f"✓ Created {len(training_df)} training samples")
    logger.info(f"✓ Engineered {len(engineer.get_feature_columns(training_df))} features")
    
    # ========================================
    # STEP 3: MODEL TRAINING
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: TRAINING MODEL")
    logger.info("="*80)
    
    predictor = PolymarketPredictor()
    
    # Prepare features and target
    X, y = predictor.prepare_features_and_target(training_df)
    
    # Split data
    data_splits = predictor.split_data(X, y, test_size=0.2, val_size=0.1)
    
    # Scale features
    data_splits = predictor.scale_features(data_splits)
    
    # Train and compare models
    results = predictor.compare_models(data_splits)
    
    # Select best model
    best_model_name, best_model = predictor.select_best_model(results)
    
    # Evaluate on test set
    test_metrics = predictor.evaluate_model(
        best_model,
        data_splits['X_test_scaled'],
        data_splits['y_test'],
        "Test"
    )
    
    # Save model
    predictor.model = best_model
    predictor.save_model(best_model, model_name="best_model")
    
    logger.info(f"✓ Best model: {best_model_name}")
    logger.info(f"✓ Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"✓ Test F1 score: {test_metrics['f1']:.4f}")
    
    # ========================================
    # STEP 4: VISUALIZATIONS
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    # Feature importance (if Random Forest)
    if best_model_name == 'Random Forest':
        predictor.plot_feature_importance()
        logger.info("✓ Feature importance plot saved")
    
    # Confusion matrix
    predictor.plot_confusion_matrix(
        best_model,
        data_splits['X_test_scaled'],
        data_splits['y_test']
    )
    logger.info("✓ Confusion matrix saved")
    
    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*80)
    
    logger.info("\nGenerated Files:")
    logger.info("📁 polymarket_data/")
    logger.info("  ├── polymarket_training_data.json          (Raw data)")
    logger.info("  ├── polymarket_training_data_summary.csv   (Summary)")
    logger.info("  ├── processed_training_data.csv            (ML features)")
    logger.info("  ├── best_model.pkl                         (Trained model)")
    logger.info("  ├── scaler.pkl                             (Feature scaler)")
    logger.info("  ├── feature_names.json                     (Feature list)")
    logger.info("  ├── feature_importance.png                 (Visualization)")
    logger.info("  └── confusion_matrix.png                   (Visualization)")
    
    logger.info("\nNext Steps:")
    logger.info("1. Run 'python predict_markets.py' to make live predictions")
    logger.info("2. Check visualizations in polymarket_data/ folder")
    logger.info("3. Review model performance metrics above")
    logger.info("4. Collect more data and retrain for better accuracy")
    
    logger.info("\n" + "="*80)
    logger.info("Model Performance Summary:")
    logger.info("="*80)
    logger.info(f"Model Type:      {best_model_name}")
    logger.info(f"Accuracy:        {test_metrics['accuracy']:.2%}")
    logger.info(f"Precision:       {test_metrics['precision']:.2%}")
    logger.info(f"Recall:          {test_metrics['recall']:.2%}")
    logger.info(f"F1 Score:        {test_metrics['f1']:.4f}")
    logger.info(f"ROC AUC:         {test_metrics['roc_auc']:.4f}")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"\n\nPipeline failed with error: {e}")
        raise
