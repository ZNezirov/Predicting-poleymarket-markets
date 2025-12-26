# Polymarket Price Prediction Project

A complete machine learning pipeline for predicting Polymarket outcome prices using historical trade data and technical indicators.

## 🎯 Project Overview

This project collects historical data from Polymarket's API, engineers features from trade data, trains ML models, and makes predictions on live markets.

### Features
- **Data Collection**: Automated fetching of market data, trades, and orderbooks
- **Feature Engineering**: Technical indicators, time features, volume metrics
- **Model Training**: Multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
- **Live Predictions**: Real-time predictions on active markets
- **Model Evaluation**: Comprehensive metrics and visualizations

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- Python 3.8+
- requests
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 🚀 Quick Start

### Step 1: Collect Training Data

```bash
python collect_polymarket_data.py
```

This script will:
- Fetch active markets from Polymarket
- Collect historical trades for each market
- Calculate market features and statistics
- Save data to `polymarket_data/polymarket_training_data.json`

**Parameters you can adjust:**
- `num_markets`: Number of markets to collect (default: 20)
- `resample_freq`: Time interval for resampling ('1H', '4H', '1D')

### Step 2: Engineer Features

```bash
python feature_engineering.py
```

This script will:
- Load collected market data
- Create technical indicators (SMA, EMA, RSI, Bollinger Bands)
- Add time-based features
- Calculate orderbook features
- Create target variable (price movement prediction)
- Save processed data to `polymarket_data/processed_training_data.csv`

**Key features created:**
- **Technical**: Moving averages, momentum, volatility, RSI
- **Time-based**: Hour, day of week, cyclical encoding
- **Volume**: Volume changes, volume moving averages
- **Market**: Total trades, liquidity, spread

### Step 3: Train Model

```bash
python train_model.py
```

This script will:
- Load processed training data
- Split into train/validation/test sets (70/10/20)
- Train multiple models and compare performance
- Select best model based on validation F1 score
- Evaluate on test set
- Save model, scaler, and feature names

**Models trained:**
- Random Forest (200 trees)
- Gradient Boosting (200 estimators)
- Logistic Regression

**Outputs:**
- `best_model.pkl`: Trained model
- `scaler.pkl`: Feature scaler
- `feature_names.json`: List of feature names
- `feature_importance.png`: Feature importance plot
- `confusion_matrix.png`: Confusion matrix visualization

### Step 4: Make Predictions

```bash
python predict_markets.py
```

This script will:
- Load trained model
- Fetch current active markets
- Collect recent trade data
- Calculate features in real-time
- Make predictions
- Save predictions to `polymarket_data/predictions.json`

## 📊 Output Files

```
polymarket_data/
├── polymarket_training_data.json      # Raw collected data
├── polymarket_training_data_summary.csv  # Summary statistics
├── processed_training_data.csv        # ML-ready features
├── processed_training_data_features.txt  # Feature names
├── best_model.pkl                     # Trained model
├── scaler.pkl                         # Feature scaler
├── feature_names.json                 # Model features
├── feature_importance.png             # Feature importance plot
├── confusion_matrix.png               # Model performance
├── predictions.json                   # Live predictions
└── predictions.csv                    # Predictions summary
```

## 🎓 Understanding the Prediction

The model predicts whether the market price will **increase** over the next 6 hours (configurable).

**Prediction Output:**
```json
{
  "condition_id": "0x123...",
  "question": "Will X happen by Y date?",
  "prediction": 1,  // 1 = increase, 0 = decrease/flat
  "probability": 0.75,  // 75% confidence
  "confidence": "High",  // High/Medium/Low
  "current_price": 0.65,
  "prediction_label": "Price will increase"
}
```

**Confidence Levels:**
- **High**: Probability > 0.7 or < 0.3
- **Medium**: Probability between 0.6-0.7 or 0.3-0.4
- **Low**: Probability near 0.5 (uncertain)

## ⚙️ Configuration

### Customize Data Collection

Edit `collect_polymarket_data.py`:

```python
# Collect more/fewer markets
datasets = collector.collect_multiple_markets(
    num_markets=50,  # Increase for more data
    resample_freq='4H'  # '1H', '4H', '1D'
)
```

### Customize Prediction Horizon

Edit `feature_engineering.py`:

```python
# Predict further into the future
training_df = engineer.prepare_training_data(
    datasets, 
    prediction_horizon=12  # Hours (default: 6)
)
```

### Customize Model Parameters

Edit `train_model.py`:

```python
# Random Forest parameters
model = RandomForestClassifier(
    n_estimators=300,  # More trees
    max_depth=20,      # Deeper trees
    ...
)
```

## 📈 Model Performance Tips

### Improving Accuracy

1. **Collect More Data**
   - Increase `num_markets` in collection script
   - Run collection multiple times over days/weeks
   - Focus on markets with high volume

2. **Feature Engineering**
   - Add more technical indicators
   - Include sentiment analysis (if API available)
   - Add market-specific features (category, end date)

3. **Model Tuning**
   - Use GridSearchCV for hyperparameter optimization
   - Try ensemble methods (voting classifier)
   - Experiment with different prediction horizons

4. **Data Quality**
   - Filter markets with sufficient trade volume
   - Remove outliers and anomalies
   - Balance dataset if class imbalance exists

## 🔍 Interpreting Results

### Feature Importance

After training, check `feature_importance.png` to see which features are most predictive:

- **High importance**: Price momentum, moving averages, volatility
- **Medium importance**: Volume indicators, time features
- **Low importance**: Less relevant market metadata

### Model Metrics

- **Accuracy**: Overall correctness (aim for >60%)
- **Precision**: When predicting increase, how often correct (aim for >70%)
- **Recall**: Of actual increases, how many caught (aim for >60%)
- **F1 Score**: Balance of precision and recall (aim for >0.65)
- **ROC AUC**: Model's ability to distinguish classes (aim for >0.70)

## 🛠️ Troubleshooting

### Issue: No markets found
**Solution**: Check Polymarket API status. Try different API endpoints.

### Issue: No trade data available
**Solution**: Markets might be too new. Try older, more established markets.

### Issue: Low model accuracy
**Solution**: 
- Collect more data
- Increase prediction horizon
- Add more features
- Try different models

### Issue: API rate limiting
**Solution**: Add delays between requests in collection script:
```python
time.sleep(2)  # Increase delay
```

## 📚 API Documentation

- **CLOB API**: https://docs.polymarket.com (if available)
- **Gamma API**: Used for market metadata

## 🤝 Usage Examples

### Predict Specific Market

```python
from predict_markets import PolymarketLivePredictor

predictor = PolymarketLivePredictor()
result = predictor.predict_market("0x1234...")  # Your condition_id
print(result)
```

### Batch Predictions

```python
condition_ids = ["0x1234...", "0x5678...", "0xabcd..."]
results = predictor.predict_multiple_markets(condition_ids)
```

### Custom Feature Calculation

```python
from feature_engineering import PolymarketFeatureEngineering

engineer = PolymarketFeatureEngineering()
features = engineer.create_technical_indicators(your_df)
```

## 📝 Notes

- **Market Selection**: Focus on liquid markets with consistent trading
- **Time Zones**: All timestamps are in UTC
- **Data Freshness**: Collect new data regularly for best results
- **Backtesting**: The model is trained on historical data; performance may vary on live markets
- **Risk Warning**: This is for educational purposes. Always do your own research before trading.

## 🔮 Future Enhancements

- [ ] Add sentiment analysis from social media
- [ ] Implement real-time streaming predictions
- [ ] Add more advanced models (LSTM, Transformer)
- [ ] Create web dashboard for visualizations
- [ ] Add backtesting framework
- [ ] Implement automated trading signals
- [ ] Add market correlation analysis

## 📄 License

This project is for educational purposes. Use at your own risk.

## 🙋 Support

If you encounter issues:
1. Check API connectivity
2. Verify data collection completed successfully
3. Ensure all dependencies are installed
4. Check log files for error messages

---

**Happy Predicting! 🚀📊**
