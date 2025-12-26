# Polymarket Price Prediction - Project Structure

## 📁 Project Files

```
polymarket-prediction/
│
├── collect_polymarket_data.py       # Data collection from Polymarket API
├── feature_engineering.py           # Feature engineering and preprocessing
├── train_model.py                   # Model training and evaluation
├── predict_markets.py               # Live market predictions
├── run_complete_pipeline.py         # End-to-end automated pipeline
├── polymarket_analysis.ipynb        # Interactive Jupyter notebook
│
├── requirements.txt                 # Python dependencies
├── README.md                        # Complete documentation
│
└── polymarket_data/                 # Generated data directory
    ├── polymarket_training_data.json
    ├── polymarket_training_data_summary.csv
    ├── processed_training_data.csv
    ├── processed_training_data_features.txt
    ├── best_model.pkl
    ├── scaler.pkl
    ├── feature_names.json
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── predictions.json
    └── predictions.csv
```

## 🔧 Script Descriptions

### 1. `collect_polymarket_data.py`
**Purpose**: Fetch historical market data from Polymarket

**Key Features**:
- Connects to Polymarket CLOB API
- Fetches active markets and their trade history
- Collects orderbook snapshots
- Resamples data to regular time intervals
- Saves raw data and summary statistics

**Usage**:
```bash
python collect_polymarket_data.py
```

**Customization**:
```python
datasets = collector.collect_multiple_markets(
    num_markets=20,      # Number of markets to collect
    resample_freq='1H'   # Time interval ('1H', '4H', '1D')
)
```

---

### 2. `feature_engineering.py`
**Purpose**: Transform raw data into ML-ready features

**Key Features**:
- Technical indicators (SMA, EMA, RSI, Bollinger Bands)
- Time-based features (hour, day of week, cyclical encoding)
- Volume metrics and momentum indicators
- Orderbook features (spread, bid/ask volume)
- Target variable creation (price direction)

**Usage**:
```bash
python feature_engineering.py
```

**Customization**:
```python
training_df = engineer.prepare_training_data(
    datasets,
    prediction_horizon=6  # Hours to predict ahead
)
```

---

### 3. `train_model.py`
**Purpose**: Train and evaluate ML models

**Key Features**:
- Multiple model types (Random Forest, Gradient Boosting, Logistic Regression)
- Automated model comparison
- Cross-validation
- Feature importance analysis
- Performance visualization

**Usage**:
```bash
python train_model.py
```

**Models Trained**:
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential boosting algorithm
- **Logistic Regression**: Linear baseline model

**Outputs**:
- Trained model file (`.pkl`)
- Feature scaler
- Performance metrics
- Visualizations

---

### 4. `predict_markets.py`
**Purpose**: Make predictions on live markets

**Key Features**:
- Loads trained model
- Fetches live market data
- Calculates features in real-time
- Generates predictions with confidence scores
- Saves predictions to JSON/CSV

**Usage**:
```bash
python predict_markets.py
```

**Prediction Output**:
```json
{
  "question": "Will X happen?",
  "prediction": "Price will increase",
  "probability": 0.75,
  "confidence": "High",
  "current_price": 0.65
}
```

---

### 5. `run_complete_pipeline.py`
**Purpose**: Execute entire workflow automatically

**Key Features**:
- Runs all steps in sequence
- Error handling and logging
- Progress reporting
- Summary statistics

**Usage**:
```bash
python run_complete_pipeline.py
```

**Pipeline Steps**:
1. Data collection
2. Feature engineering
3. Model training
4. Model evaluation
5. Visualization generation

---

### 6. `polymarket_analysis.ipynb`
**Purpose**: Interactive exploration and analysis

**Key Features**:
- Step-by-step walkthrough
- Data visualization
- Model experimentation
- Live predictions
- Educational content

**Usage**:
```bash
jupyter notebook polymarket_analysis.ipynb
```

---

## 🎯 Workflow

### Quick Start (Automated)
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_complete_pipeline.py

# Make predictions
python predict_markets.py
```

### Manual Step-by-Step
```bash
# Step 1: Collect data
python collect_polymarket_data.py

# Step 2: Engineer features
python feature_engineering.py

# Step 3: Train model
python train_model.py

# Step 4: Make predictions
python predict_markets.py
```

### Interactive (Notebook)
```bash
jupyter notebook polymarket_analysis.ipynb
```

---

## 📊 Data Flow

```
Polymarket API
      ↓
[collect_polymarket_data.py]
      ↓
Raw Trade Data (JSON)
      ↓
[feature_engineering.py]
      ↓
Processed Features (CSV)
      ↓
[train_model.py]
      ↓
Trained Model (.pkl)
      ↓
[predict_markets.py]
      ↓
Predictions (JSON/CSV)
```

---

## 🔑 Key Classes

### PolymarketDataCollector
- `get_active_markets()`: Fetch available markets
- `get_market_trades()`: Get trade history
- `calculate_market_features()`: Compute statistics
- `resample_to_timeseries()`: Create time series

### PolymarketFeatureEngineering
- `create_technical_indicators()`: Add TA features
- `create_time_features()`: Add temporal features
- `create_target_variable()`: Generate prediction target
- `prepare_training_data()`: Full dataset preparation

### PolymarketPredictor
- `train_random_forest()`: Train RF model
- `train_gradient_boosting()`: Train GB model
- `compare_models()`: Compare all models
- `evaluate_model()`: Calculate metrics
- `save_model()`: Persist trained model

### PolymarketLivePredictor
- `get_live_market()`: Fetch current market
- `get_recent_trades()`: Get latest trades
- `predict_market()`: Make single prediction
- `predict_multiple_markets()`: Batch predictions

---

## 🎨 Visualizations Generated

1. **feature_importance.png**
   - Bar chart of top features
   - Shows which variables drive predictions

2. **confusion_matrix.png**
   - Model prediction accuracy
   - True vs predicted labels

3. **Price History** (in notebook)
   - Time series plots
   - Market price movements

4. **Model Comparison** (in notebook)
   - Performance metrics comparison
   - Accuracy, F1, ROC AUC

---

## 🔄 Typical Development Cycle

1. **Initial Setup**
   ```bash
   python run_complete_pipeline.py
   ```

2. **Evaluate Results**
   - Check `polymarket_data/` outputs
   - Review model metrics
   - Analyze feature importance

3. **Iterate**
   - Adjust parameters in scripts
   - Add new features
   - Try different models
   - Re-run pipeline

4. **Deploy**
   - Run `predict_markets.py` regularly
   - Monitor prediction accuracy
   - Update model with new data

---

## 📈 Performance Targets

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Accuracy | >55% | >60% | >65% |
| Precision | >60% | >70% | >80% |
| Recall | >55% | >65% | >75% |
| F1 Score | >0.60 | >0.65 | >0.70 |
| ROC AUC | >0.65 | >0.70 | >0.75 |

---

## 🐛 Common Issues

**Problem**: No data collected
- **Solution**: Check API connectivity, try fewer markets

**Problem**: Low model accuracy
- **Solution**: Collect more data, adjust prediction horizon, add features

**Problem**: API rate limiting
- **Solution**: Add delays in collection script (`time.sleep()`)

**Problem**: Import errors
- **Solution**: Install dependencies: `pip install -r requirements.txt`

---

## 🚀 Next Steps

After running the basic pipeline:

1. **Improve Data Collection**
   - Collect more markets over time
   - Focus on high-volume markets
   - Add historical depth

2. **Enhance Features**
   - Add sentiment analysis
   - Include external data (news, social media)
   - Market-specific indicators

3. **Model Optimization**
   - Hyperparameter tuning (GridSearchCV)
   - Ensemble methods
   - Deep learning models (LSTM, Transformer)

4. **Deployment**
   - Automate daily predictions
   - Create web dashboard
   - Set up alerts for high-confidence predictions

---

## 📚 References

- **Polymarket Docs**: https://docs.polymarket.com
- **Technical Analysis**: Standard TA indicators
- **scikit-learn**: https://scikit-learn.org/
- **pandas**: https://pandas.pydata.org/

---

**Happy Predicting! 🎯📊**
