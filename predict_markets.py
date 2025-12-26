"""
prediction live markets on Polymarket using a trained model
"""

import requests
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketLivePredictor:
    """Make predictions on live Polymarket markets"""
    
    def __init__(self, model_path: str = "polymarket_data/best_model.pkl",
                 scaler_path: str = "polymarket_data/scaler.pkl",
                 features_path: str = "polymarket_data/feature_names.json"):
        
        self.clob_base = "https://clob.polymarket.com"
        
        # Load trained model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        
        logger.info("Model loaded successfully")
    
    def get_live_market(self, condition_id: str) -> Dict:
        """Fetch live market data"""
        url = f"{self.clob_base}/markets"
        
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            markets = resp.json()
            
            #find specific market
            for market in markets:
                if market.get('condition_id') == condition_id:
                    return market
            
            logger.error(f"Market {condition_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching market: {e}")
            return None
    
    def get_recent_trades(self, condition_id: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch recent trades for a market"""
        url = f"{self.clob_base}/trades"
        params = {
            "condition_id": condition_id,
            "limit": limit
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            trades = resp.json()
            
            df = pd.DataFrame(trades)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp')
                
                if 'price' in df.columns:
                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                if 'size' in df.columns:
                    df['size'] = pd.to_numeric(df['size'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return pd.DataFrame()
    
    def calculate_features_from_trades(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate features from recent trades"""
        
        if trades_df.empty or 'price' not in trades_df:
            return {}
        
        features = {}
        
        # b asic statistics
        features['avg_price'] = trades_df['price'].mean()
        features['price_std'] = trades_df['price'].std()
        features['price_min'] = trades_df['price'].min()
        features['price_max'] = trades_df['price'].max()
        features['price_range'] = features['price_max'] - features['price_min']
        
        #volume of the trades
        if 'size' in trades_df:
            features['total_volume'] = trades_df['size'].sum()
            features['avg_volume'] = trades_df['size'].mean()
        
        # Recent price action
        recent_trades = trades_df.tail(100)
        if len(recent_trades) > 1:
            features['recent_avg_price'] = recent_trades['price'].mean()
            features['recent_price_std'] = recent_trades['price'].std()
        
        #price momentum
        if len(trades_df) > 1:
            features['momentum_1h'] = (trades_df['price'].iloc[-1] - trades_df['price'].iloc[0]) / trades_df['price'].iloc[0]
        
        # moving averages
        if len(trades_df) > 10:
            features['sma_10'] = trades_df['price'].tail(10).mean()
        if len(trades_df) > 50:
            features['sma_50'] = trades_df['price'].tail(50).mean()
        
        # Volatility
        if len(trades_df) > 20:
            features['volatility_24h'] = trades_df['price'].tail(20).std()
        
        return features
    
    def prepare_features_for_prediction(self, market_features: Dict) -> pd.DataFrame:
        """Prepare features in the format expected by the model"""
        
        # Create a DataFrame with all required features
        feature_dict = {name: 0 for name in self.feature_names}
        
        for key, value in market_features.items():
            if key in feature_dict:
                feature_dict[key] = value
        
        df = pd.DataFrame([feature_dict])
        
        return df
    
    def predict_market(self, condition_id: str) -> Dict:
        """Make prediction for a specific market"""
        
        logger.info(f"Analyzing market: {condition_id}")
        
        market = self.get_live_market(condition_id)
        if not market:
            return None
        
        trades_df = self.get_recent_trades(condition_id)
        if trades_df.empty:
            logger.error("No trade data available")
            return None
        
        features = self.calculate_features_from_trades(trades_df)
        
        features['market_num_tokens'] = len(market.get('tokens', []))
        features['market_is_active'] = int(market.get('active', False))
        
        X = self.prepare_features_for_prediction(features)
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]
        
        result = {
            'condition_id': condition_id,
            'question': market.get('question', 'Unknown'),
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': 'High' if abs(probability - 0.5) > 0.2 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low',
            'current_price': trades_df['price'].iloc[-1] if not trades_df.empty else None,
            'prediction_label': 'Price will increase' if prediction == 1 else 'Price will decrease/stay flat',
            'timestamp': datetime.now().isoformat(),
            'features_used': features
        }
        
        return result
    
    def predict_multiple_markets(self, condition_ids: List[str]) -> List[Dict]:
        """Make predictions for multiple markets"""
        
        results = []
        
        for condition_id in condition_ids:
            try:
                result = self.predict_market(condition_id)
                if result:
                    results.append(result)
                    logger.info(f"✓ {result['question'][:60]}")
                    logger.info(f"  Prediction: {result['prediction_label']}")
                    logger.info(f"  Probability: {result['probability']:.2%}")
                    logger.info(f"  Confidence: {result['confidence']}\n")
            except Exception as e:
                logger.error(f"Error predicting {condition_id}: {e}")
        
        return results
    
    def get_top_markets_to_predict(self, limit: int = 10) -> List[str]:
        """Get active markets with good liquidity for prediction"""
        
        url = f"{self.clob_base}/markets"
        
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            markets = resp.json()
            
            active_markets = [m for m in markets if m.get('active', False)]
            
            active_markets.sort(key=lambda x: float(x.get('volume', 0)), reverse=True)
            
            condition_ids = [m['condition_id'] for m in active_markets[:limit]]
            
            logger.info(f"Found {len(condition_ids)} markets for prediction")
            
            return condition_ids
            
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
    
    def save_predictions(self, predictions: List[Dict], 
                        filepath: str = "polymarket_data/predictions.json"):
        """Save predictions to file"""
        
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Predictions saved to {filepath}")
        
        # Also save as CSV for easy viewing
        csv_path = filepath.replace('.json', '.csv')
        df = pd.DataFrame([{
            'condition_id': p['condition_id'],
            'question': p['question'],
            'prediction': p['prediction_label'],
            'probability': p['probability'],
            'confidence': p['confidence'],
            'current_price': p['current_price'],
            'timestamp': p['timestamp']
        } for p in predictions])
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Predictions summary saved to {csv_path}")


def main():
    """Main execution"""
    
    #init predictor
    predictor = PolymarketLivePredictor()
    
    # optie 1: Predict specific market
    # condition_id = "your_condition_id_here"
    # result = predictor.predict_market(condition_id)
    # print(json.dumps(result, indent=2))
    
    #optie 2: Predict top active markets
    logger.info("Finding top active markets...")
    condition_ids = predictor.get_top_markets_to_predict(limit=10)
    
    if condition_ids:
        logger.info(f"\nMaking predictions for {len(condition_ids)} markets...\n")
        predictions = predictor.predict_multiple_markets(condition_ids)
        
        if predictions:
            predictor.save_predictions(predictions)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Prediction Summary")
            logger.info(f"{'='*70}")
            logger.info(f"Total predictions: {len(predictions)}")
            logger.info(f"High confidence: {sum(1 for p in predictions if p['confidence'] == 'High')}")
            logger.info(f"Medium confidence: {sum(1 for p in predictions if p['confidence'] == 'Medium')}")
            logger.info(f"Low confidence: {sum(1 for p in predictions if p['confidence'] == 'Low')}")
    else:
        logger.error("No markets found for prediction")


if __name__ == "__main__":
    main()
