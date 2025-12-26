"""
Feature Engineering for Polymarket Prediction Model
Transforms raw market data into ML-ready features
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketFeatureEngineering:
    """Engineer features from Polymarket data for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath: str) -> List[Dict]:
        """Load collected market data"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} markets from {filepath}")
        return data
    
    def create_technical_indicators(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""
        df = timeseries_df.copy()
        
        if df.empty:
            return df
        
        for window in [3, 6, 12, 24]:  # hours
            if 'close' in df.columns:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        if 'close' in df.columns:
            df['momentum_1h'] = df['close'].pct_change(1)
            df['momentum_6h'] = df['close'].pct_change(6)
            df['momentum_24h'] = df['close'].pct_change(24)
        
        if 'close' in df.columns:
            df['volatility_6h'] = df['close'].rolling(window=6).std()
            df['volatility_24h'] = df['close'].rolling(window=24).std()
        
        if 'volume' in df.columns:
            df['volume_sma_6h'] = df['volume'].rolling(window=6).mean()
            df['volume_change'] = df['volume'].pct_change()
        
        if 'high' in df.columns and 'low' in df.columns:
            df['price_range'] = df['high'] - df['low']
            df['price_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        
        if 'close' in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        #bllinger Bands
        if 'close' in df.columns:
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        return df
    
    def create_time_features(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = timeseries_df.copy()
        
        if 'timestamp' not in df.columns:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_market_features(self, market_data: Dict) -> Dict:
        """Create features from market metadata"""
        market_info = market_data.get('market_info', {})
        features = market_data.get('features', {})
        
        market_features = {
            'total_trades': features.get('total_trades', 0),
            'total_volume': features.get('total_volume', 0),
            'avg_price': features.get('avg_price', 0),
            'price_std': features.get('price_std', 0),
            'price_range': features.get('price_range', 0),
            'trades_per_hour': features.get('trades_per_hour', 0),
            'time_span_hours': features.get('time_span_hours', 0),
            
            'num_tokens': len(market_info.get('tokens', [])),
            'is_active': int(market_info.get('active', False)),
            'is_closed': int(market_info.get('closed', False)),
        }
        
        return market_features
    
    def create_orderbook_features(self, orderbooks: Dict) -> Dict:
        """Extract features from orderbook data"""
        features = {}
        
        for token_id, orderbook in orderbooks.items():
            if not orderbook:
                continue
            
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if bids:
                features[f'{token_id}_best_bid'] = float(bids[0].get('price', 0)) if bids else 0
                features[f'{token_id}_bid_volume'] = sum(float(b.get('size', 0)) for b in bids[:5])
            
            if asks:
                features[f'{token_id}_best_ask'] = float(asks[0].get('price', 0)) if asks else 0
                features[f'{token_id}_ask_volume'] = sum(float(a.get('size', 0)) for a in asks[:5])
            
            # Spread
            if bids and asks:
                best_bid = float(bids[0].get('price', 0))
                best_ask = float(asks[0].get('price', 0))
                features[f'{token_id}_spread'] = best_ask - best_bid
                features[f'{token_id}_spread_pct'] = (features[f'{token_id}_spread'] / best_ask * 100) if best_ask > 0 else 0
        
        return features
    
    def create_target_variable(self, timeseries_df: pd.DataFrame, 
                              horizon: int = 6, 
                              threshold: float = 0.05) -> pd.Series:
        """
        Create target variable for prediction
        horizon: hours into the future to predict
        threshold: price change threshold for binary classification
        """
        if 'close' not in timeseries_df.columns or timeseries_df.empty:
            return pd.Series()
        
        future_price = timeseries_df['close'].shift(-horizon)
        current_price = timeseries_df['close']
        
        price_change_pct = ((future_price - current_price) / current_price) * 100
        
        target = (price_change_pct > threshold).astype(int)
        
        timeseries_df['target_price_change'] = price_change_pct
        timeseries_df['target_binary'] = target
        
        return target
    
    def prepare_training_data(self, datasets: List[Dict], 
                             prediction_horizon: int = 6) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare complete training dataset from all markets
        Returns: (features_df, target_series)
        """
        all_samples = []
        
        for dataset in datasets:
            try:
                timeseries = pd.DataFrame(dataset.get('timeseries', []))
                
                if timeseries.empty:
                    continue
                
                timeseries = self.create_technical_indicators(timeseries)
                
                timeseries = self.create_time_features(timeseries)
                
                self.create_target_variable(timeseries, horizon=prediction_horizon)
                
                market_features = self.create_market_features(dataset)
                for key, value in market_features.items():
                    timeseries[f'market_{key}'] = value
                
                orderbook_features = self.create_orderbook_features(dataset.get('orderbooks', {}))
                for key, value in orderbook_features.items():
                    timeseries[f'ob_{key}'] = value
                
                timeseries['condition_id'] = dataset['market_info'].get('condition_id', 'unknown')
                
                all_samples.append(timeseries)
                
            except Exception as e:
                logger.error(f"Error processing dataset: {e}")
                continue
        
        if not all_samples:
            logger.error("No samples created!")
            return pd.DataFrame(), pd.Series()
        
        # Combine all markets
        full_df = pd.concat(all_samples, ignore_index=True)
        
        # Remove rows with NaN target
        full_df = full_df.dropna(subset=['target_binary'])
        
        logger.info(f"Created dataset with {len(full_df)} samples")
        
        return full_df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude target and metadata)"""
        exclude_cols = ['target_binary', 'target_price_change', 'timestamp', 
                       'condition_id', 'open', 'high', 'low', 'close']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining NaN columns
        feature_cols = [col for col in feature_cols if df[col].notna().sum() > 0]
        
        return feature_cols
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str = "polymarket_data/processed_training_data.csv"):
        """Save processed data to CSV"""
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
        
        feature_cols = self.get_feature_columns(df)
        feature_names_path = filepath.replace('.csv', '_features.txt')
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(feature_cols))
        logger.info(f"Saved feature names to {feature_names_path}")


def main():
    """Main execution"""
    engineer = PolymarketFeatureEngineering()
    
    datasets = engineer.load_data("polymarket_data/polymarket_training_data.json")
    
    logger.info("Engineering features...")
    training_df = engineer.prepare_training_data(datasets, prediction_horizon=6)
    
    if not training_df.empty:
        engineer.save_processed_data(training_df)
        
        logger.info(f"\nDataset Summary:")
        logger.info(f"Total samples: {len(training_df)}")
        logger.info(f"Feature columns: {len(engineer.get_feature_columns(training_df))}")
        logger.info(f"Target distribution:\n{training_df['target_binary'].value_counts()}")
    else:
        logger.error("Failed to create training data!")


if __name__ == "__main__":
    main()
