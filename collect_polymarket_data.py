"""
Polymarket Data Collection Script
Collects historical trade data and market snapshots for ML model training
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PolymarketDataCollector:
    """Collects historical data from Polymarket for ML training"""
    
    def __init__(self):
        self.clob_base = "https://clob.polymarket.com"
        self.gamma_base = "https://gamma-api.polymarket.com"
        self.data_dir = "polymarket_data"
        
    def get_active_markets(self, limit: int = 100) -> List[Dict]:
        """Fetch active markets from Polymarket"""
        
        # Try CLOB API first
        url = f"{self.clob_base}/markets"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            # Handle different response formats
            if isinstance(data, list):
                markets = data
            elif isinstance(data, dict):
                markets = data.get('data', data.get('markets', []))
                if not isinstance(markets, list):
                    markets = list(data.values()) if data else []
            else:
                markets = []
            
            if markets:
                logger.info(f"Retrieved {len(markets)} markets from CLOB API")
                return markets[:limit]
        except Exception as e:
            logger.warning(f"CLOB API failed: {e}, trying Gamma API...")
        
        # Fallback to Gamma API
        try:
            url = f"{self.gamma_base}/markets"
            resp = requests.get(url, params={'limit': limit, 'active': 'true'}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list):
                markets = data
            elif isinstance(data, dict):
                markets = data.get('data', data.get('markets', []))
            else:
                markets = []
            
            logger.info(f"Retrieved {len(markets)} markets from Gamma API")
            return markets[:limit]
        except Exception as e:
            logger.error(f"Both APIs failed. Error: {e}")
            return []
    
    def get_markets_by_events(self, event_slugs: List[str]) -> List[Dict]:
        """Fetch markets from specific events (more reliable)"""
        all_markets = []
        
        for slug in event_slugs:
            try:
                url = f"{self.gamma_base}/events/slug/{slug}"
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                event = resp.json()
                
                markets = event.get('markets', [])
                all_markets.extend(markets)
                logger.info(f"Retrieved {len(markets)} markets from event: {slug}")
                
            except Exception as e:
                logger.warning(f"Could not get event {slug}: {e}")
        
        return all_markets
    
    def get_market_trades(self, condition_id: str, limit: int = 1000) -> List[Dict]:
        """Fetch historical trades for a specific market"""
        url = f"{self.clob_base}/trades"
        params = {
            "condition_id": condition_id,
            "limit": limit
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            trades = resp.json()
            logger.info(f"Retrieved {len(trades)} trades for {condition_id}")
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades for {condition_id}: {e}")
            return []
    
    def get_orderbook(self, token_id: str) -> Dict:
        """Get current orderbook state for a token"""
        url = f"{self.clob_base}/book"
        params = {"token_id": token_id}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return {}
    
    def get_market_prices_timeseries(self, condition_id: str, days_back: int = 30) -> pd.DataFrame:
        """
        Construct price timeseries from trades data
        """
        trades = self.get_market_trades(condition_id, limit=10000)
        
        if not trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(trades)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp')
        
        # Extract price (usually in the 'price' field)
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Calculate volume metrics
        if 'size' in df.columns:
            df['size'] = pd.to_numeric(df['size'], errors='coerce')
            df['cumulative_volume'] = df['size'].cumsum()
        
        return df
    
    def calculate_market_features(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate statistical features from trades data"""
        if trades_df.empty:
            return {}
        
        features = {
            'total_trades': len(trades_df),
            'total_volume': trades_df['size'].sum() if 'size' in trades_df else 0,
            'avg_price': trades_df['price'].mean() if 'price' in trades_df else 0,
            'price_std': trades_df['price'].std() if 'price' in trades_df else 0,
            'price_min': trades_df['price'].min() if 'price' in trades_df else 0,
            'price_max': trades_df['price'].max() if 'price' in trades_df else 0,
            'price_range': (trades_df['price'].max() - trades_df['price'].min()) if 'price' in trades_df else 0,
        }
        
        # Calculate price momentum
        if 'price' in trades_df and len(trades_df) > 1:
            features['price_change'] = trades_df['price'].iloc[-1] - trades_df['price'].iloc[0]
            features['price_change_pct'] = (features['price_change'] / trades_df['price'].iloc[0]) * 100
        
        # Calculate time-based features
        if 'timestamp' in trades_df:
            features['time_span_hours'] = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).total_seconds() / 3600
            features['trades_per_hour'] = features['total_trades'] / max(features['time_span_hours'], 1)
        
        return features
    
    def resample_to_timeseries(self, trades_df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
        """
        Resample trades to regular time intervals
        freq: pandas frequency string like '1H', '4H', '1D'
        """
        if trades_df.empty or 'timestamp' not in trades_df:
            return pd.DataFrame()
        
        trades_df = trades_df.set_index('timestamp')
        
        resampled = pd.DataFrame()
        
        if 'price' in trades_df:
            resampled['open'] = trades_df['price'].resample(freq).first()
            resampled['high'] = trades_df['price'].resample(freq).max()
            resampled['low'] = trades_df['price'].resample(freq).min()
            resampled['close'] = trades_df['price'].resample(freq).last()
            resampled['mean'] = trades_df['price'].resample(freq).mean()
        
        if 'size' in trades_df:
            resampled['volume'] = trades_df['size'].resample(freq).sum()
            resampled['num_trades'] = trades_df['size'].resample(freq).count()
        
        # Forward fill missing values
        resampled = resampled.fillna(method='ffill')
        
        return resampled.reset_index()
    
    def collect_market_dataset(self, market: Dict, resample_freq: str = '1H') -> Dict[str, Any]:
        """Collect complete dataset for a single market"""
        condition_id = market.get('condition_id')
        question = market.get('question', 'Unknown')
        
        logger.info(f"Collecting data for: {question[:70]}")
        
        # Get trades
        trades_df = self.get_market_prices_timeseries(condition_id)
        
        if trades_df.empty:
            logger.warning(f"No trades found for {condition_id}")
            return None
        
        # Calculate features
        features = self.calculate_market_features(trades_df)
        
        # Resample to time series
        timeseries = self.resample_to_timeseries(trades_df, freq=resample_freq)
        
        # Get current orderbook
        tokens = market.get('tokens', [])
        orderbooks = {}
        for token in tokens:
            token_id = token.get('token_id')
            if token_id:
                orderbooks[token_id] = self.get_orderbook(token_id)
                time.sleep(0.5)  # Rate limiting
        
        dataset = {
            'market_info': market,
            'features': features,
            'raw_trades': trades_df.to_dict('records'),
            'timeseries': timeseries.to_dict('records'),
            'orderbooks': orderbooks,
            'collection_timestamp': datetime.now().isoformat()
        }
        
        return dataset
    
    def collect_multiple_markets(self, num_markets: int = 20, resample_freq: str = '1H') -> List[Dict]:
        """Collect datasets for multiple markets"""
        markets = self.get_active_markets(limit=num_markets)
        
        datasets = []
        for i, market in enumerate(markets):
            logger.info(f"Processing market {i+1}/{len(markets)}")
            
            try:
                dataset = self.collect_market_dataset(market, resample_freq)
                if dataset:
                    datasets.append(dataset)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                continue
        
        return datasets
    
    def save_datasets(self, datasets: List[Dict], filename: str = "polymarket_training_data.json"):
        """Save collected datasets to JSON file"""
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(datasets, f, indent=2)
        
        logger.info(f"Saved {len(datasets)} datasets to {filepath}")
        
        # Also save as CSV for easy inspection
        csv_filepath = filepath.replace('.json', '_summary.csv')
        summary_data = []
        for dataset in datasets:
            summary = {
                'question': dataset['market_info'].get('question', 'N/A'),
                'condition_id': dataset['market_info'].get('condition_id', 'N/A'),
                **dataset['features']
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_filepath, index=False)
        logger.info(f"Saved summary to {csv_filepath}")
        
        return filepath


def main():
    """Main execution function"""
    collector = PolymarketDataCollector()
    
    # Try to collect data
    logger.info("Starting data collection...")
    datasets = collector.collect_multiple_markets(num_markets=20, resample_freq='1H')
    
    # If no data, try event-based approach
    if not datasets:
        logger.info("Trying event-based collection...")
        
        # List of popular/known event slugs (update these with current events)
        event_slugs = [
            "presidential-election-winner-2024",
            "will-biden-be-the-2024-democratic-nominee",
            "federal-reserve-interest-rate-decision",
            "ukraine-russia-conflict",
            "2024-olympics"
        ]
        
        markets = collector.get_markets_by_events(event_slugs)
        
        if markets:
            logger.info(f"Found {len(markets)} markets from events")
            datasets = []
            
            for i, market in enumerate(markets[:20]):  # Limit to 20
                logger.info(f"Processing market {i+1}/{min(len(markets), 20)}")
                try:
                    dataset = collector.collect_market_dataset(market, resample_freq='1H')
                    if dataset:
                        datasets.append(dataset)
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing market: {e}")
                    continue
    
    if datasets:
        filepath = collector.save_datasets(datasets)
        logger.info(f"Data collection complete! {len(datasets)} markets saved.")
        logger.info(f"Data saved to: {filepath}")
    else:
        logger.error("No data collected! Please check:")
        logger.error("1. Internet connection")
        logger.error("2. Polymarket API status")
        logger.error("3. Try updating event_slugs with current events from polymarket.com")


if __name__ == "__main__":
    main()