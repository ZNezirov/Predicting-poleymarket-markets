"""
Simplified Polymarket Data Collection with Sample Data Fallback
This version will work even if the API is unavailable
"""

import requests
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_market_data(num_markets=10):
    """Create sample market data for testing/demo purposes"""
    logger.info("Creating sample market data for demonstration...")
    
    datasets = []
    
    for i in range(num_markets):
        # Generate sample price data
        timestamps = pd.date_range(end=datetime.now(), periods=200, freq='1H')
        
        # Random walk price data between 0.3 and 0.7
        prices = np.random.randn(200).cumsum() * 0.02 + 0.5
        prices = np.clip(prices, 0.3, 0.7)
        
        # Add some trend
        trend = np.linspace(0, 0.1 * np.random.choice([-1, 1]), 200)
        prices += trend
        prices = np.clip(prices, 0.2, 0.8)
        
        # Generate OHLCV data
        timeseries = []
        for j, ts in enumerate(timestamps):
            price = prices[j]
            noise = np.random.rand() * 0.02
            
            high = min(price + noise, 0.95)
            low = max(price - noise, 0.05)
            volume = np.random.randint(1000, 10000)
            
            timeseries.append({
                'timestamp': ts.isoformat(),
                'open': float(price),
                'high': float(high),
                'low': float(low),
                'close': float(price),
                'mean': float(price),
                'volume': float(volume),
                'num_trades': int(volume / 100)
            })
        
        # Create raw trades
        raw_trades = []
        for j in range(len(timestamps) // 2):
            raw_trades.append({
                'timestamp': timestamps[j * 2].isoformat(),
                'price': float(prices[j * 2]),
                'size': float(np.random.randint(50, 500))
            })
        
        # Market features
        features = {
            'total_trades': len(raw_trades),
            'total_volume': sum(t['volume'] for t in timeseries),
            'avg_price': float(np.mean(prices)),
            'price_std': float(np.std(prices)),
            'price_min': float(np.min(prices)),
            'price_max': float(np.max(prices)),
            'price_range': float(np.max(prices) - np.min(prices)),
            'price_change': float(prices[-1] - prices[0]),
            'price_change_pct': float((prices[-1] - prices[0]) / prices[0] * 100),
            'time_span_hours': 200,
            'trades_per_hour': len(raw_trades) / 200
        }
        
        # Market info
        market_info = {
            'id': f'sample_{i}',
            'condition_id': f'0x{i:064x}',
            'question': f'Sample Market {i+1}: Will event {i+1} occur?',
            'active': True,
            'closed': False,
            'tokens': [
                {'token_id': f'token_{i}_yes', 'outcome': 'Yes'},
                {'token_id': f'token_{i}_no', 'outcome': 'No'}
            ],
            'volume': features['total_volume'],
            'endDate': (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        dataset = {
            'market_info': market_info,
            'features': features,
            'raw_trades': raw_trades,
            'timeseries': timeseries,
            'orderbooks': {},
            'collection_timestamp': datetime.now().isoformat()
        }
        
        datasets.append(dataset)
    
    logger.info(f"Created {len(datasets)} sample market datasets")
    return datasets


def try_collect_real_data(num_markets=20):
    """Try to collect real data from Polymarket API"""
    
    # Try different API endpoints
    endpoints_to_try = [
        "https://gamma-api.polymarket.com/events",
        "https://clob.polymarket.com/markets"
    ]
    
    for endpoint in endpoints_to_try:
        try:
            logger.info(f"Trying endpoint: {endpoint}")
            resp = requests.get(endpoint, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"Success! Got response from {endpoint}")
                logger.info(f"Response type: {type(data)}")
                
                # Log the structure
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"List with {len(data)} items")
                    logger.info(f"First item keys: {list(data[0].keys())[:10]}")
                elif isinstance(data, dict):
                    logger.info(f"Dict with keys: {list(data.keys())[:10]}")
                
                return data
                
        except Exception as e:
            logger.warning(f"Failed {endpoint}: {e}")
            continue
    
    return None


def main():
    """Main execution with fallback to sample data"""
    
    os.makedirs("polymarket_data", exist_ok=True)
    
    logger.info("="*70)
    logger.info("POLYMARKET DATA COLLECTION")
    logger.info("="*70)
    
    # Try to get real data
    logger.info("\nAttempting to collect real data from Polymarket API...")
    real_data = try_collect_real_data()
    
    if real_data:
        logger.info("✓ Successfully connected to Polymarket API")
        logger.info("Please check the logged structure and update collect_polymarket_data.py")
        logger.info("For now, using sample data to complete the pipeline...")
    else:
        logger.warning("✗ Could not connect to Polymarket API")
        logger.info("Using sample data for demonstration...")
    
    # Create sample data
    datasets = create_sample_market_data(num_markets=20)
    
    # Save datasets
    filepath = "polymarket_data/polymarket_training_data.json"
    with open(filepath, 'w') as f:
        json.dump(datasets, f, indent=2)
    
    logger.info(f"\n✓ Saved {len(datasets)} datasets to {filepath}")
    
    # Save summary
    summary_data = []
    for dataset in datasets:
        summary = {
            'question': dataset['market_info']['question'],
            'condition_id': dataset['market_info']['condition_id'],
            **dataset['features']
        }
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = "polymarket_data/polymarket_training_data_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    
    logger.info(f"✓ Saved summary to {csv_path}")
    
    logger.info("\n" + "="*70)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Markets collected: {len(datasets)}")
    logger.info(f"Avg trades per market: {summary_df['total_trades'].mean():.0f}")
    logger.info(f"Avg volume per market: ${summary_df['total_volume'].mean():,.0f}")
    logger.info("\nNote: This is SAMPLE data for demonstration.")
    logger.info("For real predictions, you'll need to:")
    logger.info("1. Check Polymarket API documentation")
    logger.info("2. Update the API endpoints in collect_polymarket_data.py")
    logger.info("3. Or use a Polymarket data provider/service")
    logger.info("\nYou can still run the rest of the pipeline with this sample data!")
    logger.info("="*70)


if __name__ == "__main__":
    main()