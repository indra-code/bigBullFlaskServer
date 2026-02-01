import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # API Keys
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
    
    # Redis Configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # ML Configuration
    MODEL_RETRAIN_INTERVAL_DAYS = int(os.getenv('MODEL_RETRAIN_INTERVAL_DAYS', 7))
    PREDICTION_HORIZON_DAYS = int(os.getenv('PREDICTION_HORIZON_DAYS', 30))
    
    # Risk Analysis Configuration
    RISK_LOOKBACK_PERIOD_DAYS = int(os.getenv('RISK_LOOKBACK_PERIOD_DAYS', 90))
    
    # Period and interval mappings for historical data
    PERIOD_INTERVAL_MAP = {
        '1D': {'period': '1d', 'interval': '1m'},
        '5D': {'period': '5d', 'interval': '5m'},
        '1M': {'period': '1mo', 'interval': '30m'},
        '3M': {'period': '3mo', 'interval': '1d'},
        '6M': {'period': '6mo', 'interval': '1d'},
        '1Y': {'period': '1y', 'interval': '1d'},
        '5Y': {'period': '5y', 'interval': '1wk'},
        'MAX': {'period': 'max', 'interval': '1mo'}
    }
    
    # Asset Categories
    ASSET_CATEGORIES = ['stocks', 'crypto', 'mutualfunds']
    
    # Indian Stock Exchanges suffixes
    INDIAN_STOCK_SUFFIX = '.NS'  # NSE
    INDIAN_STOCK_SUFFIX_BSE = '.BO'  # BSE
    
    # Popular Indian Stocks (for autocomplete)
    POPULAR_INDIAN_STOCKS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'ASIANPAINT.NS', 'AXISBANK.NS', 'LT.NS', 'WIPRO.NS', 'MARUTI.NS',
        'BAJFINANCE.NS', 'HCLTECH.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS'
    ]
    
    # Popular Cryptocurrencies (for autocomplete)
    POPULAR_CRYPTOS = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
        'DOGE-USD', 'SOL-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD',
        'SHIB-USD', 'TRX-USD', 'AVAX-USD', 'UNI-USD', 'LINK-USD',
        'ATOM-USD', 'XLM-USD', 'ALGO-USD', 'VET-USD', 'FIL-USD'
    ]
    
    # Popular Mutual Funds in India (for autocomplete)
    POPULAR_MUTUAL_FUNDS = [
        '0P0000XVLO.BO',  # SBI Bluechip Fund
        '0P00013IZ6.BO',  # HDFC Top 100 Fund
        '0P0000XVH1.BO',  # ICICI Prudential Bluechip Fund
        '0P0000XV3C.BO',  # Axis Bluechip Fund
        '0P0000XUYT.BO',  # Mirae Asset Large Cap Fund
        '0P00018QX7.BO',  # Parag Parikh Flexi Cap Fund
        '0P0000XUYQ.BO',  # Kotak Standard Multicap Fund
        '0P00013J07.BO',  # SBI Small Cap Fund
        '0P0000XVKI.BO',  # Axis Midcap Fund
        '0P0000XV0R.BO',  # HDFC Mid-Cap Opportunities Fund
    ]

config = Config()