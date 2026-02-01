'''import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Unified data fetcher for stocks, crypto, and mutual funds"""
    
    @staticmethod
    def get_historical_data(symbol: str, period: str = '3mo', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical data for any asset type
        
        Args:
            symbol: Asset symbol (e.g., 'RELIANCE.NS', 'BTC-USD', '0P0000XVLO.BO')
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with historical data
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            return hist
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def get_current_price(symbol: str) -> Optional[Dict]:
        """
        Get current/latest price for an asset
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with current price data or None
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='1m')
            
            if hist.empty:
                # Try with larger period if 1d fails
                hist = ticker.history(period='5d', interval='1d')
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'timestamp': hist.index[-1].isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return None
    
    @staticmethod
    def get_asset_info(symbol: str) -> Dict:
        """
        Get detailed information about an asset
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with asset information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        
        except Exception as e:
            logger.error(f"Error fetching asset info for {symbol}: {str(e)}")
            return {}
    
    @staticmethod
    def search_assets(query: str, category: str = 'all', max_results: int = 10) -> List[Dict]:
        """
        Search for assets based on query
        
        Args:
            query: Search query
            category: Asset category (stocks, crypto, mutualfunds, all)
            max_results: Maximum number of results
            
        Returns:
            List of matching assets
        """
        try:
            from config import config
            
            results = []
            query_lower = query.lower()
            
            # Search in stocks
            if category in ['stocks', 'all']:
                for stock in config.POPULAR_INDIAN_STOCKS:
                    if query_lower in stock.lower():
                        results.append({
                            'symbol': stock,
                            'name': stock.replace('.NS', '').replace('.BO', ''),
                            'category': 'stocks',
                            'exchange': 'NSE' if '.NS' in stock else 'BSE'
                        })
            
            # Search in crypto
            if category in ['crypto', 'all']:
                for crypto in config.POPULAR_CRYPTOS:
                    if query_lower in crypto.lower():
                        results.append({
                            'symbol': crypto,
                            'name': crypto.replace('-USD', ''),
                            'category': 'crypto',
                            'exchange': 'Crypto'
                        })
            
            # Search in mutual funds
            if category in ['mutualfunds', 'all']:
                for mf in config.POPULAR_MUTUAL_FUNDS:
                    if query_lower in mf.lower():
                        results.append({
                            'symbol': mf,
                            'name': mf.replace('.BO', ''),
                            'category': 'mutualfunds',
                            'exchange': 'BSE'
                        })
            
            # If no results from predefined lists, use yfinance search
            if len(results) == 0:
                try:
                    search_result = yf.Search(query, max_results=max_results)
                    if hasattr(search_result, 'quotes') and search_result.quotes:
                        for quote in search_result.quotes[:max_results]:
                            results.append({
                                'symbol': quote.get('symbol', ''),
                                'name': quote.get('longname', quote.get('shortname', '')),
                                'category': DataFetcher._determine_category(quote.get('symbol', '')),
                                'exchange': quote.get('exchange', '')
                            })
                except Exception as e:
                    logger.error(f"Error in yfinance search: {str(e)}")
            
            return results[:max_results]
        
        except Exception as e:
            logger.error(f"Error searching assets: {str(e)}")
            return []
    
    @staticmethod
    def _determine_category(symbol: str) -> str:
        """Determine asset category from symbol"""
        if '-USD' in symbol or symbol.startswith('BTC') or symbol.startswith('ETH'):
            return 'crypto'
        elif '.NS' in symbol or '.BO' in symbol:
            if symbol.startswith('0P'):
                return 'mutualfunds'
            return 'stocks'
        return 'unknown'
    
    @staticmethod
    def get_multiple_prices(symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple assets
        
        Args:
            symbols: List of asset symbols
            
        Returns:
            Dictionary with symbol as key and price data as value
        """
        results = {}
        
        for symbol in symbols:
            price_data = DataFetcher.get_current_price(symbol)
            if price_data:
                results[symbol] = price_data
            else:
                results[symbol] = {
                    'symbol': symbol,
                    'error': 'Unable to fetch price'
                }
        
        return results'''

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Unified data fetcher for stocks, crypto, and mutual funds"""
    
    @staticmethod
    def get_historical_data(symbol: str, period: str = '3mo', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical data for any asset type
        """
        try:
            ticker = yf.Ticker(symbol)
            # Update: Mutual funds (0P...) often fail on small intervals; 
            # ensure we use '1d' if it's a mutual fund
            if symbol.startswith('0P'):
                interval = '1d'
                
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            return hist
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def get_current_price(symbol: str) -> Optional[Dict]:
        """
        Get current/latest price for an asset with fallback logic
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Update: Mutual Funds (0P...) do not support 1m intervals.
            # We force 1d interval and a 5d period to ensure we get the latest NAV.
            is_mf = symbol.startswith('0P')
            interval = '1d' if is_mf else '1m'
            period = '5d' if is_mf else '1d'
            
            hist = ticker.history(period=period, interval=interval)
            
            # If 1m failed for a stock, fallback to 1d/5d
            if hist.empty and not is_mf:
                hist = ticker.history(period='5d', interval='1d')
            
            if hist.empty:
                logger.warning(f"No price data available for {symbol}")
                return None
            
            latest = hist.iloc[-1]
            
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'timestamp': hist.index[-1].isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return None
    
    @staticmethod
    def get_asset_info(symbol: str) -> Dict:
        """
        Get detailed information about an asset
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            logger.error(f"Error fetching asset info for {symbol}: {str(e)}")
            return {}
    
    @staticmethod
    def search_assets(query: str, category: str = 'all', max_results: int = 10) -> List[Dict]:
        """
        Search for assets based on query with category filtering
        """
        try:
            from config import config
            
            results = []
            query_lower = query.lower()
            
            # Search in stocks
            if category in ['stocks', 'all']:
                for stock in config.POPULAR_INDIAN_STOCKS:
                    if query_lower in stock.lower():
                        results.append({
                            'symbol': stock,
                            'name': stock.replace('.NS', '').replace('.BO', ''),
                            'category': 'stocks',
                            'exchange': 'NSE' if '.NS' in stock else 'BSE'
                        })
            
            # Search in crypto
            if category in ['crypto', 'all']:
                for crypto in config.POPULAR_CRYPTOS:
                    if query_lower in crypto.lower():
                        results.append({
                            'symbol': crypto,
                            'name': crypto.replace('-USD', ''),
                            'category': 'crypto',
                            'exchange': 'Crypto'
                        })
            
            # Search in mutual funds
            if category in ['mutualfunds', 'all']:
                for mf in config.POPULAR_MUTUAL_FUNDS:
                    if query_lower in mf.lower():
                        results.append({
                            'symbol': mf,
                            'name': mf.replace('.BO', ''),
                            'category': 'mutualfunds',
                            'exchange': 'BSE'
                        })
            
            # Fallback to yfinance search if predefined list yields few results
            if len(results) < max_results:
                try:
                    search_result = yf.Search(query, max_results=max_results)
                    if hasattr(search_result, 'quotes') and search_result.quotes:
                        for quote in search_result.quotes:
                            sym = quote.get('symbol', '')
                            # Avoid duplicates
                            if any(r['symbol'] == sym for r in results):
                                continue
                                
                            results.append({
                                'symbol': sym,
                                'name': quote.get('longname', quote.get('shortname', sym)),
                                'category': DataFetcher._determine_category(sym),
                                'exchange': quote.get('exchange', '')
                            })
                except Exception as e:
                    logger.error(f"Error in yfinance search: {str(e)}")
            
            return results[:max_results]
        
        except Exception as e:
            logger.error(f"Error searching assets: {str(e)}")
            return []
    
    @staticmethod
    def _determine_category(symbol: str) -> str:
        """Determine asset category from symbol"""
        if '-USD' in symbol or any(c in symbol for c in ['BTC', 'ETH', 'SOL', 'XRP']):
            return 'crypto'
        elif '.NS' in symbol or '.BO' in symbol:
            if symbol.startswith('0P'):
                return 'mutualfunds'
            return 'stocks'
        return 'unknown'
    
    @staticmethod
    def get_multiple_prices(symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple assets
        """
        results = {}
        for symbol in symbols:
            price_data = DataFetcher.get_current_price(symbol)
            if price_data:
                results[symbol] = price_data
            else:
                results[symbol] = {
                    'symbol': symbol,
                    'error': 'Data currently unavailable'
                }
        return results