from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
import yfinance as yf
import asyncio
import json
from datetime import datetime, timedelta
import threading
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import finnhub
import re
import time

from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import RealAmplitudes
from qiskit.result import QuasiDistribution

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Finnhub API Configuration
FINNHUB_API_KEY = 'd5ur29pr01qr4f8a5t20d5ur29pr01qr4f8a5t2g'
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Period and interval mappings
PERIOD_INTERVAL_MAP = {
    '1D': {'period': '1d', 'interval': '1m'},
    '5D': {'period': '5d', 'interval': '5m'},
    '1M': {'period': '1mo', 'interval': '30m'},
    '3M': {'period': '3mo', 'interval': '5d'},
    '6M': {'period': '6mo', 'interval': '1d'},
    '1Y': {'period': '1y', 'interval': '1d'},
    '5Y': {'period': '5y', 'interval': '1wk'},
    'MAX': {'period': 'max', 'interval': '1mo'}
}

# WebSocket connections storage
active_ws_connections = {}

# Cache for search results
search_cache = {}
cache_expiry = {}


# ==================== HEALTH ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Python API is running',
        'timestamp': datetime.now().isoformat()
    }), 200


# ==================== STOCK ENDPOINTS ====================

@app.route('/api/stock/quote/<symbol>', methods=['GET'])
def get_stock_quote(symbol):
    """Get current stock quote with live price"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        previous_close = info.get('previousClose', current_price)
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close != 0 else 0

        return jsonify({
            'symbol': symbol,
            'price': round(current_price, 2),
            'change': round(change, 2),
            'changePercent': round(change_percent, 2),
            'timestamp': datetime.now().isoformat(),
            'open': info.get('open', current_price),
            'high': info.get('dayHigh', current_price),
            'low': info.get('dayLow', current_price),
            'volume': info.get('volume', 0),
            'marketCap': info.get('marketCap', 0)
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Failed to fetch stock quote: {str(e)}',
            'symbol': symbol
        }), 500


@app.route('/api/stock/history/<symbol>', methods=['GET'])
def get_stock_history(symbol):
    """Get stock historical data"""
    try:
        timeframe = request.args.get('timeframe', '1M').upper()
        period = request.args.get('period')
        interval = request.args.get('interval')

        if period and interval:
            config = {'period': period, 'interval': interval}
        else:
            config = PERIOD_INTERVAL_MAP.get(timeframe, PERIOD_INTERVAL_MAP['1M'])

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=config['period'], interval=config['interval'])

        if hist.empty:
            return jsonify({'error': 'No data available', 'symbol': symbol}), 404

        hist.reset_index(inplace=True)
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'period': config['period'],
            'interval': config['interval'],
            'data': hist.to_dict(orient='records')
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Failed to fetch stock history: {str(e)}',
            'symbol': symbol
        }), 500


@app.route('/api/stock/info/<symbol>', methods=['GET'])
def get_stock_info(symbol):
    """Get detailed stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return jsonify({
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A'),
            'website': info.get('website', 'N/A'),
            'employees': info.get('fullTimeEmployees', 0),
            'marketCap': info.get('marketCap', 0),
            'peRatio': info.get('trailingPE', 0),
            'dividendYield': info.get('dividendYield', 0),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', 0),
            '52WeekLow': info.get('fiftyTwoWeekLow', 0)
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Failed to fetch stock info: {str(e)}',
            'symbol': symbol
        }), 500


@app.route('/api/search', methods=['GET'])
@app.route('/api/stock/search', methods=['GET'])
def universal_search():
    """
    Universal search for ANY asset type supported by yfinance
    Supports: Stocks, Crypto, Mutual Funds, Commodities, ETFs, Bonds, Indices, etc.
    Query params:
    - query: search query (required) - symbol or name
    - max_results: maximum results (default: 10)
    - include_news: include recent news (default: false)
    - include_full_info: include all available info (default: false)
    """
    try:
        query = request.args.get('query', '').strip()
        if not query or len(query) < 1:
            return jsonify({'error': 'Query must be at least 1 character'}), 400

        max_results = int(request.args.get('max_results', 10))
        include_news = request.args.get('include_news', 'false').lower() == 'true'
        include_full_info = request.args.get('include_full_info', 'false').lower() == 'true'

        # Check cache
        cache_key = f"{query}_{max_results}_{include_news}_{include_full_info}"
        if cache_key in search_cache:
            expiry = cache_expiry.get(cache_key)
            if expiry and datetime.now() < expiry:
                return jsonify(search_cache[cache_key]), 200

        results = []

        # Primary search using yfinance
        try:
            search_result = yf.Search(query, max_results=max_results * 3)

            if hasattr(search_result, 'response') and search_result.response:
                for item in search_result.response:
                    if len(results) >= max_results:
                        break

                    symbol = item.get('symbol', 'N/A')
                    name = item.get('name', 'N/A')
                    exchange = item.get('exchange', 'N/A')
                    instrument_type = item.get('instrumentType', 'N/A')
                    quote_type = item.get('quoteType', 'N/A')

                    # Get comprehensive data for the symbol
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        # Get price data
                        hist = ticker.history(period='5d')
                        
                        if not hist.empty:
                            latest = hist.iloc[-1]
                            current_price = float(latest['Close'])
                            open_price = float(latest['Open'])
                            high_price = float(latest['High'])
                            low_price = float(latest['Low'])
                            volume = int(latest['Volume'])
                            change_pct = ((current_price - open_price) / open_price * 100) if open_price != 0 else 0
                        else:
                            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
                            open_price = info.get('open', 'N/A')
                            high_price = info.get('dayHigh', 'N/A')
                            low_price = info.get('dayLow', 'N/A')
                            volume = info.get('volume', 'N/A')
                            change_pct = info.get('regularMarketChangePercent', 0)
                    except Exception as e:
                        print(f"Error fetching data for {symbol}: {e}")
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
                        open_price = info.get('open', 'N/A')
                        high_price = info.get('dayHigh', 'N/A')
                        low_price = info.get('dayLow', 'N/A')
                        volume = info.get('volume', 'N/A')
                        change_pct = 0

                    result_item = {
                        'symbol': symbol,
                        'name': name,
                        'type': instrument_type or quote_type,
                        'exchange': exchange,
                        'current_price': current_price if isinstance(current_price, str) else round(float(current_price), 2) if current_price != 'N/A' else 'N/A',
                        'open': open_price if isinstance(open_price, str) else round(float(open_price), 2) if open_price != 'N/A' else 'N/A',
                        'high': high_price if isinstance(high_price, str) else round(float(high_price), 2) if high_price != 'N/A' else 'N/A',
                        'low': low_price if isinstance(low_price, str) else round(float(low_price), 2) if low_price != 'N/A' else 'N/A',
                        'volume': volume,
                        'change_percent': round(change_pct, 2) if isinstance(change_pct, (int, float)) else 0,
                        'currency': info.get('currency', 'USD')
                    }

                    # Add full info if requested
                    if include_full_info:
                        full_info = {
                            'sector': info.get('sector', 'N/A'),
                            'industry': info.get('industry', 'N/A'),
                            'market_cap': info.get('marketCap', 'N/A'),
                            'pe_ratio': info.get('trailingPE', 'N/A'),
                            'dividend_yield': info.get('dividendYield', 'N/A'),
                            '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                            '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                            'avg_volume': info.get('averageVolume', 'N/A'),
                            'description': info.get('longBusinessSummary', 'N/A')[:200] + '...' if info.get('longBusinessSummary') else 'N/A',
                            'website': info.get('website', 'N/A'),
                            'employees': info.get('fullTimeEmployees', 'N/A'),
                            'ytd_return': info.get('ytdReturn', 'N/A'),
                            'trailing_return': info.get('trailingReturn', 'N/A')
                        }
                        result_item.update(full_info)

                    # Add news if requested
                    if include_news:
                        try:
                            news = get_asset_news(symbol, limit=2)
                            result_item['recent_news'] = news
                        except:
                            result_item['recent_news'] = []

                    results.append(result_item)
        except Exception as e:
            print(f"Primary search error: {e}")
            results = universal_fallback_search(query, max_results, include_full_info)

        response_data = {
            'query': query,
            'count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        # Cache for 5 minutes
        search_cache[cache_key] = response_data
        cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': str(e), 'query': query}), 500


@app.route('/api/asset/<symbol>', methods=['GET'])
def get_asset_details(symbol):
    """
    Get comprehensive details for ANY asset type
    Fetches all available information from yfinance
    Parameters:
    - symbol: Any yfinance-supported symbol (stocks, crypto, commodities, indices, etfs, etc.)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get historical data
        hist = ticker.history(period='1y')
        
        if hist.empty:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        # Calculate technical metrics
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Get latest data
        latest = hist.iloc[-1]
        
        # Get 52-week stats
        year_high = hist['Close'].max()
        year_low = hist['Close'].min()
        
        # Comprehensive asset details
        asset_data = {
            'symbol': symbol,
            'type': info.get('quoteType', info.get('instrumentType', 'Unknown')),
            'name': info.get('longName', info.get('shortName', 'N/A')),
            'currency': info.get('currency', 'USD'),
            
            # PRICE DATA
            'price': {
                'current': round(float(latest['Close']), 2),
                'open': round(float(latest['Open']), 2),
                'high_today': round(float(latest['High']), 2),
                'low_today': round(float(latest['Low']), 2),
                '52_week_high': round(year_high, 2),
                '52_week_low': round(year_low, 2),
                'previous_close': info.get('previousClose', 'N/A'),
                'regular_market_price': info.get('regularMarketPrice', 'N/A'),
                'bid': info.get('bid', 'N/A'),
                'ask': info.get('ask', 'N/A'),
                'bid_size': info.get('bidSize', 'N/A'),
                'ask_size': info.get('askSize', 'N/A'),
            },
            
            # VOLUME & TRADING DATA
            'volume': {
                'current': int(latest['Volume']),
                'average_volume': info.get('averageVolume', 'N/A'),
                'average_volume_10d': info.get('averageVolume10days', 'N/A'),
            },
            
            # CHANGE & PERFORMANCE
            'performance': {
                'change': round(info.get('regularMarketChange', 0), 2),
                'change_percent': round(info.get('regularMarketChangePercent', 0), 2),
                'ytd_return': info.get('ytdReturn', 'N/A'),
                'trailing_return': info.get('trailingReturn', 'N/A'),
            },
            
            # COMPANY INFO
            'company': {
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'website': info.get('website', 'N/A'),
                'employees': info.get('fullTimeEmployees', 'N/A'),
                'country': info.get('country', 'N/A'),
                'city': info.get('city', 'N/A'),
                'state': info.get('state', 'N/A'),
                'zip_code': info.get('zip', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')[:500] if info.get('longBusinessSummary') else 'N/A',
            },
            
            # FINANCIAL METRICS
            'financials': {
                'market_cap': info.get('marketCap', 'N/A'),
                'enterprise_value': info.get('enterpriseValue', 'N/A'),
                'revenue': info.get('totalRevenue', 'N/A'),
                'net_income': info.get('netIncome', 'N/A'),
                'profit_margin': info.get('profitMargins', 'N/A'),
                'operating_margin': info.get('operatingMargins', 'N/A'),
                'return_on_assets': info.get('returnOnAssets', 'N/A'),
                'return_on_equity': info.get('returnOnEquity', 'N/A'),
            },
            
            # VALUATION METRICS
            'valuation': {
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'price_to_book': info.get('priceToBook', 'N/A'),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 'N/A'),
                'ev_to_revenue': info.get('enterpriseToRevenue', 'N/A'),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 'N/A'),
                'peg_ratio': info.get('pegRatio', 'N/A'),
            },
            
            # DIVIDEND INFO
            'dividend': {
                'dividend_rate': info.get('dividendRate', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'dividend_date': info.get('dividendDate', 'N/A'),
                'ex_dividend_date': info.get('exDividendDate', 'N/A'),
                'payout_ratio': info.get('payoutRatio', 'N/A'),
                'last_dividend': info.get('lastDividendValue', 'N/A'),
            },
            
            # DEBT & CASH
            'balance_sheet': {
                'total_debt': info.get('totalDebt', 'N/A'),
                'total_cash': info.get('totalCash', 'N/A'),
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'current_ratio': info.get('currentRatio', 'N/A'),
                'quick_ratio': info.get('quickRatio', 'N/A'),
                'book_value': info.get('bookValue', 'N/A'),
            },
            
            # RISK METRICS
            'risk': {
                'volatility_annual': round(volatility, 4),
                'beta': info.get('beta', 'N/A'),
                'trailing_annual_dividend_rate': info.get('trailingAnnualDividendRate', 'N/A'),
                'trailing_annual_dividend_yield': info.get('trailingAnnualDividendYield', 'N/A'),
            },
            
            # GROWTH & ESTIMATES
            'growth': {
                'revenue_growth': info.get('revenueGrowth', 'N/A'),
                'earnings_growth': info.get('earningsGrowth', 'N/A'),
                'earnings_estimate_current_year': info.get('epsTrailingTwelveMonths', 'N/A'),
                'earnings_estimate_next_year': info.get('epsCurrentYear', 'N/A'),
                'earnings_estimate_next_quarter': info.get('epsEstimate', 'N/A'),
            },
            
            # ANALYST RATINGS
            'analyst': {
                'target_price': info.get('targetMeanPrice', 'N/A'),
                'target_high': info.get('targetHighPrice', 'N/A'),
                'target_low': info.get('targetLowPrice', 'N/A'),
                'number_of_analysts': info.get('numberOfAnalysts', 'N/A'),
                'recommendation_key': info.get('recommendationKey', 'N/A'),
            },
            
            # OWNERSHIP
            'ownership': {
                'shares_outstanding': info.get('sharesOutstanding', 'N/A'),
                'float_shares': info.get('floatShares', 'N/A'),
                'shares_short': info.get('sharesShort', 'N/A'),
                'shares_short_prior_month': info.get('sharesShortPriorMonth', 'N/A'),
                'insider_holdings_percent': info.get('insidersPercent', 'N/A'),
                'institutional_holdings_percent': info.get('institutionsPercent', 'N/A'),
            },
            
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(asset_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'symbol': symbol}), 500


@app.route('/api/stock/multiple', methods=['POST'])
def get_multiple_stocks():
    """
    Get data for multiple stocks at once
    Request body:
    {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "timeframe": "1M"
    }
    """
    try:
        data = request.get_json()

        if not data or 'symbols' not in data:
            return jsonify({'error': 'symbols array is required'}), 400

        symbols = data['symbols']
        timeframe = data.get('timeframe', '1M').upper()

        if timeframe not in PERIOD_INTERVAL_MAP:
            return jsonify({
                'error': f'Invalid timeframe. Choose from: {", ".join(PERIOD_INTERVAL_MAP.keys())}'
            }), 400

        period = PERIOD_INTERVAL_MAP[timeframe]['period']
        interval = PERIOD_INTERVAL_MAP[timeframe]['interval']

        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                results.append({
                    'symbol': symbol,
                    'price': round(current_price, 2),
                    'change': round(info.get('regularMarketChange', 0), 2),
                    'changePercent': round(info.get('regularMarketChangePercent', 0), 2)
                })
            except:
                continue

        return jsonify({
            'timeframe': timeframe,
            'period': period,
            'interval': interval,
            'results': results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== CRYPTO ENDPOINTS ====================

@app.route('/api/crypto/quote/<symbol>', methods=['GET'])
def get_crypto_quote(symbol):
    """Get current crypto quote with live price"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        previous_close = info.get('previousClose', current_price)

        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close != 0 else 0

        return jsonify({
            'symbol': symbol,
            'price': round(current_price, 2),
            'change': round(change, 2),
            'changePercent': round(change_percent, 2),
            'timestamp': datetime.now().isoformat(),
            'open': info.get('open', current_price),
            'high': info.get('dayHigh', current_price),
            'low': info.get('dayLow', current_price),
            'volume': info.get('volume', 0),
            'marketCap': info.get('marketCap', 0)
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Failed to fetch crypto quote: {str(e)}',
            'symbol': symbol
        }), 500


@app.route('/api/crypto/history/<symbol>', methods=['GET'])
def get_crypto_history(symbol):
    """Get crypto historical data"""
    try:
        timeframe = request.args.get('timeframe', '1M').upper()
        period = request.args.get('period')
        interval = request.args.get('interval')

        if period and interval:
            config = {'period': period, 'interval': interval}
        else:
            config = PERIOD_INTERVAL_MAP.get(timeframe, PERIOD_INTERVAL_MAP['1M'])

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=config['period'], interval=config['interval'])

        if hist.empty:
            return jsonify({'error': 'No data available', 'symbol': symbol}), 404

        hist.reset_index(inplace=True)
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'period': config['period'],
            'interval': config['interval'],
            'data': hist.to_dict(orient='records')
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Failed to fetch crypto history: {str(e)}',
            'symbol': symbol
        }), 500


# ==================== NEWS ENDPOINTS ====================

@app.route('/api/asset/news/<symbol>', methods=['GET'])
def get_asset_news_endpoint(symbol):
    """
    Get latest news for an asset
    Query params:
    - limit: number of news items (default: 10)
    """
    try:
        limit = int(request.args.get('limit', 10))
        news = get_asset_news(symbol, limit)

        return jsonify({
            'symbol': symbol,
            'news_count': len(news),
            'news': news,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_asset_news(symbol, limit=10):
    """Fetch news from Finnhub API"""
    try:
        # Remove .NS or .BO suffix for Finnhub API
        symbol_clean = symbol.split('.')[0]

        news_data = finnhub_client.company_news(
            symbol_clean,
            _from=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            to=datetime.now().strftime('%Y-%m-%d')
        )

        news_list = []
        for item in news_data[:limit]:
            news_list.append({
                'headline': item.get('headline', 'N/A'),
                'summary': item.get('summary', 'N/A')[:200],
                'source': item.get('source', 'N/A'),
                'url': item.get('url', 'N/A'),
                'datetime': datetime.fromtimestamp(item.get('datetime', 0)).isoformat()
            })

        return news_list

    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []


# ==================== RISK ANALYSIS ENDPOINTS ====================

@app.route('/api/asset/risk/<symbol>', methods=['GET'])
def predict_asset_risk(symbol):
    """
    Predict asset risk based on historical volatility and ML
    Query params:
    - period: analysis period (default: 3M)
    """
    try:
        period = request.args.get('period', '3M').upper()

        if period not in PERIOD_INTERVAL_MAP:
            return jsonify({'error': f'Invalid period. Choose from: {", ".join(PERIOD_INTERVAL_MAP.keys())}'}), 400

        period_days = PERIOD_INTERVAL_MAP[period]['period']

        # Fetch historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period_days)

        if hist.empty or len(hist) < 5:
            return jsonify({'error': f'Insufficient data for {symbol}'}), 404

        # Calculate metrics
        daily_returns = hist['Close'].pct_change().dropna()

        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)

        # Value at Risk (VaR) - 95% confidence
        var_95 = daily_returns.quantile(0.05)

        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sharpe Ratio
        sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0

        # Risk score (0-100)
        risk_score = min(100, volatility * 100)

        # Risk level classification
        if risk_score < 15:
            risk_level = 'Very Low'
        elif risk_score < 30:
            risk_level = 'Low'
        elif risk_score < 50:
            risk_level = 'Medium'
        elif risk_score < 75:
            risk_level = 'High'
        else:
            risk_level = 'Very High'

        return jsonify({
            'symbol': symbol,
            'period': period,
            'risk_metrics': {
                'volatility_annual': round(volatility, 4),
                'value_at_risk_95': round(float(var_95), 4),
                'maximum_drawdown': round(float(max_drawdown), 4),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'risk_score': round(risk_score, 2),
                'risk_level': risk_level
            },
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== FORECAST ENDPOINTS ====================

@app.route('/api/asset/forecast/<symbol>', methods=['GET'])
def forecast_asset_value(symbol):
    """
    Forecast asset price for next 10 days based on 3 months historical data
    Uses machine learning (Random Forest)
    """
    try:
        # Fetch 3 months of historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')

        if hist.empty or len(hist) < 20:
            return jsonify({'error': f'Insufficient data for forecasting {symbol}'}), 404

        # Prepare data
        hist_reset = hist.reset_index()
        hist_reset['Days'] = np.arange(len(hist_reset))
        hist_reset['MA_5'] = hist_reset['Close'].rolling(window=5).mean()
        hist_reset['MA_10'] = hist_reset['Close'].rolling(window=10).mean()
        hist_reset['Volatility'] = hist_reset['Close'].pct_change().rolling(window=10).std()

        # Remove NaN values
        hist_clean = hist_reset.dropna()

        if len(hist_clean) < 10:
            return jsonify({'error': 'Insufficient clean data for forecasting'}), 400

        # Features and target
        X = hist_clean[['Days', 'Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'Volatility']].values
        y = hist_clean['Close'].values

        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)

        # Generate forecast for next 10 days
        last_day = hist_clean['Days'].iloc[-1]
        forecast_days = []
        forecast_prices = []
        forecast_confidence = []

        for i in range(1, 11):
            next_day = last_day + i

            # Create feature vector for next day
            last_row = hist_clean.iloc[-1]
            next_features = np.array([[
                next_day,
                last_row['Close'],
                last_row['High'],
                last_row['Low'],
                last_row['Volume'],
                last_row['MA_5'],
                last_row['MA_10'],
                last_row['Volatility']
            ]])

            next_features_scaled = scaler.transform(next_features)
            predicted_price = model.predict(next_features_scaled)[0]

            # Calculate confidence based on model predictions
            predictions = np.array([tree.predict(next_features_scaled)[0] for tree in model.estimators_])
            confidence = 1 - (predictions.std() / predictions.mean()) if predictions.mean() > 0 else 0.7
            confidence = max(0.5, min(0.95, confidence))

            forecast_days.append((datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'))
            forecast_prices.append(round(float(predicted_price), 2))
            forecast_confidence.append(round(float(confidence), 3))

        current_price = float(hist_clean['Close'].iloc[-1])
        price_change = forecast_prices[-1] - current_price
        price_change_pct = (price_change / current_price) * 100

        return jsonify({
            'symbol': symbol,
            'current_price': current_price,
            'forecast_period': '10 days',
            'forecast': {
                'dates': forecast_days,
                'prices': forecast_prices,
                'confidence': forecast_confidence
            },
            'summary': {
                'predicted_price_day_10': forecast_prices[-1],
                'expected_change': round(price_change, 2),
                'expected_change_percent': round(price_change_pct, 2),
                'trend': 'Uptrend' if price_change > 0 else 'Downtrend'
            },
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== INSIGHTS ENDPOINTS ====================

@app.route('/api/asset/insights/<symbol>', methods=['GET'])
def get_asset_insights(symbol):
    """
    Generate AI-powered insights about an asset
    Analyzes price action, technical metrics, news sentiment
    Returns human-readable insights
    """
    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')

        if hist.empty or len(hist) < 10:
            return jsonify({'error': f'Insufficient data for insights {symbol}'}), 404

        # Get current and recent data
        current_price = float(hist['Close'].iloc[-1])
        price_30_days_ago = float(hist['Close'].iloc[-30]) if len(hist) >= 30 else float(hist['Close'].iloc[0])
        price_change_30d = ((current_price - price_30_days_ago) / price_30_days_ago) * 100

        # Calculate technical metrics
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)

        # Moving averages
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else hist['Close'].mean()

        # Volume analysis
        avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Get news sentiment
        try:
            news = get_asset_news(symbol, limit=5)
            positive_keywords = ['rally', 'surge', 'gain', 'profit', 'growth', 'up', 'strong', 'beat']
            negative_keywords = ['drop', 'fall', 'loss', 'decline', 'down', 'weak', 'miss', 'sell']

            positive_count = sum(1 for article in news if any(kw in article['headline'].lower() for kw in positive_keywords))
            negative_count = sum(1 for article in news if any(kw in article['headline'].lower() for kw in negative_keywords))
        except:
            positive_count = 0
            negative_count = 0

        # Generate insights
        insights = []

        # Price insights
        if price_change_30d > 10:
            insights.append(f"📈 Strong uptrend: {symbol} has gained {abs(price_change_30d):.1f}% in the last 30 days")
        elif price_change_30d > 0:
            insights.append(f"↗️ Mild uptrend: {symbol} is up {price_change_30d:.1f}% over the last month")
        elif price_change_30d < -10:
            insights.append(f"📉 Strong downtrend: {symbol} has declined {abs(price_change_30d):.1f}% in the last 30 days")
        else:
            insights.append(f"↘️ Slight downtrend: {symbol} is down {abs(price_change_30d):.1f}% over the last month")

        # Technical insights
        if current_price > ma_20 > ma_50:
            insights.append(f"📊 Bullish structure: Price is above both 20-day and 50-day moving averages")
        elif current_price < ma_20 < ma_50:
            insights.append(f"📊 Bearish structure: Price is below both 20-day and 50-day moving averages")

        # Volatility insights
        if volatility > 0.3:
            insights.append(f"⚠️ High volatility: {symbol} shows high price swings ({volatility*100:.1f}% annualized). Consider position sizing carefully")
        elif volatility < 0.1:
            insights.append(f"🟢 Low volatility: {symbol} is relatively stable with {volatility*100:.1f}% annualized volatility")

        # Volume insights
        if volume_ratio > 1.5:
            insights.append(f"📢 High activity: Current trading volume is {volume_ratio:.1f}x the average. Increased interest detected")
        elif volume_ratio < 0.5:
            insights.append(f"🔇 Low activity: Trading volume is below average. Consider liquidity before large positions")

        # News sentiment
        if positive_count > negative_count:
            sentiment = 'Positive'
            insights.append(f"📰 Market sentiment: Recent news is predominantly {sentiment.lower()} ({positive_count} positive vs {negative_count} negative articles)")
        elif negative_count > positive_count:
            sentiment = 'Negative'
            insights.append(f"📰 Market sentiment: Recent news is predominantly {sentiment.lower()} ({negative_count} negative vs {positive_count} positive articles)")
        else:
            sentiment = 'Neutral'
            insights.append(f"📰 Market sentiment: Recent news is {sentiment.lower()}")

        # Risk assessment
        if volatility > 0.25:
            risk_assessment = 'High risk - Suitable for experienced investors'
        elif volatility > 0.15:
            risk_assessment = 'Medium risk - Balanced investors should consider careful position sizing'
        else:
            risk_assessment = 'Lower risk - Suitable for conservative investors'

        insights.append(f"⚡ Risk level: {risk_assessment}")

        return jsonify({
            'symbol': symbol,
            'current_price': current_price,
            'price_change_30d': round(price_change_30d, 2),
            'technical_metrics': {
                'volatility_annual': round(volatility, 4),
                'ma_20': round(float(ma_20), 2),
                'ma_50': round(float(ma_50), 2),
                'volume_ratio': round(volume_ratio, 2)
            },
            'sentiment': {
                'positive_articles': positive_count,
                'negative_articles': negative_count,
                'overall': sentiment
            },
            'insights': insights,
            'recommendation': 'Do your own research and consult a financial advisor before investing',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== PORTFOLIO ENDPOINTS ====================

@app.route('/api/timeframes', methods=['GET'])
def get_available_timeframes():
    """Get available timeframes and their configurations"""
    return jsonify({
        'timeframes': PERIOD_INTERVAL_MAP
    }), 200


@app.route('/api/portfolio/optimize', methods=['GET'])
def optimize_portfolio_quantum():
    """
    Quantum Portfolio Optimization using Qiskit
    Fetches user's assets from localhost:8080/api/assets
    and computes optimal portfolio allocation using quantum QAOA algorithm
    Query params:
    - budget: investment budget (default: 10000)
    - risk_factor: risk tolerance 0-1 (default: 0.3)
    - target_assets: number of assets to select (default: half of available)
    - max_assets: maximum assets to invest in (optional filter)
    """
    try:
        # Get assets from the user's portfolio API
        portfolio_response = requests.get('http://localhost:8080/api/assets', timeout=5)

        if portfolio_response.status_code != 200:
            return jsonify({
                'error': f'Failed to fetch assets from API: {portfolio_response.status_code}'
            }), 500

        assets_data = portfolio_response.json()

        # Extract symbols from the response
        if isinstance(assets_data, list):
            symbols = [asset.get('symbol') for asset in assets_data if asset.get('symbol')]
        elif isinstance(assets_data, dict) and 'assets' in assets_data:
            symbols = [asset.get('symbol') for asset in assets_data['assets'] if asset.get('symbol')]
        else:
            return jsonify({
                'error': 'Unexpected response format from assets API'
            }), 500

        if not symbols:
            return jsonify({
                'error': 'No symbols found in portfolio'
            }), 404

        # Limit to first 5 assets for quantum optimization (NISQ constraints)
        if len(symbols) > 5:
            symbols = symbols[:5]
            truncated = True
        else:
            truncated = False

        # Get parameters
        budget = float(request.args.get('budget', 10000))
        risk_factor = float(request.args.get('risk_factor', 0.9))

        # Max assets to actually invest in (filter after quantum optimization)
        max_assets = request.args.get('max_assets')
        max_assets = int(max_assets) if max_assets else None

        # Fetch historical data for returns calculation
        returns_data = []
        valid_symbols = []

        print(f"\nFetching historical data for {len(symbols)} symbols...")
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Use 2 years of weekly data for more reliable statistics
                hist = ticker.history(period='2y', interval='1wk')

                if not hist.empty and len(hist) >= 52:
                    # Calculate weekly returns
                    weekly_returns = hist['Close'].pct_change().dropna()
                    returns_data.append(weekly_returns.values)
                    valid_symbols.append(symbol)
                    print(f"  {symbol}: {len(weekly_returns)} weeks of data")
                else:
                    print(f"  {symbol}: Insufficient data ({len(hist)} weeks)")
            except Exception as e:
                print(f"  Error fetching data for {symbol}: {e}")
                continue

        if len(valid_symbols) < 2:
            return jsonify({
                'error': 'Need at least 2 valid symbols with historical data'
            }), 400

        # Calculate mean returns and covariance matrix
        returns_array = np.array(returns_data)
        mean_returns = np.mean(returns_array, axis=1)
        cov_matrix = np.cov(returns_array)

        # Number of assets
        num_assets = len(valid_symbols)

        # Get target number of assets to select (default: 2-3 for better optimization)
        target_assets_str = request.args.get('target_assets', None)
        if target_assets_str:
            target_assets = int(target_assets_str)
        else:
            # Default: aim for ~half the available assets (better balance)
            target_assets = max(2, num_assets // 2)

        print(f"\nOptimizing portfolio with {num_assets} assets...")
        print(f"Target assets to select: {target_assets}")
        print(f"Risk factor: {risk_factor}")

        # Setup portfolio optimization problem
        q = risk_factor  # risk tolerance
        portfolio = PortfolioOptimization(
            expected_returns=mean_returns,
            covariances=cov_matrix,
            risk_factor=q,
            budget=target_assets  # Target number of assets to select
        )

        qp = portfolio.to_quadratic_program()

        # Solve using QAOA (Quantum Approximate Optimization Algorithm)
        try:
            import traceback
            # Use StatevectorSampler for compatibility with Qiskit 2.x
            sampler = StatevectorSampler()
            # Increased iterations and reps for better optimization
            optimizer = COBYLA(maxiter=500)
            qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=3)
            qaoa_optimizer = MinimumEigenOptimizer(qaoa)

            print("Running QAOA optimization...")
            result = qaoa_optimizer.solve(qp)
            print(f"QAOA completed. Objective value: {result.fval:.6f}")
        except Exception as qaoa_error:
            # Return the error to the user
            error_details = traceback.format_exc()
            print(f"QAOA failed with error:\n{error_details}")
            return jsonify({
                'error': 'Quantum optimization failed',
                'details': str(qaoa_error),
                'traceback': error_details
            }), 500

        # Parse results - Extract quantum probabilities from eigenstate
        selection = result.x

        # Extract eigenstate and probabilities from quantum result
        quantum_solutions = []
        if hasattr(result, 'min_eigen_solver_result'):
            eigenstate = result.min_eigen_solver_result.eigenstate

            # Get probabilities from eigenstate
            if isinstance(eigenstate, QuasiDistribution):
                probabilities = eigenstate.binary_probabilities()
            elif isinstance(eigenstate, dict):
                probabilities = {k: np.abs(v) ** 2 for k, v in eigenstate.items()}
            else:
                # Try to convert to dict if it has to_dict method
                probabilities = {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}

            # Sort by probability and get top 10 solutions
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10]

            for bitstring, prob in sorted_probs:
                # Convert bitstring to array
                x = np.array([int(i) for i in list(reversed(bitstring))])
                value = qp.objective.evaluate(x)

                quantum_solutions.append({
                    'selection': x.tolist(),
                    'bitstring': bitstring,
                    'assets': [valid_symbols[i] for i in range(len(x)) if x[i] == 1],
                    'objective_value': float(value),
                    'probability': float(prob),
                    'num_assets': int(sum(x))
                })

            # Add the actual selected solution if not in top 10
            selected_bitstring = ''.join(str(int(b)) for b in reversed(selection))
            if not any(sol['bitstring'] == selected_bitstring for sol in quantum_solutions):
                selected_prob = probabilities.get(selected_bitstring, 0.0)
                quantum_solutions.insert(0, {
                    'selection': selection.tolist(),
                    'bitstring': selected_bitstring,
                    'assets': [valid_symbols[i] for i in range(len(selection)) if selection[i] > 0.5],
                    'objective_value': float(result.fval),
                    'probability': float(selected_prob),
                    'num_assets': int(sum(selection > 0.5)),
                    'is_selected': True
                })
            print("\n----------------- Quantum Solutions (Top 10) ---------------------")
            print("selection\tvalue\t\tprobability\t# assets")
            print("------------------------------------------------------------------")
            for sol in quantum_solutions[:10]:
                marker = " ← SELECTED" if sol.get('is_selected') else ""
                print(f"{sol['selection']}\t{sol['objective_value']:.6f}\t{sol['probability']:.6f}\t{sol['num_assets']}{marker}")

        selected_indices = [i for i, val in enumerate(selection) if val > 0.5]

        if not selected_indices:
            return jsonify({
                'error': 'No assets selected by optimizer',
                'suggestion': 'Try adjusting risk_factor parameter',
                'quantum_solutions': quantum_solutions if quantum_solutions else None
            }), 400

        print(f"\nQuantum selection: {[valid_symbols[i] for i in selected_indices]}")

        # Apply max_assets filter if specified
        if max_assets and len(selected_indices) > max_assets:
            # Keep only the top max_assets by expected return
            sorted_by_return = sorted(selected_indices, key=lambda i: mean_returns[i], reverse=True)
            selected_indices = sorted_by_return[:max_assets]
            print(f"Filtered to top {max_assets} assets by return: {[valid_symbols[i] for i in selected_indices]}")

        # Pure quantum approach: Equal weight allocation for selected assets
        n_selected = len(selected_indices)
        equal_weight = 1.0 / n_selected

        print(f"Allocating equally: {equal_weight:.4f} ({equal_weight*100:.2f}%) to each of {n_selected} assets")

        # Build allocation dictionary with equal weights
        optimal_allocation = {}
        for i, symbol in enumerate(valid_symbols):
            if i in selected_indices:
                optimal_allocation[symbol] = {
                    'selected': True,
                    'weight': round(float(equal_weight), 4),
                    'allocation': round(float(budget * equal_weight), 2),
                    'percentage': round(float(equal_weight * 100), 2)
                }
            else:
                optimal_allocation[symbol] = {
                    'selected': False,
                    'weight': 0,
                    'allocation': 0,
                    'percentage': 0
                }

        # Calculate portfolio metrics with equal weights
        selected_returns = mean_returns[selected_indices]
        selected_cov = cov_matrix[np.ix_(selected_indices, selected_indices)]
        equal_weights = np.array([equal_weight] * n_selected)

        portfolio_return = np.dot(equal_weights, selected_returns)
        portfolio_variance = np.dot(equal_weights, np.dot(selected_cov, equal_weights))
        portfolio_risk = np.sqrt(portfolio_variance)

        return jsonify({
            'status': 'success',
            'solver': 'quantum (QAOA)',
            'allocation_method': 'quantum_binary_equal_weight',
            'truncated': truncated,
            'original_symbols': symbols,
            'valid_symbols': valid_symbols,
            'budget': budget,
            'target_assets': target_assets,
            'assets_selected': n_selected,
            'max_assets': max_assets,
            'risk_factor': risk_factor,
            'optimal_allocation': optimal_allocation,
            'portfolio_metrics': {
                'expected_return': float(portfolio_return),
                'risk': float(portfolio_risk),
                'sharpe_ratio': float(portfolio_return / portfolio_risk) if portfolio_risk > 0 else 0
            },
            'objective_value': float(result.fval),
            'quantum_solutions': quantum_solutions if quantum_solutions else None
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': f'Failed to connect to portfolio API: {str(e)}',
            'hint': 'Make sure the API at localhost:8080/api/assets is running'
        }), 503
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


# ==================== HELPER FUNCTIONS ====================

def universal_fallback_search(query, max_results, include_full_info=False):
    """Universal fallback search for various asset types"""
    results = []
    
    # Comprehensive list of common symbols across all asset types
    common_symbols = {
        # US STOCKS
        'apple': {'symbol': 'AAPL', 'name': 'Apple Inc.', 'type': 'Stock'},
        'microsoft': {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'type': 'Stock'},
        'google': {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'type': 'Stock'},
        'amazon': {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'type': 'Stock'},
        'tesla': {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'type': 'Stock'},
        'nvidia': {'symbol': 'NVDA', 'name': 'NVIDIA Corp', 'type': 'Stock'},
        'meta': {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'type': 'Stock'},
        
        # CRYPTOCURRENCIES
        'bitcoin': {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'type': 'Crypto'},
        'ethereum': {'symbol': 'ETH-USD', 'name': 'Ethereum', 'type': 'Crypto'},
        'cardano': {'symbol': 'ADA-USD', 'name': 'Cardano', 'type': 'Crypto'},
        'ripple': {'symbol': 'XRP-USD', 'name': 'Ripple', 'type': 'Crypto'},
        'solana': {'symbol': 'SOL-USD', 'name': 'Solana', 'type': 'Crypto'},
        'dogecoin': {'symbol': 'DOGE-USD', 'name': 'Dogecoin', 'type': 'Crypto'},
        
        # COMMODITIES
        'gold': {'symbol': 'GC=F', 'name': 'Gold', 'type': 'Commodity'},
        'silver': {'symbol': 'SI=F', 'name': 'Silver', 'type': 'Commodity'},
        'crude oil': {'symbol': 'CL=F', 'name': 'Crude Oil', 'type': 'Commodity'},
        'natural gas': {'symbol': 'NG=F', 'name': 'Natural Gas', 'type': 'Commodity'},
        
        # INDICES
        'sp500': {'symbol': '^GSPC', 'name': 'S&P 500', 'type': 'Index'},
        'dow': {'symbol': '^DJI', 'name': 'Dow Jones', 'type': 'Index'},
        'nasdaq': {'symbol': '^IXIC', 'name': 'Nasdaq', 'type': 'Index'},
        
        # ETFS
        'spy': {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF', 'type': 'ETF'},
        'qqq': {'symbol': 'QQQ', 'name': 'Invesco QQQ ETF', 'type': 'ETF'},
        'iwm': {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'type': 'ETF'},
        
        # INDIAN STOCKS
        'reliance': {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries', 'type': 'Stock'},
        'tcs': {'symbol': 'TCS.NS', 'name': 'Tata Consultancy Services', 'type': 'Stock'},
        'infosys': {'symbol': 'INFY.NS', 'name': 'Infosys Limited', 'type': 'Stock'},
        'hdfc': {'symbol': 'HDFC.NS', 'name': 'HDFC Bank', 'type': 'Stock'},
        'icici': {'symbol': 'ICICIBANK.NS', 'name': 'ICICI Bank', 'type': 'Stock'},
        
        # BONDS & TREASURIES
        'treasury': {'symbol': '^TNX', 'name': '10-Year US Treasury Yield', 'type': 'Bond'},
        '2year': {'symbol': '^IRX', 'name': '13-Week Treasury Bill', 'type': 'Bond'},
    }

    query_lower = query.lower()

    for key, value in common_symbols.items():
        if key.startswith(query_lower) and len(results) < max_results:
            try:
                ticker = yf.Ticker(value['symbol'])
                info = ticker.info
                hist = ticker.history(period='5d')

                if not hist.empty:
                    latest = hist.iloc[-1]
                    current_price = float(latest['Close'])
                    open_price = float(latest['Open'])
                    high_price = float(latest['High'])
                    low_price = float(latest['Low'])
                    volume = int(latest['Volume'])
                    change_pct = ((current_price - open_price) / open_price * 100) if open_price != 0 else 0
                else:
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                    open_price = info.get('open', 0)
                    high_price = info.get('dayHigh', 0)
                    low_price = info.get('dayLow', 0)
                    volume = info.get('volume', 0)
                    change_pct = 0

                result = {
                    'symbol': value['symbol'],
                    'name': value['name'],
                    'type': value['type'],
                    'current_price': round(current_price, 2) if current_price else 'N/A',
                    'open': round(open_price, 2) if open_price else 'N/A',
                    'high': round(high_price, 2) if high_price else 'N/A',
                    'low': round(low_price, 2) if low_price else 'N/A',
                    'volume': volume,
                    'change_percent': round(change_pct, 2),
                    'currency': info.get('currency', 'USD')
                }

                if include_full_info:
                    result.update({
                        'market_cap': info.get('marketCap', 'N/A'),
                        'pe_ratio': info.get('trailingPE', 'N/A'),
                        'dividend_yield': info.get('dividendYield', 'N/A'),
                        '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                        '52_week_low': info.get('fiftyTwoWeekLow', 'N/A')
                    })

                results.append(result)
            except Exception as e:
                print(f"Error fetching fallback data for {value['symbol']}: {e}")
                continue

    return results


# ==================== WEBSOCKET ENDPOINTS ====================

@sock.route('/ws/stream')
def websocket_stream(ws):
    """
    WebSocket endpoint for real-time stock data streaming
    Client should send JSON message with symbols to subscribe:
    {"action": "subscribe", "symbols": ["AAPL", "BTC-USD"]}
    {"action": "unsubscribe"}
    """
    connection_id = id(ws)
    active_ws_connections[connection_id] = {'ws': ws, 'task': None, 'symbols': []}

    try:
        while True:
            message = ws.receive()
            if message:
                try:
                    data = json.loads(message)
                    action = data.get('action')

                    if action == 'subscribe':
                        symbols = data.get('symbols', [])
                        if symbols:
                            # Cancel existing task if any
                            if active_ws_connections[connection_id]['task']:
                                active_ws_connections[connection_id]['task'].cancel()

                            # Start streaming in background thread
                            active_ws_connections[connection_id]['symbols'] = symbols
                            thread = threading.Thread(
                                target=stream_stock_data,
                                args=(ws, symbols)
                            )
                            thread.daemon = True
                            thread.start()

                            ws.send(json.dumps({
                                'type': 'subscribed',
                                'symbols': symbols
                            }))
                        else:
                            ws.send(json.dumps({
                                'type': 'error',
                                'message': 'No symbols provided'
                            }))

                    elif action == 'unsubscribe':
                        if active_ws_connections[connection_id]['task']:
                            active_ws_connections[connection_id]['task'].cancel()
                        active_ws_connections[connection_id]['symbols'] = []
                        ws.send(json.dumps({
                            'type': 'unsubscribed'
                        }))

                except json.JSONDecodeError:
                    ws.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON'
                    }))

    except Exception as e:
        print(f"WebSocket error: {e}")

    finally:
        # Cleanup
        if connection_id in active_ws_connections:
            if active_ws_connections[connection_id]['task']:
                active_ws_connections[connection_id]['task'].cancel()
            del active_ws_connections[connection_id]


def stream_stock_data(ws, symbols):
    """
    Stream stock data by polling yfinance at regular intervals
    """
    try:
        ws.send(json.dumps({
            'type': 'stream_started',
            'symbols': symbols,
            'interval': '5 seconds'
        }))

        while True:
            try:
                # Fetch current data for all symbols
                stock_data = {}

                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period='1d', interval='1m')

                        if not hist.empty:
                            latest = hist.iloc[-1]
                            stock_data[symbol] = {
                                'symbol': symbol,
                                'price': float(latest['Close']),
                                'open': float(latest['Open']),
                                'high': float(latest['High']),
                                'low': float(latest['Low']),
                                'volume': int(latest['Volume']),
                                'timestamp': hist.index[-1].isoformat()
                            }
                    except Exception as e:
                        stock_data[symbol] = {
                            'symbol': symbol,
                            'error': str(e)
                        }

                # Send data
                ws.send(json.dumps({
                    'type': 'data',
                    'data': stock_data,
                    'timestamp': datetime.now().isoformat()
                }))

                time.sleep(5)

            except Exception as e:
                print(f"Error in stream loop: {e}")
                break

    except Exception as e:
        print(f"Stream error: {e}")
        try:
            ws.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
        except:
            pass


# ==================== APPLICATION STARTUP ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Starting BigBull Flask Server (Main API)")
    print("=" * 80)
    print(f"Server: http://localhost:5000")
    print(f"Health Check: http://localhost:5000/health")
    print("\nAvailable Endpoints:")
    print("\n[STOCK ENDPOINTS]")
    print("  GET  /api/stock/quote/<symbol>")
    print("  GET  /api/stock/history/<symbol>")
    print("  GET  /api/stock/info/<symbol>")
    print("  GET  /api/stock/search?query=<query>")
    print("  POST /api/stock/multiple")
    print("\n[CRYPTO ENDPOINTS]")
    print("  GET  /api/crypto/quote/<symbol>")
    print("  GET  /api/crypto/history/<symbol>")
    print("\n[NEWS & ANALYSIS ENDPOINTS]")
    print("  GET  /api/asset/news/<symbol>")
    print("  GET  /api/asset/risk/<symbol>")
    print("  GET  /api/asset/forecast/<symbol>")
    print("  GET  /api/asset/insights/<symbol>")
    print("\n[PORTFOLIO ENDPOINTS]")
    print("  GET  /api/timeframes")
    print("  GET  /api/portfolio/optimize")
    print("\n[WEBSOCKET ENDPOINTS]")
    print("  WS   /ws/stream")
    print("\n[HEALTH]")
    print("  GET  /health")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
