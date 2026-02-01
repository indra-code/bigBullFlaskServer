from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
import yfinance as yf
import json
from datetime import datetime
import threading
import logging

# Import custom modules
from config import config
from utils.data_fetcher import DataFetcher
from utils.news_service import NewsService
from utils.risk_analyzer import RiskAnalyzer
from utils.ml_predictor import MLPredictor, SimpleMovingAveragePredictor
from utils.insights_generator import InsightsGenerator
from utils.portfolio_service import PortfolioService
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Initialize services
news_service = NewsService(config.FINNHUB_API_KEY)
data_fetcher = DataFetcher()

# WebSocket connections storage
active_ws_connections = {}

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


# ============================================================================
# SEARCH ENDPOINTS (Category-wise autocomplete)
# ============================================================================

@app.route('/api/search/<category>', methods=['GET'])
def search_by_category(category):
    """
    Search assets by category with autocomplete
    
    Categories: stocks, crypto, mutualfunds, all
    Query params:
    - query: search query (required)
    - max_results: maximum number of results (default: 10)
    
    Example: /api/search/crypto?query=bit&max_results=5
    """
    try:
        query = request.args.get('query', '')
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        max_results = int(request.args.get('max_results', 10))
        
        # Validate category
        valid_categories = ['stocks', 'crypto', 'mutualfunds', 'all']
        if category not in valid_categories:
            return jsonify({
                'error': f'Invalid category. Choose from: {", ".join(valid_categories)}'
            }), 400
        
        # Search assets
        results = data_fetcher.search_assets(query, category, max_results)
        
        return jsonify({
            'category': category,
            'query': query,
            'count': len(results),
            'results': results
        }), 200
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ASSET DATA ENDPOINTS
# ============================================================================

@app.route('/api/asset/<category>/<symbol>/quote', methods=['GET'])
def get_asset_quote(category, symbol):
    """
    Get current quote for an asset
    
    Example: /api/asset/stocks/RELIANCE.NS/quote
    """
    try:
        price_data = data_fetcher.get_current_price(symbol)
        
        if not price_data:
            return jsonify({
                'error': f'No data found for {symbol}'
            }), 404
        
        price_data['category'] = category
        
        return jsonify(price_data), 200
    
    except Exception as e:
        logger.error(f"Error fetching quote: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/asset/<category>/<symbol>/history', methods=['GET'])
def get_asset_history(category, symbol):
    """
    Get historical data for an asset
    
    Query params:
    - timeframe: 1D, 5D, 1M, 3M, 6M, 1Y, 5Y, MAX (default: 3M)
    - period: custom period (overrides timeframe)
    - interval: custom interval (overrides timeframe)
    
    Example: /api/asset/stocks/RELIANCE.NS/history?timeframe=3M
    """
    try:
        timeframe = request.args.get('timeframe', '3M').upper()
        custom_period = request.args.get('period')
        custom_interval = request.args.get('interval')
        
        if custom_period and custom_interval:
            period = custom_period
            interval = custom_interval
        elif timeframe in config.PERIOD_INTERVAL_MAP:
            period = config.PERIOD_INTERVAL_MAP[timeframe]['period']
            interval = config.PERIOD_INTERVAL_MAP[timeframe]['interval']
        else:
            return jsonify({
                'error': f'Invalid timeframe. Choose from: {", ".join(config.PERIOD_INTERVAL_MAP.keys())}'
            }), 400
        
        # Fetch historical data
        history = data_fetcher.get_historical_data(symbol, period, interval)
        
        if history.empty:
            return jsonify({
                'error': f'No data found for {symbol}'
            }), 404
        
        # Convert to JSON-serializable format
        history_reset = history.reset_index()
        data_list = history_reset.to_dict(orient='records')
        
        # Convert timestamps
        for record in data_list:
            if 'Date' in record or 'Datetime' in record:
                date_key = 'Date' if 'Date' in record else 'Datetime'
                record[date_key] = record[date_key].isoformat() if hasattr(record[date_key], 'isoformat') else str(record[date_key])
        
        return jsonify({
            'symbol': symbol,
            'category': category,
            'timeframe': timeframe,
            'period': period,
            'interval': interval,
            'data': data_list
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/asset/<category>/<symbol>/info', methods=['GET'])
def get_asset_info(category, symbol):
    """
    Get detailed information about an asset
    
    Example: /api/asset/stocks/RELIANCE.NS/info
    """
    try:
        info = data_fetcher.get_asset_info(symbol)
        
        return jsonify({
            'symbol': symbol,
            'category': category,
            'info': info
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching info: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# NEWS ENDPOINTS
# ============================================================================

@app.route('/api/news/<symbol>', methods=['GET'])
def get_asset_news(symbol):
    """
    Get news for a specific asset
    
    Query params:
    - days_back: number of days to look back (default: 7)
    
    Example: /api/news/RELIANCE.NS?days_back=7
    """
    try:
        days_back = int(request.args.get('days_back', 7))
        
        news = news_service.get_company_news(symbol, days_back)
        
        return jsonify({
            'symbol': symbol,
            'days_back': days_back,
            'count': len(news),
            'news': news
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/news/market/<category>', methods=['GET'])
def get_market_news(category):
    """
    Get general market news
    
    Categories: general, forex, crypto, merger
    Query params:
    - count: number of articles (default: 10)
    
    Example: /api/news/market/crypto?count=5
    """
    try:
        count = int(request.args.get('count', 10))
        
        news = news_service.get_market_news(category, count)
        
        return jsonify({
            'category': category,
            'count': len(news),
            'news': news
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching market news: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sentiment/<symbol>', methods=['GET'])
def get_news_sentiment(symbol):
    """
    Get aggregated news sentiment for an asset
    
    Query params:
    - days_back: number of days to analyze (default: 7)
    
    Example: /api/sentiment/RELIANCE.NS?days_back=7
    """
    try:
        days_back = int(request.args.get('days_back', 7))
        
        sentiment = news_service.get_news_sentiment_summary(symbol, days_back)
        
        return jsonify(sentiment), 200
    
    except Exception as e:
        logger.error(f"Error fetching sentiment: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# RISK ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/risk/<symbol>', methods=['GET'])
def analyze_risk(symbol):
    """
    Analyze risk for an asset
    
    Query params:
    - period: lookback period for analysis (default: 3mo)
    - include_news: include news sentiment in analysis (default: true)
    
    Example: /api/risk/RELIANCE.NS?period=3mo&include_news=true
    """
    try:
        period = request.args.get('period', '3mo')
        include_news = request.args.get('include_news', 'true').lower() == 'true'
        
        # Fetch historical data
        historical_data = data_fetcher.get_historical_data(symbol, period, '1d')
        
        if historical_data.empty:
            return jsonify({
                'error': f'No data found for {symbol}'
            }), 404
        
        # Get current price
        current_price_data = data_fetcher.get_current_price(symbol)
        current_price = current_price_data['price'] if current_price_data else historical_data['Close'].iloc[-1]
        
        # Get news sentiment if requested
        news_sentiment = None
        if include_news:
            news_sentiment = news_service.get_news_sentiment_summary(symbol, 7)
        
        # Calculate risk score
        risk_analysis = RiskAnalyzer.calculate_risk_score(
            historical_data,
            current_price,
            news_sentiment
        )
        
        return jsonify({
            'symbol': symbol,
            'current_price': float(current_price),
            'analysis_period': period,
            'risk_analysis': risk_analysis
        }), 200
    
    except Exception as e:
        logger.error(f"Error analyzing risk: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/predict/<symbol>', methods=['GET'])
def predict_prices(symbol):
    """
    Predict future prices using ML
    
    Query params:
    - days: number of days to predict (default: 30)
    - method: ml or simple (default: ml)
    
    Example: /api/predict/RELIANCE.NS?days=30&method=ml
    """
    try:
        days = int(request.args.get('days', 30))
        method = request.args.get('method', 'ml').lower()
        
        # Validate days
        if days < 1 or days > 90:
            return jsonify({
                'error': 'Days must be between 1 and 90'
            }), 400
        
        # Fetch historical data (use more data for better predictions)
        historical_data = data_fetcher.get_historical_data(symbol, '1y', '1d')
        
        if historical_data.empty or len(historical_data) < 60:
            return jsonify({
                'error': f'Insufficient historical data for {symbol}'
            }), 404
        
        # Make predictions
        if method == 'ml':
            try:
                predictor = MLPredictor()
                prediction = predictor.predict_future(historical_data, days)
            except Exception as ml_error:
                logger.warning(f"ML prediction failed, falling back to simple method: {str(ml_error)}")
                prediction = SimpleMovingAveragePredictor.predict(historical_data, days)
        else:
            prediction = SimpleMovingAveragePredictor.predict(historical_data, days)
        
        return jsonify({
            'symbol': symbol,
            'prediction_method': method,
            'prediction': prediction
        }), 200
    
    except Exception as e:
        logger.error(f"Error predicting prices: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# INSIGHTS ENDPOINTS
# ============================================================================

@app.route('/api/insights/<symbol>', methods=['GET'])
def get_investment_insights(symbol):
    """
    Get comprehensive AI-powered investment insights
    
    Query params:
    - include_prediction: include price predictions (default: true)
    - include_news: include news sentiment (default: true)
    - prediction_days: days to predict (default: 30)
    
    Example: /api/insights/RELIANCE.NS?include_prediction=true&include_news=true
    """
    try:
        include_prediction = request.args.get('include_prediction', 'true').lower() == 'true'
        include_news = request.args.get('include_news', 'true').lower() == 'true'
        prediction_days = int(request.args.get('prediction_days', 30))
        
        # Fetch historical data
        historical_data = data_fetcher.get_historical_data(symbol, '1y', '1d')
        
        if historical_data.empty:
            return jsonify({
                'error': f'No data found for {symbol}'
            }), 404
        
        # Get current price
        current_price_data = data_fetcher.get_current_price(symbol)
        current_price = current_price_data['price'] if current_price_data else historical_data['Close'].iloc[-1]
        
        # Get risk analysis
        news_sentiment = None
        if include_news:
            news_sentiment = news_service.get_news_sentiment_summary(symbol, 7)
        
        risk_analysis = RiskAnalyzer.calculate_risk_score(
            historical_data,
            current_price,
            news_sentiment
        )
        
        # Get price prediction
        prediction = None
        if include_prediction:
            try:
                predictor = MLPredictor()
                prediction = predictor.predict_future(historical_data, prediction_days)
            except Exception as e:
                logger.warning(f"ML prediction failed: {str(e)}")
                prediction = SimpleMovingAveragePredictor.predict(historical_data, prediction_days)
        else:
            # Provide dummy prediction for insights generation
            prediction = {
                'summary': {
                    'expected_change_percent': 0,
                    'trend': 'neutral',
                    'confidence': 'medium'
                }
            }
        
        # Generate insights
        insights = InsightsGenerator.generate_asset_insights(
            symbol,
            current_price,
            historical_data,
            risk_analysis,
            prediction,
            news_sentiment
        )
        
        return jsonify(insights), 200
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# PORTFOLIO ANALYSIS (Multi-asset)
# ============================================================================

@app.route('/api/portfolio/analyze', methods=['POST'])
def analyze_portfolio():
    """
    Analyze a portfolio of multiple assets
    
    Request body:
    {
        "assets": [
            {"symbol": "RELIANCE.NS", "quantity": 10, "category": "stocks"},
            {"symbol": "BTC-USD", "quantity": 0.5, "category": "crypto"}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'assets' not in data:
            return jsonify({'error': 'assets array is required'}), 400
        
        assets = data['assets']
        portfolio_results = []
        total_value = 0
        
        for asset in assets:
            symbol = asset.get('symbol')
            quantity = asset.get('quantity', 0)
            category = asset.get('category', 'stocks')
            
            # Get current price
            price_data = data_fetcher.get_current_price(symbol)
            
            if price_data:
                current_price = price_data['price']
                asset_value = current_price * quantity
                total_value += asset_value
                
                # Get basic risk score
                historical_data = data_fetcher.get_historical_data(symbol, '3mo', '1d')
                risk_analysis = RiskAnalyzer.calculate_risk_score(historical_data, current_price)
                
                portfolio_results.append({
                    'symbol': symbol,
                    'category': category,
                    'quantity': quantity,
                    'current_price': current_price,
                    'asset_value': round(asset_value, 2),
                    'risk_score': risk_analysis.get('risk_score', 50),
                    'risk_level': risk_analysis.get('risk_level', 'medium')
                })
            else:
                portfolio_results.append({
                    'symbol': symbol,
                    'category': category,
                    'error': 'Unable to fetch price'
                })
        
        # Calculate portfolio-level metrics
        if portfolio_results:
            avg_risk_score = sum(a['risk_score'] for a in portfolio_results if 'risk_score' in a) / len([a for a in portfolio_results if 'risk_score' in a])
            
            # Determine portfolio risk level
            if avg_risk_score < 30:
                portfolio_risk = 'low'
            elif avg_risk_score < 50:
                portfolio_risk = 'medium-low'
            elif avg_risk_score < 70:
                portfolio_risk = 'medium'
            else:
                portfolio_risk = 'high'
        else:
            avg_risk_score = 50
            portfolio_risk = 'medium'
        
        return jsonify({
            'portfolio': portfolio_results,
            'summary': {
                'total_assets': len(assets),
                'total_value': round(total_value, 2),
                'average_risk_score': round(avg_risk_score, 2),
                'portfolio_risk_level': portfolio_risk
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

@app.route('/api/batch/quotes', methods=['POST'])
def get_batch_quotes():
    """
    Get quotes for multiple assets at once
    
    Request body:
    {
        "symbols": ["RELIANCE.NS", "TCS.NS", "BTC-USD"]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'symbols' not in data:
            return jsonify({'error': 'symbols array is required'}), 400
        
        symbols = data['symbols']
        results = data_fetcher.get_multiple_prices(symbols)
        
        return jsonify({
            'count': len(results),
            'quotes': results
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching batch quotes: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# WEBSOCKET FOR REAL-TIME DATA
# ============================================================================

@sock.route('/ws/stream')
def websocket_stream(ws):
    """
    WebSocket endpoint for real-time data streaming
    
    Client messages:
    - Subscribe: {"action": "subscribe", "symbols": ["RELIANCE.NS", "BTC-USD"]}
    - Unsubscribe: {"action": "unsubscribe"}
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
                            # Cancel existing task
                            if active_ws_connections[connection_id]['task']:
                                active_ws_connections[connection_id]['task'].cancel()
                            
                            # Start streaming
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
        logger.error(f"WebSocket error: {str(e)}")
    
    finally:
        if connection_id in active_ws_connections:
            if active_ws_connections[connection_id]['task']:
                active_ws_connections[connection_id]['task'].cancel()
            del active_ws_connections[connection_id]


def stream_stock_data(ws, symbols):
    """Stream stock data at regular intervals"""
    import time
    
    try:
        ws.send(json.dumps({
            'type': 'stream_started',
            'symbols': symbols,
            'interval': '5 seconds'
        }))
        
        while True:
            try:
                stock_data = data_fetcher.get_multiple_prices(symbols)
                
                ws.send(json.dumps({
                    'type': 'data',
                    'data': stock_data,
                    'timestamp': datetime.now().isoformat()
                }))
                
                time.sleep(5)
            
            except Exception as e:
                logger.error(f"Error in stream loop: {str(e)}")
                break
    
    except Exception as e:
        logger.error(f"Stream error: {str(e)}")
        try:
            ws.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
        except:
            pass


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.route('/api/timeframes', methods=['GET'])
def get_timeframes():
    """Get available timeframes"""
    return jsonify({
        'timeframes': config.PERIOD_INTERVAL_MAP
    }), 200


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available asset categories"""
    return jsonify({
        'categories': config.ASSET_CATEGORIES
    }), 200


@app.route('/api/popular/<category>', methods=['GET'])
def get_popular_assets(category):
    """
    Get popular assets by category
    
    Example: /api/popular/stocks
    """
    try:
        if category == 'stocks':
            assets = config.POPULAR_INDIAN_STOCKS
        elif category == 'crypto':
            assets = config.POPULAR_CRYPTOS
        elif category == 'mutualfunds':
            assets = config.POPULAR_MUTUAL_FUNDS
        else:
            return jsonify({
                'error': f'Invalid category. Choose from: {", ".join(config.ASSET_CATEGORIES)}'
            }), 400
        
        return jsonify({
            'category': category,
            'count': len(assets),
            'assets': assets
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching popular assets: {str(e)}")
        return jsonify({'error': str(e)}), 500




portfolio_manager = PortfolioService()

# ============================================================================
# PORTFOLIO TRANSACTIONS
# ============================================================================

@app.route('/api/portfolio/buy', methods=['POST'])
def buy_asset():
    """Endpoint to buy or add units to an asset"""
    try:
        data = request.json
        symbol = data.get('symbol')
        units = float(data.get('units', 0))
        price = float(data.get('price', 0))
        
        if not symbol or units <= 0:
            return jsonify({'error': 'Invalid symbol or units'}), 400
            
        result = portfolio_manager.buy_asset(symbol, units, price)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/sell', methods=['POST'])
def sell_asset():
    """Endpoint to sell assets and calculate average-cost profit"""
    try:
        data = request.json
        symbol = data.get('symbol')
        units = float(data.get('units', 0))
        
        # Automatically fetch the current market price for the sale
        market_data = data_fetcher.get_current_price(symbol)
        if not market_data:
            return jsonify({'error': 'Could not verify market price for sale'}), 404
            
        current_price = market_data['price']
        result = portfolio_manager.sell_asset(symbol, units, current_price)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/holdings', methods=['GET'])
def get_holdings():
    """Get current portfolio state"""
    return jsonify(portfolio_manager.get_portfolio()), 200
# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 BigBull Portfolio Management Microservice")
    print("=" * 80)
    print(f"\n📡 Server running on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"🔧 Environment: {config.FLASK_ENV}")
    print(f"🔑 Finnhub API: {'Configured' if config.FINNHUB_API_KEY else 'Not configured'}")
    print("\n📚 Available API Endpoints:\n")
    
    endpoints = [
        ("Health Check", "GET", "/health"),
        ("Search Assets", "GET", "/api/search/<category>?query=<q>"),
        ("Asset Quote", "GET", "/api/asset/<category>/<symbol>/quote"),
        ("Asset History", "GET", "/api/asset/<category>/<symbol>/history"),
        ("Asset Info", "GET", "/api/asset/<category>/<symbol>/info"),
        ("Asset News", "GET", "/api/news/<symbol>"),
        ("Market News", "GET", "/api/news/market/<category>"),
        ("News Sentiment", "GET", "/api/sentiment/<symbol>"),
        ("Risk Analysis", "GET", "/api/risk/<symbol>"),
        ("Price Prediction", "GET", "/api/predict/<symbol>"),
        ("Investment Insights", "GET", "/api/insights/<symbol>"),
        ("Portfolio Analysis", "POST", "/api/portfolio/analyze"),
        ("Batch Quotes", "POST", "/api/batch/quotes"),
        ("Popular Assets", "GET", "/api/popular/<category>"),
        ("Available Timeframes", "GET", "/api/timeframes"),
        ("Asset Categories", "GET", "/api/categories"),
        ("WebSocket Stream", "WS", "/ws/stream")
    ]
    
    for name, method, endpoint in endpoints:
        print(f"  {method:6s} {endpoint:50s} - {name}")
    
    print("\n" + "=" * 80)
    print("✨ Ready to serve requests!")
    print("=" * 80 + "\n")
    
    app.run(
        debug=config.FLASK_DEBUG,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT
    )