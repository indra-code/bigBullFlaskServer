from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
import yfinance as yf
import asyncio
import json
from datetime import datetime
import threading
import requests
import numpy as np
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import RealAmplitudes
from qiskit.result import QuasiDistribution
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)
sock = Sock(app)

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


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200


@app.route('/api/stock/history/<symbol>', methods=['GET'])
def get_stock_history(symbol):
    """
    Get stock historical data (from test3.py)
    Query params:
    - timeframe: 1D, 5D, 1M, 3M, 6M, 1Y, 5Y, MAX (default: 1M)
    - period: custom period (overrides timeframe)
    - interval: custom interval (overrides timeframe)
    """
    try:
        # Get timeframe from query params
        timeframe = request.args.get('timeframe', '1M').upper()
        
        # Check if custom period/interval provided
        custom_period = request.args.get('period')
        custom_interval = request.args.get('interval')
        
        if custom_period and custom_interval:
            period = custom_period
            interval = custom_interval
        elif timeframe in PERIOD_INTERVAL_MAP:
            period = PERIOD_INTERVAL_MAP[timeframe]['period']
            interval = PERIOD_INTERVAL_MAP[timeframe]['interval']
        else:
            return jsonify({
                'error': f'Invalid timeframe. Choose from: {", ".join(PERIOD_INTERVAL_MAP.keys())}'
            }), 400
        
        # Get ticker data (from test3.py)
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period, interval=interval)
        
        # Convert DataFrame to JSON-serializable format
        if history.empty:
            return jsonify({
                'error': f'No data found for symbol {symbol}'
            }), 404
        
        # Reset index to make Date a column
        history_reset = history.reset_index()
        
        # Convert to dict
        data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'interval': interval,
            'data': history_reset.to_dict(orient='records')
        }
        
        # Convert Timestamp objects to strings
        for record in data['data']:
            if 'Date' in record or 'Datetime' in record:
                date_key = 'Date' if 'Date' in record else 'Datetime'
                record[date_key] = record[date_key].isoformat() if hasattr(record[date_key], 'isoformat') else str(record[date_key])
        
        return jsonify(data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/info/<symbol>', methods=['GET'])
def get_stock_info(symbol):
    """Get detailed stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return jsonify({
            'symbol': symbol,
            'info': info
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search_stocks():
    """
    Search for stocks (from test2.py)
    Query params:
    - query: search query (required)
    - max_results: maximum number of results (default: 5)
    - news_count: number of news items to include (default: 3)
    """
    try:
        query = request.args.get('query')
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        max_results = int(request.args.get('max_results', 5))
        #news_count = int(request.args.get('news_count', 3))
        
        # Perform search (from test2.py)
        search_result = yf.Search(query, max_results=max_results)
        
        return jsonify({
            'query': query,
            'results': search_result.response
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/quote/<symbol>', methods=['GET'])
def get_stock_quote(symbol):
    """Get current stock quote"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current data
        hist = ticker.history(period='1d', interval='1m')
        if hist.empty:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        latest = hist.iloc[-1]
        
        quote = {
            'symbol': symbol,
            'price': float(latest['Close']),
            'open': float(latest['Open']),
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'volume': int(latest['Volume']),
            'timestamp': hist.index[-1].isoformat()
        }
        
        return jsonify(quote), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
        
        results = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                history = ticker.history(period=period, interval=interval)
                
                if not history.empty:
                    history_reset = history.reset_index()
                    data_list = history_reset.to_dict(orient='records')
                    
                    # Convert timestamps
                    for record in data_list:
                        if 'Date' in record or 'Datetime' in record:
                            date_key = 'Date' if 'Date' in record else 'Datetime'
                            record[date_key] = record[date_key].isoformat() if hasattr(record[date_key], 'isoformat') else str(record[date_key])
                    
                    results[symbol] = {
                        'success': True,
                        'data': data_list
                    }
                else:
                    results[symbol] = {
                        'success': False,
                        'error': 'No data found'
                    }
            except Exception as e:
                results[symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        return jsonify({
            'timeframe': timeframe,
            'period': period,
            'interval': interval,
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
        # Note: budget parameter in Qiskit is not a hard constraint, it's a penalty weight
        # We'll filter results after quantum optimization
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


# WebSocket endpoint for real-time streaming (from test.py)
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
    import time
    
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
@app.route('/getAssets', methods=['GET'])
def get_assets():
    assetData = requests.get('http://localhost:8080/api/assets')
    return assetData.json()


@app.route('/api/stock/risk/<symbol>', methods=['GET'])
def get_stock_risk(symbol):
    """
    Get risk metrics for a stock using basic financial metrics
    
    ALGORITHM EXPLANATION:
    =====================
    Uses yfinance stock info data to calculate risk score based on:
    - Beta: Measures volatility relative to market (S&P 500)
    - 52-week change: Annual performance volatility
    
    Risk Score Formula (0-100):
    - Beta component (max 50 points): |beta - 1| × 50
    - Volatility component (max 50 points): |52-week change| × 100
    
    Risk Ratings:
    - 0-20:   Low Risk (stable, defensive stocks)
    - 20-50:  Moderate Risk (typical market stocks)
    - 50-75:  High Risk (volatile growth stocks)
    - 75-100: Very High Risk (extremely volatile)
    
    Query params: None required
    Returns risk score and detailed metrics
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        print(f"\n📊 Risk Analysis for {symbol.upper()}")
        
        # Get beta and volatility from info
        beta = info.get('beta', 1.0)
        beta = float(beta) if beta else 1.0
        
        # Get 52-week change as volatility proxy
        fifty_two_week_change = info.get('52WeekChange', 0)
        fifty_two_week_change = abs(float(fifty_two_week_change)) if fifty_two_week_change else 0
        
        print(f"Beta: {beta}")
        print(f"52-week change: {fifty_two_week_change}")
        
        # Risk score calculation
        # Beta component: Distance from market neutral (1.0)
        # A beta of 0.5 or 1.5 both indicate deviation from market
        beta_score = min(abs(beta - 1) * 50, 50)  # Max 50 points from beta
        
        # Volatility component: 52-week price change magnitude
        volatility_score = min(fifty_two_week_change * 100, 50)  # Max 50 points from 52w change
        
        # Composite score
        risk_score = beta_score + volatility_score
        
        print(f"Beta score: {beta_score:.2f}, Volatility score: {volatility_score:.2f}")
        print(f"Total risk score: {risk_score:.2f}")
        
        # Assign risk rating
        if risk_score < 20:
            risk_rating = "Low"
        elif risk_score < 50:
            risk_rating = "Moderate"
        elif risk_score < 75:
            risk_rating = "High"
        else:
            risk_rating = "Very High"
        
        return jsonify({
            'symbol': symbol.upper(),
            'risk_score': round(risk_score, 2),
            'risk_rating': risk_rating,
            'risk_type': 'Basic Financial',
            'metrics': {
                'beta': round(beta, 4),
                'fifty_two_week_change': round(fifty_two_week_change, 4),
                'beta_score': round(beta_score, 2),
                'volatility_score': round(volatility_score, 2)
            },
            'interpretation': {
                'beta': f"Beta of {round(beta, 2)} - {'More' if beta > 1 else 'Less'} volatile than market",
                'volatility': f"52-week change of {round(fifty_two_week_change * 100, 2)}%",
                'risk_assessment': f"{risk_rating} risk based on market correlation and price volatility"
            },
            'data_source': 'yfinance stock info',
            'algorithm_version': '3.0-BasicFinancial'
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


# ==================== CHATBOT ENDPOINT ====================

# Tool definitions for the chatbot
CHATBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_assets",
            "description": "Get the user's current portfolio assets and holdings. Use this when the user asks about their portfolio, holdings, or what stocks they own.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "Get detailed information about a stock including company info, market cap, PE ratio, dividend yield, etc. Use when user asks about company details or fundamentals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Get current real-time price and quote data for a stock. Use when user asks about current price, today's price, or latest quote.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_history",
            "description": "Get historical price data for a stock over a specified time period. Use when user asks about historical performance, price trends, or past data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y", "MAX"],
                        "description": "Time period for historical data (default: 1M)",
                        "default": "1M"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_stocks",
            "description": "Search for stocks by company name or ticker symbol. Use when user mentions a company name but you need the ticker, or when searching for stocks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (company name or partial ticker)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_recommendations",
            "description": "Get analyst recommendations and ratings for a stock. Use when user asks about analyst opinions, recommendations, or ratings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": "Get latest news articles about a stock. Use when user asks about news, recent events, or updates about a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_portfolio",
            "description": "Use quantum computing to optimize a portfolio of stocks. Use when user asks about portfolio optimization, asset allocation, or which stocks to invest in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols to include in optimization"
                    },
                    "target_assets": {
                        "type": "integer",
                        "description": "Target number of assets to select (default: half of provided symbols)",
                        "default": None
                    }
                },
                "required": ["symbols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_risk",
            "description": "Calculate risk score and metrics for a stock including volatility, beta, maximum drawdown, VaR, and Sharpe ratio. Use when user asks about risk, safety, or volatility of a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical period for analysis (default: 1y). Options: 6mo, 1y, 2y, 5y",
                        "default": "1y"
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]


def execute_tool(tool_name, arguments):
    """Execute the requested tool and return results"""
    print(f"\n🔧 Tool Called: {tool_name}")
    print(f"   Arguments: {arguments}")
    try:
        if tool_name == "get_user_assets":
            # Call the getAssets endpoint internally
            response = requests.get('http://localhost:8080/api/assets')
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": "Failed to fetch user assets"}
        
        elif tool_name == "get_stock_info":
            symbol = arguments.get("symbol", "").upper()
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {"success": True, "symbol": symbol, "info": info}
        
        elif tool_name == "get_stock_quote":
            symbol = arguments.get("symbol", "").upper()
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='1m')
            if hist.empty:
                return {"success": False, "error": f"No data found for {symbol}"}
            latest = hist.iloc[-1]
            quote = {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'timestamp': hist.index[-1].isoformat()
            }
            return {"success": True, "quote": quote}
        
        elif tool_name == "get_stock_history":
            symbol = arguments.get("symbol", "").upper()
            timeframe = arguments.get("timeframe", "1M").upper()
            
            if timeframe not in PERIOD_INTERVAL_MAP:
                return {"success": False, "error": f"Invalid timeframe: {timeframe}"}
            
            period = PERIOD_INTERVAL_MAP[timeframe]['period']
            interval = PERIOD_INTERVAL_MAP[timeframe]['interval']
            
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if history.empty:
                return {"success": False, "error": f"No data found for {symbol}"}
            
            # Convert to list of dicts (limit to last 30 data points for chatbot)
            history_reset = history.reset_index().tail(30)
            data_list = history_reset.to_dict(orient='records')
            
            # Convert timestamps
            for record in data_list:
                if 'Date' in record or 'Datetime' in record:
                    date_key = 'Date' if 'Date' in record else 'Datetime'
                    record[date_key] = record[date_key].isoformat() if hasattr(record[date_key], 'isoformat') else str(record[date_key])
            
            return {"success": True, "symbol": symbol, "timeframe": timeframe, "data": data_list}
        
        elif tool_name == "search_stocks":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            
            search_result = yf.Search(query, max_results=max_results)
            return {"success": True, "query": query, "results": search_result.response}
        
        elif tool_name == "get_stock_recommendations":
            symbol = arguments.get("symbol", "").upper()
            ticker = yf.Ticker(symbol)
            
            # Get recommendations
            recommendations = ticker.recommendations
            if recommendations is not None and not recommendations.empty:
                # Convert to dict and get recent recommendations
                rec_data = recommendations.tail(10).reset_index().to_dict(orient='records')
                # Convert timestamps
                for rec in rec_data:
                    if 'Date' in rec:
                        rec['Date'] = rec['Date'].isoformat() if hasattr(rec['Date'], 'isoformat') else str(rec['Date'])
                return {"success": True, "symbol": symbol, "recommendations": rec_data}
            else:
                return {"success": False, "error": f"No recommendations found for {symbol}"}
        
        elif tool_name == "get_stock_news":
            symbol = arguments.get("symbol", "").upper()
            ticker = yf.Ticker(symbol)
            
            # Get news
            news = ticker.news
            if news:
                return {"success": True, "symbol": symbol, "news": news[:10]}  # Limit to 10 news items
            else:
                return {"success": False, "error": f"No news found for {symbol}"}
        
        elif tool_name == "optimize_portfolio":
            symbols = arguments.get("symbols", [])
            target_assets = arguments.get("target_assets")
            
            if not symbols or len(symbols) < 2:
                return {"success": False, "error": "At least 2 symbols required for optimization"}
            
            # Build query params
            params = {"symbols": ",".join(symbols)}
            if target_assets:
                params["target_assets"] = target_assets
            
            # This is a simplified response - user can call the full endpoint for details
            return {
                "success": True,
                "message": f"To optimize a portfolio with {', '.join(symbols)}, please use the /api/portfolio/optimize endpoint with these symbols.",
                "endpoint": "/api/portfolio/optimize",
                "params": params
            }
        
        elif tool_name == "get_stock_risk":
            # Call the risk endpoint internally (avoid code duplication)
            symbol = arguments.get("symbol", "").upper()
            period = arguments.get("period", "1y")
            
            try:
                # Make internal request to the risk endpoint
                response = requests.get(f'http://localhost:5000/api/stock/risk/{symbol}?period={period}')
                if response.status_code == 200:
                    risk_data = response.json()
                    return {
                        "success": True,
                        "symbol": risk_data['symbol'],
                        "risk_score": risk_data['risk_score'],
                        "risk_rating": risk_data['risk_rating'],
                        "volatility": risk_data['metrics']['volatility'],
                        "beta": risk_data['metrics']['beta'],
                        "max_drawdown_pct": round(risk_data['metrics']['max_drawdown'] * 100, 2),
                        "sharpe_ratio": risk_data['metrics']['sharpe_ratio']
                    }
                else:
                    return {"success": False, "error": f"Failed to get risk data for {symbol}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        else:
            result = {"success": False, "error": f"Unknown tool: {tool_name}"}
            print(f"   Result: {result}")
            return result
    
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        print(f"   ❌ Error: {str(e)}")
        return error_result


@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """
    Natural language chatbot for stock queries
    Request body:
    {
        "message": "What is the current price of Apple stock?",
        "conversation_history": []  # Optional: previous messages for context
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'message is required'}), 400
        
        user_message = data['message']
        conversation_history = data.get('conversation_history', [])
        
        # Get Google API key from environment
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return jsonify({
                'error': 'Google API key not configured. Please set GOOGLE_API_KEY environment variable.'
            }), 500
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create model with tools using proper Gemini function declarations
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite',
            tools=[
                {
                    'function_declarations': [
                        {'name': 'get_user_assets', 'description': 'Get the user\'s current portfolio assets and holdings. Use this when the user asks about their portfolio, holdings, or what stocks they own.', 'parameters': {'type': 'OBJECT', 'properties': {}, 'required': []}},
                        {'name': 'get_stock_info', 'description': 'Get detailed information about a stock including company info, market cap, PE ratio, dividend yield, etc. Use when user asks about company details or fundamentals.', 'parameters': {'type': 'OBJECT', 'properties': {'symbol': {'type': 'STRING', 'description': 'Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'}}, 'required': ['symbol']}},
                        {'name': 'get_stock_quote', 'description': 'Get current real-time price and quote data for a stock. Use when user asks about current price, today\'s price, or latest quote.', 'parameters': {'type': 'OBJECT', 'properties': {'symbol': {'type': 'STRING', 'description': 'Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'}}, 'required': ['symbol']}},
                        {'name': 'get_stock_history', 'description': 'Get historical price data for a stock over a specified time period. Use when user asks about historical performance, price trends, or past data.', 'parameters': {'type': 'OBJECT', 'properties': {'symbol': {'type': 'STRING', 'description': 'Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'}, 'timeframe': {'type': 'STRING', 'description': 'Time period for historical data (1D, 5D, 1M, 3M, 6M, 1Y, 5Y, MAX)', 'enum': ['1D', '5D', '1M', '3M', '6M', '1Y', '5Y', 'MAX']}}, 'required': ['symbol']}},
                        {'name': 'search_stocks', 'description': 'Search for stocks by company name or ticker symbol. Use when user mentions a company name but you need the ticker, or when searching for stocks.', 'parameters': {'type': 'OBJECT', 'properties': {'query': {'type': 'STRING', 'description': 'Search query (company name or partial ticker)'}, 'max_results': {'type': 'INTEGER', 'description': 'Maximum number of results to return (default: 5)'}}, 'required': ['query']}},
                        {'name': 'get_stock_recommendations', 'description': 'Get analyst recommendations and ratings for a stock. Use when user asks about analyst opinions, recommendations, or ratings.', 'parameters': {'type': 'OBJECT', 'properties': {'symbol': {'type': 'STRING', 'description': 'Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'}}, 'required': ['symbol']}},
                        {'name': 'get_stock_news', 'description': 'Get latest news articles about a stock. Use when user asks about news, recent events, or updates about a company.', 'parameters': {'type': 'OBJECT', 'properties': {'symbol': {'type': 'STRING', 'description': 'Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'}}, 'required': ['symbol']}},
                        {'name': 'get_stock_risk', 'description': 'Calculate risk score and metrics for a stock including volatility, beta, maximum drawdown, VaR, and Sharpe ratio. Use when user asks about risk, safety, or volatility of a stock.', 'parameters': {'type': 'OBJECT', 'properties': {'symbol': {'type': 'STRING', 'description': 'Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'}, 'period': {'type': 'STRING', 'description': 'Historical period for analysis (default: 1y)'}}, 'required': ['symbol']}},
                        {'name': 'optimize_portfolio', 'description': 'Use quantum computing to optimize a portfolio of stocks. Use when user asks about portfolio optimization, asset allocation, or which stocks to invest in.', 'parameters': {'type': 'OBJECT', 'properties': {'symbols': {'type': 'ARRAY', 'items': {'type': 'STRING'}, 'description': 'List of stock ticker symbols to include in optimization'}, 'target_assets': {'type': 'INTEGER', 'description': 'Target number of assets to select (default: half of provided symbols)'}}, 'required': ['symbols']}}
                    ]
                }
            ]
        )
        
        # Build chat history for Gemini
        chat_history = []
        for msg in conversation_history:
            if msg['role'] == 'user':
                chat_history.append({'role': 'user', 'parts': [msg['content']]})
            elif msg['role'] == 'assistant':
                chat_history.append({'role': 'model', 'parts': [msg['content']]})
        
        # Start chat with history
        chat = model.start_chat(history=chat_history)
        
        # System instruction context
        system_context = """You are a helpful financial assistant for the BigBull trading platform. 
You can help users with:
- Information about stocks (prices, company details, historical data)
- Their portfolio and holdings
- Stock recommendations and analyst ratings
- Latest news about companies
- Portfolio optimization using quantum computing

CRITICAL RULES - Follow these strictly:

1. SINGLE STOCK ANALYSIS: When analyzing a stock, analyze ONLY ONE stock. Never provide analysis for multiple stocks in a single response.

2. SEARCH BEHAVIOR: When using search_stocks tool:
   - ALWAYS take the FIRST symbol from search results
   - Use that symbol for ALL subsequent analysis (risk, news, recommendations)
   - NEVER ask the user to choose between multiple options
   - NEVER list multiple stock options
   - Proceed directly with analysis of the first matching stock

3. STOCK RECOMMENDATIONS: When providing stock recommendations:
   - ALWAYS use get_stock_risk tool to check the risk score and metrics
   - ALWAYS use get_stock_news tool to check recent news and events
   - ALWAYS use get_stock_recommendations tool to see analyst ratings
   - Synthesize insights into 2-3 short paragraphs with clear conclusion
   - For Indian stocks: Try both .NS (NSE) and .BO (BSE) ticker suffixes if one fails

4. RESPONSE FORMAT: Keep responses concise and focused:
   - Paragraph 1: Risk assessment (score, beta, volatility) with interpretation
   - Paragraph 2: Key news insights (recent earnings, partnerships, sector trends)
   - Paragraph 3: Clear recommendation with reasoning (buy/hold/avoid and why)

5. CLARITY: Your response should contain ONLY the analysis for the ONE stock being discussed. No multiple options, no alternatives, no suggestions to search for other stocks.

Always be concise and helpful. When providing numerical data, format it clearly.
When a stock symbol is mentioned, always use uppercase ticker symbols."""
        
        # Send message with context
        full_message = f"{system_context}\n\nUser: {user_message}"
        response = chat.send_message(full_message)
        
        # Check if model wants to use tools (max 10 iterations to prevent infinite loops)
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if response has valid candidates and parts
            if not response.candidates or not response.candidates[0].content.parts:
                break
            
            part = response.candidates[0].content.parts[0]
            
            # Check for function call
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                function_name = function_call.name
                function_args = dict(function_call.args)
                
                print(f"\n🤖 Gemini requested tool: {function_name}")
                
                # Execute the tool
                function_response = execute_tool(function_name, function_args)
                print(f"   ✅ Tool completed")
                
                # Send function response back to model
                response = chat.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=function_name,
                                response={'result': function_response}
                            )
                        )]
                    )
                )
            else:
                # No function call, we should have text
                break
        
        # Extract final text response with proper error handling
        final_message = None
        try:
            if hasattr(response, 'text') and response.text:
                final_message = response.text
        except (ValueError, AttributeError) as e:
            print(f"   ⚠️ Error extracting text: {e}")
        
        # Fallback: try to extract from parts directly
        if not final_message:
            try:
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_message = part.text
                            break
            except Exception as e:
                print(f"   ⚠️ Error extracting from parts: {e}")
        
        # Final fallback
        if not final_message:
            final_message = "I apologize, but I encountered an error generating a response. Please try again."
        
        print(f"\n💬 Final Response: {final_message[:100]}..." if len(final_message) > 100 else f"\n💬 Final Response: {final_message}")
        
        # Build updated conversation history
        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_message}
        ]
        
        # Return response
        return jsonify({
            'response': final_message,
            'conversation_history': updated_history
        }), 200
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Chatbot error:\n{error_details}")
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'details': error_details
        }), 500


if __name__ == '__main__':
    print("Starting BigBull Flask Server...")
    print("Server running on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  - GET  /health")
    print("  - GET  /api/stock/history/<symbol>")
    print("  - GET  /api/stock/info/<symbol>")
    print("  - GET  /api/stock/quote/<symbol>")
    print("  - GET  /api/stock/risk/<symbol>")
    print("  - GET  /api/search")
    print("  - POST /api/stock/multiple")
    print("  - GET  /api/timeframes")
    print("  - GET  /api/portfolio/optimize")
    print("  - POST /api/chatbot")
    print("  - GET  /getAssets")
    print("  - WS   /ws/stream")
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
