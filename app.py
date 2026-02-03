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
        max_len = max(len(lst) for lst in returns_data)

        # Option 1: Pad arrays with NaN to the length of the longest array
        returns_data_padded = [
            np.pad(lst, (0, max_len - len(lst)), mode='constant', constant_values=0)
            for lst in returns_data
        ]
        returns_array = np.array(returns_data_padded)
        mean_returns = np.nanmean(returns_array, axis=1)
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


if __name__ == '__main__':
    print("Starting BigBull Flask Server...")
    print("Server running on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  - GET  /health")
    print("  - GET  /api/stock/history/<symbol>")
    print("  - GET  /api/stock/info/<symbol>")
    print("  - GET  /api/stock/quote/<symbol>")
    print("  - GET  /api/search")
    print("  - POST /api/stock/multiple")
    print("  - GET  /api/timeframes")
    print("  - WS   /ws/stream")
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
