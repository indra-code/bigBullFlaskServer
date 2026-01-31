from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
import yfinance as yf
import asyncio
import json
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Period and interval mappings
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
    Stream stock data using yfinance AsyncWebSocket (from test.py)
    """
    async def async_stream():
        try:
            async with yf.AsyncWebSocket() as yf_ws:
                await yf_ws.subscribe(symbols)
                
                async for message in yf_ws.stream():
                    try:
                        ws.send(json.dumps({
                            'type': 'data',
                            'data': message
                        }))
                    except Exception as e:
                        print(f"Error sending message: {e}")
                        break
        
        except Exception as e:
            try:
                ws.send(json.dumps({
                    'type': 'error',
                    'message': str(e)
                }))
            except:
                pass
    
    # Run async function
    try:
        asyncio.run(async_stream())
    except Exception as e:
        print(f"Stream error: {e}")


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
