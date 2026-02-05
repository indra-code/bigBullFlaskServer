"""
Comprehensive test for all Flask server endpoints
"""

import requests
import json
from websocket import create_connection
import time

BASE_URL = "http://localhost:5000"

def print_test(name):
    print("\n" + "=" * 60)
    print(f"TEST: {name}")
    print("=" * 60)

def print_result(success, message, data=None):
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status}: {message}")
    if data:
        print(f"Response: {json.dumps(data, indent=2)}")

def test_health():
    print_test("GET /health")
    try:
        response = requests.get(f"{BASE_URL}/health")
        success = response.status_code == 200
        print_result(success, f"Status: {response.status_code}", response.json())
        return success
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_stock_history():
    print_test("GET /api/stock/history/<symbol>")
    try:
        # Test with default timeframe
        response = requests.get(f"{BASE_URL}/api/stock/history/AAPL")
        success = response.status_code == 200
        data = response.json()
        print_result(success, f"Status: {response.status_code}, Data points: {len(data.get('data', []))}")
        
        # Test with custom timeframe
        print("\n  Testing with timeframe=1D...")
        response = requests.get(f"{BASE_URL}/api/stock/history/AAPL?timeframe=1D")
        success = response.status_code == 200
        data = response.json()
        print_result(success, f"Status: {response.status_code}, Timeframe: {data.get('timeframe')}")
        
        return success
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_stock_info():
    print_test("GET /api/stock/info/<symbol>")
    try:
        response = requests.get(f"{BASE_URL}/api/stock/info/AAPL")
        success = response.status_code == 200
        data = response.json()
        if success and data.get('info'):
            print(f"✓ PASS: Status: {response.status_code}")
            print(f"  Company: {data['info'].get('longName', 'N/A')}")
            print(f"  Sector: {data['info'].get('sector', 'N/A')}")
            print(f"  Market Cap: {data['info'].get('marketCap', 'N/A')}")
        else:
            print_result(success, f"Status: {response.status_code}")
        return success
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_stock_quote():
    print_test("GET /api/stock/quote/<symbol>")
    try:
        response = requests.get(f"{BASE_URL}/api/stock/quote/AAPL")
        success = response.status_code == 200
        data = response.json()
        if success:
            print(f"✓ PASS: Status: {response.status_code}")
            print(f"  Symbol: {data.get('symbol')}")
            print(f"  Price: ${data.get('price')}")
            print(f"  Volume: {data.get('volume'):,}")
        else:
            print_result(success, f"Status: {response.status_code}", data)
        return success
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_search():
    print_test("GET /api/search")
    try:
        response = requests.get(f"{BASE_URL}/api/search?query=Apple&max_results=3")
        success = response.status_code == 200
        data = response.json()
        if success:
            results = data.get('results', {}).get('quotes', [])
            print(f"✓ PASS: Status: {response.status_code}")
            print(f"  Query: {data.get('query')}")
            print(f"  Results found: {len(results)}")
            for r in results[:3]:
                print(f"    - {r.get('symbol')}: {r.get('longname', 'N/A')}")
        else:
            print_result(success, f"Status: {response.status_code}", data)
        return success
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_multiple_stocks():
    print_test("POST /api/stock/multiple")
    try:
        payload = {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "timeframe": "1D"
        }
        response = requests.post(
            f"{BASE_URL}/api/stock/multiple",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        success = response.status_code == 200
        data = response.json()
        if success:
            print(f"✓ PASS: Status: {response.status_code}")
            print(f"  Timeframe: {data.get('timeframe')}")
            results = data.get('results', {})
            for symbol, result in results.items():
                status = "✓" if result.get('success') else "✗"
                count = len(result.get('data', [])) if result.get('success') else 0
                print(f"    {status} {symbol}: {count} data points")
        else:
            print_result(success, f"Status: {response.status_code}", data)
        return success
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_timeframes():
    print_test("GET /api/timeframes")
    try:
        response = requests.get(f"{BASE_URL}/api/timeframes")
        success = response.status_code == 200
        data = response.json()
        if success:
            print(f"✓ PASS: Status: {response.status_code}")
            print(f"  Available timeframes:")
            for tf, config in data.get('timeframes', {}).items():
                print(f"    {tf}: period={config['period']}, interval={config['interval']}")
        else:
            print_result(success, f"Status: {response.status_code}", data)
        return success
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_websocket():
    print_test("WS /ws/stream")
    try:
        ws = create_connection("ws://localhost:5000/ws/stream")
        print("✓ Connected to websocket")
        
        # Subscribe
        msg = {"action": "subscribe", "symbols": ["AAPL"]}
        ws.send(json.dumps(msg))
        print(f"  Sent: {msg}")
        
        # Receive 2 messages
        for i in range(2):
            result = ws.recv()
            data = json.loads(result)
            print(f"  Received: type={data.get('type')}")
        
        # Unsubscribe
        msg = {"action": "unsubscribe"}
        ws.send(json.dumps(msg))
        result = ws.recv()
        data = json.loads(result)
        print(f"  Unsubscribed: {data.get('type')}")
        
        ws.close()
        print_result(True, "Websocket test completed")
        return True
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("FLASK SERVER ENDPOINT TESTS")
    print("=" * 60)
    print(f"Server: {BASE_URL}")
    print("Make sure the Flask server is running!")
    
    results = []
    
    # REST API Tests
    results.append(("Health Check", test_health()))
    results.append(("Stock History", test_stock_history()))
    results.append(("Stock Info", test_stock_info()))
    results.append(("Stock Quote", test_stock_quote()))
    results.append(("Search", test_search()))
    results.append(("Multiple Stocks", test_multiple_stocks()))
    results.append(("Timeframes", test_timeframes()))
    
    # WebSocket Test
    results.append(("WebSocket Stream", test_websocket()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
