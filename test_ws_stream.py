"""
Simple test for /ws/stream WebSocket endpoint
Run: pip install websocket-client
"""

from websocket import create_connection
import json
import time

def test_websocket():
    print("=" * 50)
    print("Testing /ws/stream endpoint")
    print("=" * 50)
    
    try:
        # Connect
        print("\nConnecting to ws://localhost:5000/ws/stream...")
        ws = create_connection("ws://localhost:5000/ws/stream")
        print("✓ Connected!")
        
        # Test 1: Subscribe to AAPL
        print("\n[TEST 1] Subscribe to AAPL")
        msg = {"action": "subscribe", "symbols": ["AAPL"]}
        ws.send(json.dumps(msg))
        print(f"Sent: {msg}")
        
        # Receive responses
        for i in range(3):
            result = ws.recv()
            print(f"Received: {result}")
            time.sleep(1)
        
        # Test 2: Unsubscribe
        print("\n[TEST 2] Unsubscribe")
        msg = {"action": "unsubscribe"}
        ws.send(json.dumps(msg))
        print(f"Sent: {msg}")
        
        result = ws.recv()
        print(f"Received: {result}")
        
        # Close
        ws.close()
        print("\n✓ Test complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("1. Flask server is running (python app.py)")
        print("2. You have websocket-client installed (pip install websocket-client)")

if __name__ == "__main__":
    test_websocket()
