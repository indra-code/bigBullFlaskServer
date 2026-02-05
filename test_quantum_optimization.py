"""
Test file for Quantum Portfolio Optimization endpoint
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"
ASSETS_URL = "http://localhost:8080"

def test_assets_api():
    """Check if assets API is available"""
    print("=" * 60)
    print("Checking Assets API (localhost:8080)")
    print("=" * 60)
    try:
        response = requests.get(f"{ASSETS_URL}/api/assets", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Assets API is available")
            
            # Try to extract tickers
            if isinstance(data, list):
                tickers = [asset.get('ticker') or asset.get('symbol') or asset.get('name') 
                          for asset in data if asset.get('ticker') or asset.get('symbol') or asset.get('name')]
            elif isinstance(data, dict) and 'assets' in data:
                tickers = [asset.get('ticker') or asset.get('symbol') or asset.get('name') 
                          for asset in data['assets'] if asset.get('ticker') or asset.get('symbol') or asset.get('name')]
            else:
                tickers = []
            
            print(f"  Found {len(tickers)} tickers: {tickers[:5]}{'...' if len(tickers) > 5 else ''}")
            return True, tickers
        else:
            print(f"✗ Assets API returned status {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to assets API")
        print("  Make sure the assets API is running on http://localhost:8080")
        return False, []
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, []

def test_quantum_optimization(risk_factor=0.5, period="1mo", interval="1d", budget=None):
    """Test quantum optimization endpoint"""
    print("\n" + "=" * 60)
    print(f"Testing Quantum Optimization")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Risk Factor: {risk_factor}")
    print(f"  Period: {period}")
    print(f"  Interval: {interval}")
    print(f"  Budget: {budget or 'auto (half of assets)'}")
    print()
    
    try:
        # Build URL with parameters
        params = {
            'risk_factor': risk_factor,
            'period': period,
            'interval': interval
        }
        if budget:
            params['budget'] = budget
        
        print("Sending request...")
        print(f"URL: {BASE_URL}/api/quantum/optimize")
        print(f"Params: {params}")
        print("\n⏳ Running quantum optimization (this may take 30-60 seconds)...\n")
        
        start_time = time.time()
        response = requests.get(
            f"{BASE_URL}/api/quantum/optimize",
            params=params,
            timeout=120
        )
        elapsed = time.time() - start_time
        
        print(f"Response received in {elapsed:.2f} seconds")
        print(f"Status Code: {response.status_code}\n")
        
        if response.status_code == 200:
            data = response.json()
            
            print("✓ OPTIMIZATION SUCCESSFUL!")
            print("\n" + "-" * 60)
            print("OPTIMIZATION PARAMETERS")
            print("-" * 60)
            opt = data.get('optimization', {})
            print(f"  Risk Factor: {opt.get('risk_factor')}")
            print(f"  Budget: {opt.get('budget')} assets")
            print(f"  Total Assets: {opt.get('num_assets')}")
            print(f"  Period: {opt.get('period')}")
            print(f"  Interval: {opt.get('interval')}")
            print(f"  Data Points: {opt.get('data_points')}")
            
            print("\n" + "-" * 60)
            print("RESULTS")
            print("-" * 60)
            results = data.get('results', {})
            print(f"  Optimal Value: {results.get('optimal_value'):.6f}")
            print(f"  Selected Assets: {results.get('num_selected')}/{opt.get('num_assets')}")
            
            print("\n  Selected Stocks:")
            for stock in results.get('selected_stocks', []):
                print(f"    • {stock}")
            
            print("\n  Selection Vector: {0=Not Selected, 1=Selected}")
            selection = results.get('selection', [])
            portfolio_data = data.get('portfolio_data', {})
            tickers = portfolio_data.get('tickers', [])
            
            for i, (ticker, selected) in enumerate(zip(tickers, selection)):
                status = "✓" if selected == 1 else "✗"
                print(f"    {status} {ticker}: {selected}")
            
            # Show failed tickers if any
            failed = portfolio_data.get('failed_tickers', [])
            if failed:
                print(f"\n  ⚠ Failed to fetch data for {len(failed)} ticker(s):")
                for ticker in failed:
                    print(f"    • {ticker}")
            
            print("\n" + "-" * 60)
            print("RISK-RETURN ANALYSIS")
            print("-" * 60)
            returns_daily = portfolio_data.get('expected_returns_daily', [])
            returns_annual = portfolio_data.get('expected_returns_annualized', [])
            period_ret = portfolio_data.get('period_returns', [])
            vol_annual = portfolio_data.get('volatility_annualized', [])
            
            print("\n  Period Returns vs Risk:")
            print(f"  {'Ticker':<8} {'Return %':<12} {'Annual Vol %':<15} {'Risk/Return':<12} Selected")
            print("  " + "-" * 65)
            for ticker, ret, vol in zip(tickers, period_ret, vol_annual):
                selected = "✓" if ticker in results.get('selected_stocks', []) else " "
                sign = "↑" if ret > 0 else "↓"
                risk_return_ratio = abs(vol / ret) if ret != 0 else float('inf')
                print(f"  {ticker:<8} {sign} {ret*100:>8.2f}%   {vol*100:>10.2f}%      {risk_return_ratio:>8.2f}      [{selected}]")
            
            print("\n  Annualized Returns:")
            for ticker, ret in zip(tickers, returns_annual):
                status = "✓" if ticker in results.get('selected_stocks', []) else " "
                sign = "+" if ret > 0 else ""
                print(f"  {status} {ticker}: {sign}{ret*100:.2f}% per year")
            
            # Show explanation if available
            explanation = data.get('explanation', {})
            if explanation.get('note'):
                print("\n" + "-" * 60)
                print("WHY THIS SELECTION?")
                print("-" * 60)
                print(f"  {explanation['note']}")
                print(f"\n  Objective: {explanation.get('objective', 'N/A')}")
                print(f"\n  {explanation.get('interpretation', '')}")
            
            print("\n" + "=" * 60)
            return True
            
        else:
            print("✗ OPTIMIZATION FAILED")
            print(f"\nError Response:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (>120 seconds)")
        print("  The optimization is taking too long")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def run_all_tests():
    """Run all quantum optimization tests"""
    print("\n" + "=" * 60)
    print("QUANTUM PORTFOLIO OPTIMIZATION TEST SUITE")
    print("=" * 60)
    print(f"Flask Server: {BASE_URL}")
    print(f"Assets API: {ASSETS_URL}")
    print("=" * 60)
    
    # Check assets API
    assets_available, tickers = test_assets_api()
    
    if not assets_available:
        print("\n" + "=" * 60)
        print("⚠ WARNING: Assets API not available!")
        print("=" * 60)
        print("The quantum optimization endpoint requires an assets API")
        print("running on http://localhost:8080/api/assets")
        print("\nTo proceed, you need to:")
        print("1. Start the assets API server on port 8080")
        print("2. Ensure it returns a list of assets with ticker/symbol fields")
        return
    
    # Run tests with different parameters
    print("\n" + "=" * 60)
    print("TEST 1: Default parameters (1 month, daily)")
    print("=" * 60)
    test_quantum_optimization(risk_factor=0.5, period="1mo", interval="1d")
    
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("TEST 2: Conservative (low risk)")
    print("=" * 60)
    test_quantum_optimization(risk_factor=0.3, period="3mo", interval="1d")
    
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("TEST 3: Aggressive (high risk)")
    print("=" * 60)
    test_quantum_optimization(risk_factor=0.8, period="1mo", interval="1d")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    # You can run individual test or full suite
    import sys
    
    if len(sys.argv) > 1:
        # Custom test with command line args
        risk = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
        period = sys.argv[2] if len(sys.argv) > 2 else "1mo"
        interval = sys.argv[3] if len(sys.argv) > 3 else "1d"
        
        # Check assets API first
        assets_available, _ = test_assets_api()
        if assets_available:
            test_quantum_optimization(risk_factor=risk, period=period, interval=interval)
    else:
        # Run full test suite
        run_all_tests()
