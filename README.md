# BigBull Flask Server API Documentation

A Flask-based REST API server providing stock market data and quantum portfolio optimization using Qiskit.

## Base URL
```
http://localhost:5000
```

## Table of Contents
- [Health Check](#health-check)
- [Stock Data Endpoints](#stock-data-endpoints)
  - [Get Stock History](#get-stock-history)
  - [Get Stock Info](#get-stock-info)
  - [Get Stock Quote](#get-stock-quote)
  - [Search Stocks](#search-stocks)
  - [Get Multiple Stocks](#get-multiple-stocks)
  - [Get Available Timeframes](#get-available-timeframes)
- [Portfolio Optimization](#portfolio-optimization)
  - [Quantum Portfolio Optimizer](#quantum-portfolio-optimizer)
- [WebSocket Streaming](#websocket-streaming)
- [Error Handling](#error-handling)

---

## Health Check

### `GET /health`

Check if the server is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-01T10:30:45.123456"
}
```

**Status Codes:**
- `200` - Server is healthy

---

## Stock Data Endpoints

### Get Stock History

`GET /api/stock/history/<symbol>`

Get historical stock data for a given symbol.

**URL Parameters:**
- `symbol` (string, required) - Stock ticker symbol (e.g., "AAPL", "TSLA")

**Query Parameters:**
- `timeframe` (string, optional) - Predefined timeframe. Default: `1M`
  - Options: `1D`, `5D`, `1M`, `3M`, `6M`, `1Y`, `5Y`, `MAX`
- `period` (string, optional) - Custom period (overrides timeframe)
- `interval` (string, optional) - Custom interval (overrides timeframe)

**Example Request:**
```
GET /api/stock/history/AAPL?timeframe=1M
GET /api/stock/history/TSLA?period=1y&interval=1d
```

**Response:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1M",
  "period": "1mo",
  "interval": "30m",
  "data": [
    {
      "Date": "2026-01-01T09:30:00",
      "Open": 150.25,
      "High": 152.30,
      "Low": 149.80,
      "Close": 151.50,
      "Volume": 1250000
    }
    // ... more data points
  ]
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid timeframe
- `404` - No data found for symbol
- `500` - Server error

---

### Get Stock Info

`GET /api/stock/info/<symbol>`

Get detailed information about a stock.

**URL Parameters:**
- `symbol` (string, required) - Stock ticker symbol

**Example Request:**
```
GET /api/stock/info/AAPL
```

**Response:**
```json
{
  "symbol": "AAPL",
  "info": {
    "shortName": "Apple Inc.",
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 2800000000000,
    "currentPrice": 175.25,
    "previousClose": 174.50,
    "fiftyTwoWeekHigh": 182.00,
    "fiftyTwoWeekLow": 135.50,
    "dividendYield": 0.0055,
    "beta": 1.25
    // ... more fields
  }
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

### Get Stock Quote

`GET /api/stock/quote/<symbol>`

Get current real-time quote for a stock.

**URL Parameters:**
- `symbol` (string, required) - Stock ticker symbol

**Example Request:**
```
GET /api/stock/quote/AAPL
```

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 175.25,
  "open": 174.50,
  "high": 176.00,
  "low": 173.80,
  "volume": 45230000,
  "timestamp": "2026-02-01T15:45:23"
}
```

**Status Codes:**
- `200` - Success
- `404` - No data found
- `500` - Server error

---

### Search Stocks

`GET /api/search`

Search for stocks by query string.

**Query Parameters:**
- `query` (string, required) - Search term (company name, ticker, etc.)
- `max_results` (integer, optional) - Maximum results to return. Default: `5`

**Example Request:**
```
GET /api/search?query=apple&max_results=5
```

**Response:**
```json
{
  "query": "apple",
  "results": {
    "quotes": [
      {
        "symbol": "AAPL",
        "shortname": "Apple Inc.",
        "exchange": "NASDAQ",
        "quoteType": "EQUITY"
      }
      // ... more results
    ]
  }
}
```

**Status Codes:**
- `200` - Success
- `400` - Missing query parameter
- `500` - Server error

---

### Get Multiple Stocks

`POST /api/stock/multiple`

Get historical data for multiple stocks at once.

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "timeframe": "1M"
}
```

**Body Parameters:**
- `symbols` (array, required) - Array of stock ticker symbols
- `timeframe` (string, optional) - Timeframe for data. Default: `1M`

**Example Request:**
```bash
POST /api/stock/multiple
Content-Type: application/json

{
  "symbols": ["AAPL", "TSLA"],
  "timeframe": "1M"
}
```

**Response:**
```json
{
  "timeframe": "1M",
  "period": "1mo",
  "interval": "30m",
  "results": {
    "AAPL": {
      "success": true,
      "data": [
        {
          "Date": "2026-01-01T09:30:00",
          "Open": 150.25,
          "Close": 151.50
        }
        // ... more data
      ]
    },
    "TSLA": {
      "success": true,
      "data": [ /* ... */ ]
    }
  }
}
```

**Status Codes:**
- `200` - Success
- `400` - Missing symbols array or invalid timeframe
- `500` - Server error

---

### Get Available Timeframes

`GET /api/timeframes`

Get list of available timeframe configurations.

**Example Request:**
```
GET /api/timeframes
```

**Response:**
```json
{
  "timeframes": {
    "1D": { "period": "1d", "interval": "1m" },
    "5D": { "period": "5d", "interval": "5m" },
    "1M": { "period": "1mo", "interval": "30m" },
    "3M": { "period": "3mo", "interval": "5d" },
    "6M": { "period": "6mo", "interval": "1d" },
    "1Y": { "period": "1y", "interval": "1d" },
    "5Y": { "period": "5y", "interval": "1wk" },
    "MAX": { "period": "max", "interval": "1mo" }
  }
}
```

**Status Codes:**
- `200` - Success

---

## Portfolio Optimization

### Quantum Portfolio Optimizer

`GET /api/portfolio/optimize`

Optimize portfolio allocation using quantum computing (QAOA algorithm). 

**Prerequisites:**
- Requires a running API at `http://localhost:8080/api/assets` that returns user's portfolio assets

**Query Parameters:**
- `budget` (float, optional) - Total investment budget in dollars. Default: `10000`
- `risk_factor` (float, optional) - Risk tolerance from 0 to 1. Default: `0.3`
  - `0.0` = Maximum return (aggressive)
  - `0.5` = Balanced risk/return
  - `1.0` = Minimum risk (conservative)
- `target_assets` (integer, optional) - Target number of assets to select. Default: `half of available assets`
- `max_assets` (integer, optional) - Maximum assets to include (filters after optimization)

**Example Request:**
```
GET /api/portfolio/optimize?budget=50000&risk_factor=0.3&target_assets=3
```

**Response:**
```json
{
  "status": "success",
  "solver": "quantum (QAOA)",
  "allocation_method": "quantum_binary_equal_weight",
  "truncated": false,
  "original_symbols": ["AAPL", "TSLA", "MSFT", "GME", "AMC"],
  "valid_symbols": ["AAPL", "TSLA", "MSFT", "GME", "AMC"],
  "budget": 50000.0,
  "target_assets": 3,
  "assets_selected": 3,
  "max_assets": null,
  "risk_factor": 0.3,
  "optimal_allocation": {
    "AAPL": {
      "selected": true,
      "weight": 0.3333,
      "allocation": 16666.67,
      "percentage": 33.33
    },
    "TSLA": {
      "selected": true,
      "weight": 0.3333,
      "allocation": 16666.67,
      "percentage": 33.33
    },
    "MSFT": {
      "selected": true,
      "weight": 0.3333,
      "allocation": 16666.67,
      "percentage": 33.33
    },
    "GME": {
      "selected": false,
      "weight": 0,
      "allocation": 0,
      "percentage": 0
    },
    "AMC": {
      "selected": false,
      "weight": 0,
      "allocation": 0,
      "percentage": 0
    }
  },
  "portfolio_metrics": {
    "expected_return": 0.0123,
    "risk": 0.0456,
    "sharpe_ratio": 0.2697
  },
  "objective_value": -0.00515,
  "quantum_solutions": [
    {
      "selection": [1, 1, 1, 0, 0],
      "bitstring": "00111",
      "assets": ["AAPL", "TSLA", "MSFT"],
      "objective_value": -0.01187,
      "probability": 0.25,
      "num_assets": 3,
      "is_selected": true
    }
    // ... top 10 quantum solutions explored
  ]
}
```

**Response Fields:**
- `status` - Operation status
- `solver` - Always "quantum (QAOA)"
- `allocation_method` - Method used for weight allocation
- `truncated` - Whether symbol list was truncated to 5 assets
- `budget` - Total investment amount
- `target_assets` - Target number of assets optimizer aimed for
- `assets_selected` - Actual number of assets selected
- `risk_factor` - Risk tolerance used
- `optimal_allocation` - Recommended allocation for each asset
  - `selected` - Whether asset is included in portfolio
  - `weight` - Portfolio weight (0-1)
  - `allocation` - Dollar amount to invest
  - `percentage` - Percentage of portfolio
- `portfolio_metrics` - Risk/return characteristics
  - `expected_return` - Weekly expected return
  - `risk` - Weekly portfolio volatility
  - `sharpe_ratio` - Risk-adjusted return ratio
- `objective_value` - Optimization objective (more negative = better)
- `quantum_solutions` - Top 10 solutions explored by quantum algorithm
  - `selection` - Binary array (1=selected, 0=not selected)
  - `bitstring` - Binary representation
  - `assets` - List of selected assets
  - `objective_value` - Objective function value
  - `probability` - Quantum probability of this solution
  - `num_assets` - Number of assets in solution
  - `is_selected` - Whether this is the final selected solution

**Status Codes:**
- `200` - Success
- `400` - Invalid parameters or insufficient data
- `500` - Optimization failed
- `503` - Portfolio API unavailable

**Notes:**
- Uses 2 years of weekly historical data for optimization
- Limited to 5 assets due to quantum computer constraints
- Equal weight allocation among selected assets
- QAOA runs with 500 iterations and 3 repetitions

---

## WebSocket Streaming

### Real-Time Stock Data Stream

`WS /ws/stream`

WebSocket endpoint for real-time stock price updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:5000/ws/stream');
```

**Subscribe to Symbols:**
```json
{
  "action": "subscribe",
  "symbols": ["AAPL", "TSLA", "BTC-USD"]
}
```

**Unsubscribe:**
```json
{
  "action": "unsubscribe"
}
```

**Server Messages:**

**Subscription Confirmed:**
```json
{
  "type": "subscribed",
  "symbols": ["AAPL", "TSLA"]
}
```

**Stream Started:**
```json
{
  "type": "stream_started",
  "symbols": ["AAPL", "TSLA"],
  "interval": "5 seconds"
}
```

**Data Update (every 5 seconds):**
```json
{
  "type": "data",
  "data": {
    "AAPL": {
      "symbol": "AAPL",
      "price": 175.25,
      "open": 174.50,
      "high": 176.00,
      "low": 173.80,
      "volume": 45230000,
      "timestamp": "2026-02-01T15:45:23"
    },
    "TSLA": {
      "symbol": "TSLA",
      "price": 245.67,
      "open": 242.30,
      "high": 248.50,
      "low": 241.00,
      "volume": 52340000,
      "timestamp": "2026-02-01T15:45:23"
    }
  },
  "timestamp": "2026-02-01T15:45:23.456789"
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Invalid JSON"
}
```

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:5000/ws/stream');

ws.onopen = () => {
  // Subscribe to symbols
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['AAPL', 'TSLA']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'data') {
    console.log('Stock prices:', data.data);
  }
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
};

// To unsubscribe
ws.send(JSON.stringify({ action: 'unsubscribe' }));
```

---

## Error Handling

All endpoints return errors in the following format:

```json
{
  "error": "Error message describing what went wrong",
  "details": "Additional technical details (if available)",
  "traceback": "Python stack trace (in debug mode)",
  "hint": "Helpful suggestion to fix the error"
}
```

### Common Status Codes

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (resource doesn't exist)
- `500` - Internal Server Error
- `503` - Service Unavailable (external API down)

---

## Data Sources

- Stock data: Yahoo Finance (yfinance library)
- Quantum optimization: Qiskit Finance
- Portfolio data: External API at `localhost:8080/api/assets`

---

## Frontend Integration Examples

### React Example - Stock Chart

```jsx
import { useState, useEffect } from 'react';

function StockChart({ symbol, timeframe }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`http://localhost:5000/api/stock/history/${symbol}?timeframe=${timeframe}`)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => console.error(err));
  }, [symbol, timeframe]);

  if (loading) return <div>Loading...</div>;
  
  return (
    <div>
      <h2>{data.symbol} - {data.timeframe}</h2>
      {/* Render chart with data.data */}
    </div>
  );
}
```

### React Example - Portfolio Optimizer

```jsx
import { useState } from 'react';

function PortfolioOptimizer() {
  const [budget, setBudget] = useState(10000);
  const [riskFactor, setRiskFactor] = useState(0.3);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const optimize = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:5000/api/portfolio/optimize?budget=${budget}&risk_factor=${riskFactor}&target_assets=3`
      );
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Optimization failed:', error);
    }
    setLoading(false);
  };

  return (
    <div>
      <h2>Quantum Portfolio Optimizer</h2>
      
      <label>
        Budget: $
        <input 
          type="number" 
          value={budget} 
          onChange={(e) => setBudget(e.target.value)} 
        />
      </label>
      
      <label>
        Risk Tolerance: {riskFactor}
        <input 
          type="range" 
          min="0" 
          max="1" 
          step="0.1" 
          value={riskFactor} 
          onChange={(e) => setRiskFactor(e.target.value)} 
        />
      </label>
      
      <button onClick={optimize} disabled={loading}>
        {loading ? 'Optimizing...' : 'Optimize Portfolio'}
      </button>
      
      {result && (
        <div>
          <h3>Optimal Allocation</h3>
          {Object.entries(result.optimal_allocation).map(([symbol, allocation]) => (
            allocation.selected && (
              <div key={symbol}>
                <strong>{symbol}</strong>: ${allocation.allocation} ({allocation.percentage}%)
              </div>
            )
          ))}
          
          <h4>Portfolio Metrics</h4>
          <p>Expected Return: {(result.portfolio_metrics.expected_return * 100).toFixed(2)}%</p>
          <p>Risk: {(result.portfolio_metrics.risk * 100).toFixed(2)}%</p>
          <p>Sharpe Ratio: {result.portfolio_metrics.sharpe_ratio.toFixed(3)}</p>
        </div>
      )}
    </div>
  );
}
```

### React Example - WebSocket Price Stream

```jsx
import { useState, useEffect } from 'react';

function LivePrices({ symbols }) {
  const [prices, setPrices] = useState({});
  const [ws, setWs] = useState(null);

  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:5000/ws/stream');
    
    websocket.onopen = () => {
      websocket.send(JSON.stringify({
        action: 'subscribe',
        symbols: symbols
      }));
    };
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'data') {
        setPrices(data.data);
      }
    };
    
    setWs(websocket);
    
    return () => {
      if (websocket) {
        websocket.send(JSON.stringify({ action: 'unsubscribe' }));
        websocket.close();
      }
    };
  }, [symbols]);

  return (
    <div>
      <h3>Live Prices</h3>
      {Object.entries(prices).map(([symbol, data]) => (
        <div key={symbol}>
          <strong>{symbol}</strong>: ${data.price?.toFixed(2) || 'N/A'}
          <span style={{ color: data.price > data.open ? 'green' : 'red' }}>
            {data.price > data.open ? '▲' : '▼'}
          </span>
        </div>
      ))}
    </div>
  );
}
```

---

## Environment Setup

### Prerequisites
- Python 3.8+
- pip packages:
  - flask
  - flask-cors
  - flask-sock
  - yfinance
  - numpy
  - qiskit
  - qiskit-finance
  - qiskit-algorithms
  - qiskit-optimization
  - qiskit-aer
  - scipy
  - requests

### Installation
```bash
pip install flask flask-cors flask-sock yfinance numpy qiskit qiskit-finance qiskit-algorithms qiskit-optimization qiskit-aer scipy requests
```

### Running the Server
```bash
python app.py
```

Server will start on `http://localhost:5000`

---

## Rate Limits & Best Practices

1. **Stock Data**: Yahoo Finance has rate limits. Avoid excessive requests.
2. **Portfolio Optimization**: Quantum optimization takes 5-15 seconds. Show loading state.
3. **WebSocket**: Polls every 5 seconds. Don't create multiple connections.
4. **Caching**: Consider caching stock history data on frontend.
5. **Error Handling**: Always handle errors gracefully with user-friendly messages.

---

## Support

For issues or questions, please check the server logs for detailed error information.
