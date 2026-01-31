# BigBull Stock Market API

Flask REST API with WebSocket support for real-time stock market data using yfinance.

## Features

- Historical stock data with predefined timeframes
- Real-time stock data streaming via WebSocket
- Stock search with news
- Detailed company information
- Batch stock data retrieval
- CORS enabled for frontend integration

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

Server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```http
GET /health
```

### Get Stock History
```http
GET /api/stock/history/<symbol>?timeframe=1M
```

**Query Parameters:**
- `timeframe`: 1D, 5D, 1M, 3M, 6M, 1Y, 5Y, MAX (default: 1M)
- `period`: Custom period (overrides timeframe)
- `interval`: Custom interval (overrides timeframe)

**Example:**
```bash
curl http://localhost:5000/api/stock/history/AAPL?timeframe=1M
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
      "Open": 150.5,
      "High": 152.0,
      "Low": 150.0,
      "Close": 151.5,
      "Volume": 1000000
    }
  ]
}
```

### Get Stock Info
```http
GET /api/stock/info/<symbol>
```

**Example:**
```bash
curl http://localhost:5000/api/stock/info/AAPL
```

### Search Stocks
```http
GET /api/search?query=Apple&max_results=5&news_count=3
```

**Query Parameters:**
- `query` (required): Search term
- `max_results` (optional): Maximum results (default: 5)
- `news_count` (optional): Number of news items (default: 3)

**Example:**
```bash
curl "http://localhost:5000/api/search?query=Apple&max_results=10"
```

### Get Stock Quote
```http
GET /api/stock/quote/<symbol>
```

**Example:**
```bash
curl http://localhost:5000/api/stock/quote/AAPL
```

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 151.5,
  "open": 150.0,
  "high": 152.0,
  "low": 149.5,
  "volume": 1000000,
  "timestamp": "2026-01-31T15:30:00"
}
```

### Get Multiple Stocks
```http
POST /api/stock/multiple
Content-Type: application/json

{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "timeframe": "1M"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/stock/multiple \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"], "timeframe": "1M"}'
```

### Get Available Timeframes
```http
GET /api/timeframes
```

## WebSocket Endpoint

### Real-time Stock Streaming
```
WS ws://localhost:5000/ws/stream
```

**Connect and Subscribe:**
```javascript
const ws = new WebSocket('ws://localhost:5000/ws/stream');

ws.onopen = () => {
  // Subscribe to symbols
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['AAPL', 'BTC-USD']
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};

// Unsubscribe
ws.send(JSON.stringify({
  action: 'unsubscribe'
}));
```

**Message Types:**
- `subscribed`: Confirmation of subscription
- `data`: Real-time stock data
- `error`: Error message
- `unsubscribed`: Confirmation of unsubscription

## Timeframe Reference

| Button | Period | Interval | Use Case      |
|--------|--------|----------|---------------|
| 1D     | 1d     | 1m       | Intraday      |
| 5D     | 5d     | 5m       | Short trend   |
| 1M     | 1mo    | 30m      | Monthly       |
| 3M     | 3mo    | 1d       | Medium term   |
| 6M     | 6mo    | 1d       | Medium term   |
| 1Y     | 1y     | 1d       | Long term     |
| 5Y     | 5y     | 1wk      | Very long     |
| MAX    | max    | 1mo      | Full history  |

## Project Structure

```
BigBull/
├── app.py              # Main Flask application
├── test.py             # WebSocket streaming example
├── test2.py            # Search functionality example
├── test3.py            # History retrieval example
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Testing Examples

### Test Historical Data
```bash
# Get 1 month of data
curl http://localhost:5000/api/stock/history/AAPL?timeframe=1M

# Get intraday data
curl http://localhost:5000/api/stock/history/AAPL?timeframe=1D

# Custom period and interval
curl http://localhost:5000/api/stock/history/AAPL?period=2mo&interval=1h
```

### Test Search
```bash
curl "http://localhost:5000/api/search?query=Tesla&max_results=5"
```

### Test Multiple Stocks
```bash
curl -X POST http://localhost:5000/api/stock/multiple \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"],
    "timeframe": "1M"
  }'
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error

Error responses include a descriptive message:
```json
{
  "error": "No data found for symbol INVALID"
}
```

## Notes

- WebSocket connections are managed per client
- Real-time data requires active market hours
- Rate limiting may apply based on yfinance API limits
- All timestamps are in ISO 8601 format
