import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import axios from "axios";

// Backend base
const API_BASE = "http://127.0.0.1:5000";

function App() {
  const [status, setStatus] = useState("Checking backend...");
  const [symbol, setSymbol] = useState("AAPL");
  const [query, setQuery] = useState("Apple");

  const [quote, setQuote] = useState(null);
  const [history, setHistory] = useState(null);
  const [info, setInfo] = useState(null);
  const [searchResults, setSearchResults] = useState(null);

  const [timeframes, setTimeframes] = useState({});
  const [timeframe, setTimeframe] = useState("1M");

  const [multiSymbols, setMultiSymbols] = useState("AAPL,MSFT,TSLA");
  const [multiData, setMultiData] = useState(null);

  const [wsConnected, setWsConnected] = useState(false);
  const [wsData, setWsData] = useState(null);
  const [wsLog, setWsLog] = useState([]);

  // Check backend health
  async function checkHealth() {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      setStatus(`✅ Backend OK (${res.data.status})`);
    } catch (err) {
      setStatus("❌ Backend not running. Start backend first.");
    }
  }

  // Get available timeframes
  async function loadTimeframes() {
    try {
      const res = await axios.get(`${API_BASE}/api/timeframes`);
      setTimeframes(res.data.timeframes || {});
    } catch (err) {
      console.log("Timeframes error:", err);
    }
  }

  // Quote
  async function fetchQuote() {
    try {
      const res = await axios.get(`${API_BASE}/api/stock/quote/${symbol}`);
      setQuote(res.data);
    } catch (err) {
      alert("Quote error: " + (err.response?.data?.error || err.message));
    }
  }

  // History
  async function fetchHistory() {
    try {
      const res = await axios.get(
        `${API_BASE}/api/stock/history/${symbol}?timeframe=${timeframe}`
      );
      setHistory(res.data);
    } catch (err) {
      alert("History error: " + (err.response?.data?.error || err.message));
    }
  }

  // Info
  async function fetchInfo() {
    try {
      const res = await axios.get(`${API_BASE}/api/stock/info/${symbol}`);
      setInfo(res.data);
    } catch (err) {
      alert("Info error: " + (err.response?.data?.error || err.message));
    }
  }

  // Search
  async function fetchSearch() {
    try {
      const res = await axios.get(
        `${API_BASE}/api/search?query=${encodeURIComponent(query)}&max_results=10`
      );
      setSearchResults(res.data);
    } catch (err) {
      alert("Search error: " + (err.response?.data?.error || err.message));
    }
  }

  // Multi stocks
  async function fetchMultiple() {
    try {
      const symbolsArray = multiSymbols
        .split(",")
        .map((s) => s.trim().toUpperCase())
        .filter(Boolean);

      const res = await axios.post(`${API_BASE}/api/stock/multiple`, {
        symbols: symbolsArray,
        timeframe: timeframe,
      });

      setMultiData(res.data);
    } catch (err) {
      alert("Multiple error: " + (err.response?.data?.error || err.message));
    }
  }

  // WebSocket streaming
  function startWebSocket() {
    const ws = new WebSocket("ws://127.0.0.1:5000/ws/stream");

    ws.onopen = () => {
      setWsConnected(true);
      setWsLog((prev) => [...prev, "✅ WebSocket Connected"]);

      ws.send(
        JSON.stringify({
          action: "subscribe",
          symbols: [symbol],
        })
      );
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        setWsData(msg);

        if (msg.type === "data") {
          setWsLog((prev) => [
            ...prev.slice(-15),
            `📡 Live: ${JSON.stringify(msg.data)}`,
          ]);
        }
      } catch (e) {
        setWsLog((prev) => [...prev, "❌ Invalid WS message"]);
      }
    };

    ws.onerror = () => {
      setWsConnected(false);
      setWsLog((prev) => [...prev, "❌ WebSocket Error"]);
    };

    ws.onclose = () => {
      setWsConnected(false);
      setWsLog((prev) => [...prev, "🔌 WebSocket Closed"]);
    };
  }

  useEffect(() => {
    checkHealth();
    loadTimeframes();
  }, []);

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <h1 style={{ margin: 0 }}>📊 BigBull Screener MBC</h1>
        <p style={{ margin: 0, color: "#666" }}>{status}</p>
      </div>

      {/* TOP BAR */}
      <div style={styles.topBar}>
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Search Stocks</h2>
          <input
            style={styles.input}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search e.g. Apple, Tesla"
          />
          <button style={styles.btn} onClick={fetchSearch}>
            Search
          </button>

          {searchResults && (
            <pre style={styles.pre}>
              {JSON.stringify(searchResults, null, 2)}
            </pre>
          )}
        </div>

        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Select Symbol</h2>

          <input
            style={styles.input}
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter symbol e.g. AAPL"
          />

          <label style={styles.label}>Timeframe</label>
          <select
            style={styles.input}
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
          >
            {Object.keys(timeframes).length > 0 ? (
              Object.keys(timeframes).map((tf) => (
                <option key={tf} value={tf}>
                  {tf}
                </option>
              ))
            ) : (
              <>
                <option value="1D">1D</option>
                <option value="5D">5D</option>
                <option value="1M">1M</option>
                <option value="3M">3M</option>
                <option value="6M">6M</option>
                <option value="1Y">1Y</option>
                <option value="5Y">5Y</option>
                <option value="MAX">MAX</option>
              </>
            )}
          </select>

          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
            <button style={styles.btn} onClick={fetchQuote}>
              Get Quote
            </button>
            <button style={styles.btn} onClick={fetchHistory}>
              Get History
            </button>
            <button style={styles.btn} onClick={fetchInfo}>
              Get Info
            </button>
          </div>
        </div>
      </div>

      {/* OUTPUT GRID */}
      <div style={styles.grid}>
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>📌 Quote</h2>
          {quote ? (
            <pre style={styles.pre}>{JSON.stringify(quote, null, 2)}</pre>
          ) : (
            <p style={styles.muted}>Click "Get Quote"</p>
          )}
        </div>

        <div style={styles.card}>
          <h2 style={styles.cardTitle}>📈 History</h2>
          {history ? (
            <pre style={styles.pre}>
              {JSON.stringify(
                {
                  symbol: history.symbol,
                  timeframe: history.timeframe,
                  period: history.period,
                  interval: history.interval,
                  sample_rows: history.data?.slice(0, 5),
                  total_rows: history.data?.length,
                },
                null,
                2
              )}
            </pre>
          ) : (
            <p style={styles.muted}>Click "Get History"</p>
          )}
        </div>

        <div style={styles.card}>
          <h2 style={styles.cardTitle}>🏢 Company Info</h2>
          {info ? (
            <pre style={styles.pre}>{JSON.stringify(info, null, 2)}</pre>
          ) : (
            <p style={styles.muted}>Click "Get Info"</p>
          )}
        </div>
      </div>

      {/* MULTI + WS */}
      <div style={styles.grid}>
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>📦 Multi Stock Fetch</h2>
          <input
            style={styles.input}
            value={multiSymbols}
            onChange={(e) => setMultiSymbols(e.target.value)}
            placeholder="AAPL,MSFT,TSLA"
          />
          <button style={styles.btn} onClick={fetchMultiple}>
            Fetch Multiple
          </button>

          {multiData && (
            <pre style={styles.pre}>{JSON.stringify(multiData, null, 2)}</pre>
          )}
        </div>

        <div style={styles.card}>
          <h2 style={styles.cardTitle}>⚡ Live Streaming (WebSocket)</h2>

          <button style={styles.btn} onClick={startWebSocket}>
            {wsConnected ? "Connected" : "Start WebSocket"}
          </button>

          <p style={styles.muted}>
            Endpoint used: <b>ws://127.0.0.1:5000/ws/stream</b>
          </p>

          {wsData && (
            <pre style={styles.pre}>{JSON.stringify(wsData, null, 2)}</pre>
          )}

          <h3 style={{ marginTop: 15 }}>Logs</h3>
          <pre style={styles.pre}>{wsLog.join("\n")}</pre>
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    fontFamily: "Arial, sans-serif",
    padding: "20px",
    background: "#f4f6f9",
    minHeight: "100vh",
  },
  header: {
    marginBottom: "20px",
    padding: "15px",
    borderRadius: "12px",
    background: "white",
    boxShadow: "0 2px 6px rgba(0,0,0,0.08)",
  },
  topBar: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "15px",
    marginBottom: "15px",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: "15px",
    marginBottom: "15px",
  },
  card: {
    background: "white",
    borderRadius: "12px",
    padding: "15px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.08)",
    overflow: "hidden",
  },
  cardTitle: {
    marginTop: 0,
    marginBottom: "10px",
    fontSize: "18px",
  },
  input: {
    width: "100%",
    padding: "10px",
    borderRadius: "8px",
    border: "1px solid #ccc",
    marginBottom: "10px",
    fontSize: "14px",
  },
  btn: {
    padding: "10px 14px",
    borderRadius: "8px",
    border: "none",
    cursor: "pointer",
    background: "#0b5ed7",
    color: "white",
    fontWeight: "bold",
  },
  pre: {
    background: "#111",
    color: "#0f0",
    padding: "10px",
    borderRadius: "8px",
    overflowX: "auto",
    fontSize: "12px",
    maxHeight: "300px",
  },
  label: {
    fontSize: "13px",
    fontWeight: "bold",
    color: "#444",
  },
  muted: {
    color: "#666",
  },
};

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
