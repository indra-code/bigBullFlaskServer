import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import axios from "axios";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";
import {
  Star,
  StarOff,
  RefreshCw,
  Search,
  Activity,
  Newspaper,
  Table2,
  LineChart as LineChartIcon,
} from "lucide-react";

// Backend base
const API_BASE = "http://127.0.0.1:5000";

function formatMoney(n) {
  if (n === null || n === undefined) return "-";
  const num = Number(n);
  if (Number.isNaN(num)) return String(n);
  return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function pctChange(open, price) {
  const o = Number(open);
  const p = Number(price);
  if (!o || !p || Number.isNaN(o) || Number.isNaN(p)) return null;
  return ((p - o) / o) * 100;
}

function safeUpper(s) {
  return (s || "").trim().toUpperCase();
}

function App() {
  const [status, setStatus] = useState("Checking backend...");
  const [symbol, setSymbol] = useState("AAPL");
  const [query, setQuery] = useState("Apple");

  const [activeTab, setActiveTab] = useState("Chart");

  const [quote, setQuote] = useState(null);
  const [history, setHistory] = useState(null);
  const [info, setInfo] = useState(null);
  const [searchResults, setSearchResults] = useState(null);

  const [timeframes, setTimeframes] = useState({});
  const [timeframe, setTimeframe] = useState("1M");

  const [multiSymbols, setMultiSymbols] = useState("AAPL,MSFT,TSLA,NVDA,BTC-USD");
  const [multiData, setMultiData] = useState(null);

  const [wsConnected, setWsConnected] = useState(false);
  const [wsData, setWsData] = useState(null);
  const [wsLog, setWsLog] = useState([]);
  const [ws, setWs] = useState(null);

  // Watchlist
  const [watchlist, setWatchlist] = useState(() => {
    try {
      const saved = localStorage.getItem("watchlist");
      return saved ? JSON.parse(saved) : ["AAPL", "MSFT", "TSLA"];
    } catch {
      return ["AAPL", "MSFT", "TSLA"];
    }
  });

  // Auto refresh
  const [autoRefresh, setAutoRefresh] = useState(false);
  const refreshTimerRef = useRef(null);

  // Sorting screener table
  const [sortKey, setSortKey] = useState("symbol");
  const [sortDir, setSortDir] = useState("asc");

  // ================= API =================
  async function checkHealth() {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      setStatus(` (${res.data.status})`);
    } catch {
      setStatus("❌ Backend not running. Start backend first.");
    }
  }

  async function loadTimeframes() {
    try {
      const res = await axios.get(`${API_BASE}/api/timeframes`);
      setTimeframes(res.data.timeframes || {});
    } catch (err) {
      console.log("Timeframes error:", err);
    }
  }

  async function fetchQuote(sym = symbol) {
    try {
      const res = await axios.get(`${API_BASE}/api/stock/quote/${safeUpper(sym)}`);
      setQuote(res.data);
    } catch (err) {
      alert("Quote error: " + (err.response?.data?.error || err.message));
    }
  }

  async function fetchHistory(sym = symbol, tf = timeframe) {
    try {
      const res = await axios.get(
        `${API_BASE}/api/stock/history/${safeUpper(sym)}?timeframe=${tf}`
      );
      setHistory(res.data);
      setActiveTab("Chart");
    } catch (err) {
      alert("History error: " + (err.response?.data?.error || err.message));
    }
  }

  async function fetchInfo(sym = symbol) {
    try {
      const res = await axios.get(`${API_BASE}/api/stock/info/${safeUpper(sym)}`);
      setInfo(res.data);
      setActiveTab("Info");
    } catch (err) {
      alert("Info error: " + (err.response?.data?.error || err.message));
    }
  }

  async function fetchSearch() {
    try {
      const res = await axios.get(
        `${API_BASE}/api/search?query=${encodeURIComponent(query)}&max_results=10&news_count=6`
      );
      setSearchResults(res.data);
      setActiveTab("News");
    } catch (err) {
      alert("Search error: " + (err.response?.data?.error || err.message));
    }
  }

  async function fetchMultiple(symbolList) {
    try {
      const symbolsArray = symbolList
        ? symbolList
        : multiSymbols
            .split(",")
            .map((s) => safeUpper(s))
            .filter(Boolean);

      const res = await axios.post(`${API_BASE}/api/stock/multiple`, {
        symbols: symbolsArray,
        timeframe: timeframe,
      });

      setMultiData(res.data);
      setActiveTab("Screener");
    } catch (err) {
      alert("Multiple error: " + (err.response?.data?.error || err.message));
    }
  }

  // ================= WebSocket =================
  function startWebSocket() {
    if (wsConnected && ws) return;

    const socket = new WebSocket("ws://127.0.0.1:5000/ws/stream");
    setWs(socket);

    socket.onopen = () => {
      setWsConnected(true);
      setWsLog((prev) => [...prev, "✅ WebSocket Connected"]);

      socket.send(
        JSON.stringify({
          action: "subscribe",
          symbols: [safeUpper(symbol)],
        })
      );
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        setWsData(msg);

        if (msg.type === "data") {
          setWsLog((prev) => [
            ...prev.slice(-10),
            `📡 ${msg.data?.symbol} → ${msg.data?.price}`,
          ]);
        }
      } catch {
        setWsLog((prev) => [...prev, "❌ Invalid WS message"]);
      }
    };

    socket.onerror = () => {
      setWsConnected(false);
      setWsLog((prev) => [...prev, "❌ WebSocket Error"]);
    };

    socket.onclose = () => {
      setWsConnected(false);
      setWsLog((prev) => [...prev, "🔌 WebSocket Closed"]);
    };
  }

  function stopWebSocket() {
    if (ws) {
      ws.close();
      setWs(null);
    }
    setWsConnected(false);
  }

  // When symbol changes while WS ON, re-subscribe by restarting
  useEffect(() => {
    if (!wsConnected) return;
    stopWebSocket();
    startWebSocket();
  }, [symbol]);

  // ================= Watchlist =================
  function addToWatchlist(sym) {
    const s = safeUpper(sym);
    if (!s) return;
    setWatchlist((prev) => {
      if (prev.includes(s)) return prev;
      const next = [s, ...prev];
      localStorage.setItem("watchlist", JSON.stringify(next));
      return next;
    });
  }

  function removeFromWatchlist(sym) {
    const s = safeUpper(sym);
    setWatchlist((prev) => {
      const next = prev.filter((x) => x !== s);
      localStorage.setItem("watchlist", JSON.stringify(next));
      return next;
    });
  }

  // ================= Auto Refresh =================
  useEffect(() => {
    if (autoRefresh) {
      refreshTimerRef.current = setInterval(() => {
        fetchQuote(symbol);
      }, 5000);
    } else {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current);
    }
    return () => {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current);
    };
  }, [autoRefresh, symbol]);

  // ================= Chart Data =================
  const chartData = useMemo(() => {
    if (!history?.data) return [];
    return history.data.map((row) => ({
      time: row.Date,
      close: row.Close,
      open: row.Open,
      high: row.High,
      low: row.Low,
      volume: row.Volume,
    }));
  }, [history]);

  // ================= Screener Rows =================
  const screenerRows = useMemo(() => {
    const d = multiData?.data;
    if (!d) return [];

    const rows = Object.keys(d).map((sym) => {
      const r = d[sym] || {};
      const change = pctChange(r.open, r.price);
      return {
        symbol: sym,
        price: r.price,
        open: r.open,
        high: r.high,
        low: r.low,
        volume: r.volume,
        changePct: change,
      };
    });

    const sorted = [...rows].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];

      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;

      if (typeof av === "string") {
        return sortDir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
      }

      return sortDir === "asc" ? av - bv : bv - av;
    });

    return sorted;
  }, [multiData, sortKey, sortDir]);

  const topGainers = useMemo(() => {
    return [...screenerRows]
      .filter((r) => r.changePct !== null)
      .sort((a, b) => b.changePct - a.changePct)
      .slice(0, 5);
  }, [screenerRows]);

  const topLosers = useMemo(() => {
    return [...screenerRows]
      .filter((r) => r.changePct !== null)
      .sort((a, b) => a.changePct - b.changePct)
      .slice(0, 5);
  }, [screenerRows]);

  function loadSymbol(sym) {
    const s = safeUpper(sym);
    setSymbol(s);
    fetchQuote(s);
    fetchHistory(s, timeframe);
  }

  // Initial load
  useEffect(() => {
    checkHealth();
    loadTimeframes();
    fetchQuote(symbol);
    fetchHistory(symbol, timeframe);
  }, []);

  // ================= UI =================
  return (
    <div style={styles.page}>
      {/* HEADER */}
      <div style={styles.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={styles.logoBox}>📊</div>
          <div>
            <h1 style={{ margin: 0, fontSize: 20 }}>BigBull </h1>
            <div style={{ fontSize: 12, color: "#9ca3af" }}>{status}</div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button
            style={autoRefresh ? styles.btnSuccess : styles.btnOutline}
            onClick={() => setAutoRefresh((v) => !v)}
            title="Auto refresh quote every 5s"
          >
            <RefreshCw size={16} style={{ marginRight: 8 }} />
            Auto Refresh
          </button>

          <button
            style={wsConnected ? styles.btnSuccess : styles.btnOutline}
            onClick={wsConnected ? stopWebSocket : startWebSocket}
            title="Live streaming WebSocket"
          >
            <Activity size={16} style={{ marginRight: 8 }} />
            {wsConnected ? "Live ON" : "Live OFF"}
          </button>
        </div>
      </div>

      {/* TOP CONTROLS */}
      <div style={styles.controlsGrid}>
        {/* SEARCH */}
        <div style={styles.panel}>
          <div style={styles.panelTitle}>
            <Search size={16} style={{ marginRight: 8 }} />
            Search + News
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <input
              style={styles.input}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Apple, Tesla, Microsoft..."
            />
            <button style={styles.btn} onClick={fetchSearch}>
              Search
            </button>
          </div>
        </div>

        {/* SYMBOL + ACTIONS */}
        <div style={styles.panel}>
          <div style={styles.panelTitle}>
            <LineChartIcon size={16} style={{ marginRight: 8 }} />
            Chart Controls
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <input
              style={styles.input}
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
            />

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
          </div>

          <div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
            <button style={styles.btn} onClick={() => loadSymbol(symbol)}>
              Load
            </button>

            <button style={styles.btnOutline} onClick={() => addToWatchlist(symbol)}>
              <Star size={16} style={{ marginRight: 8 }} />
              Watch
            </button>

            <button style={styles.btnOutline} onClick={() => fetchInfo(symbol)}>
              Info
            </button>
          </div>
        </div>

        {/* SCREENER */}
        <div style={styles.panel}>
          <div style={styles.panelTitle}>
            <Table2 size={16} style={{ marginRight: 8 }} />
            Screener
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <input
              style={styles.input}
              value={multiSymbols}
              onChange={(e) => setMultiSymbols(e.target.value)}
              placeholder="AAPL,MSFT,TSLA"
            />
            <button style={styles.btn} onClick={() => fetchMultiple()}>
              Run
            </button>
          </div>
        </div>
      </div>

      {/* MAIN GRID */}
      <div style={styles.mainGrid}>
        {/* LEFT: MAIN PANEL */}
        <div style={styles.bigPanel}>
          {/* Tabs */}
          <div style={styles.tabs}>
            {[
              { name: "Chart", icon: <LineChartIcon size={16} /> },
              { name: "Screener", icon: <Table2 size={16} /> },
              { name: "News", icon: <Newspaper size={16} /> },
              { name: "Info", icon: <Activity size={16} /> },
            ].map((t) => (
              <button
                key={t.name}
                onClick={() => setActiveTab(t.name)}
                style={activeTab === t.name ? styles.tabActive : styles.tab}
              >
                <span style={{ marginRight: 8, display: "inline-flex" }}>{t.icon}</span>
                {t.name}
              </button>
            ))}
          </div>

          {/* Content */}
          <div style={{ padding: 14 }}>
            {/* CHART */}
            {activeTab === "Chart" && (
              <>
                <div style={styles.sectionTitle}>📊 Price Trend ({symbol})</div>

                {chartData.length > 0 ? (
                  <>
                    <div style={{ height: 280 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="time" hide />
                          <YAxis domain={["auto", "auto"]} />
                          <Tooltip />
                          <Line type="monotone" dataKey="close" dot={false} strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div style={{ height: 160, marginTop: 12 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="time" hide />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="volume" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                ) : (
                  <div style={styles.emptyBox}>Click Load to fetch history</div>
                )}
              </>
            )}

            {/* SCREENER */}
            {activeTab === "Screener" && (
              <>
                <div style={styles.sectionTitle}>📋 Screener Table (sortable)</div>

                {screenerRows.length > 0 ? (
                  <>
                    {/* Gainers/Losers */}
                    <div style={styles.dualBox}>
                      <div style={styles.smallCard}>
                        <div style={styles.smallTitle}>📈 Top Gainers</div>
                        {topGainers.map((r) => (
                          <div key={r.symbol} style={styles.miniRow}>
                            <button style={styles.linkBtn} onClick={() => loadSymbol(r.symbol)}>
                              {r.symbol}
                            </button>
                            <span style={{ fontWeight: 700 }}>
                              {r.changePct?.toFixed(2)}%
                            </span>
                          </div>
                        ))}
                      </div>

                      <div style={styles.smallCard}>
                        <div style={styles.smallTitle}>📉 Top Losers</div>
                        {topLosers.map((r) => (
                          <div key={r.symbol} style={styles.miniRow}>
                            <button style={styles.linkBtn} onClick={() => loadSymbol(r.symbol)}>
                              {r.symbol}
                            </button>
                            <span style={{ fontWeight: 700 }}>
                              {r.changePct?.toFixed(2)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div style={{ overflowX: "auto", marginTop: 12 }}>
                      <table style={styles.table}>
                        <thead>
                          <tr>
                            {[
                              ["symbol", "Symbol"],
                              ["price", "Price"],
                              ["open", "Open"],
                              ["high", "High"],
                              ["low", "Low"],
                              ["volume", "Volume"],
                              ["changePct", "Change %"],
                            ].map(([key, label]) => (
                              <th
                                key={key}
                                style={styles.th}
                                onClick={() => {
                                  if (sortKey === key) {
                                    setSortDir((d) => (d === "asc" ? "desc" : "asc"));
                                  } else {
                                    setSortKey(key);
                                    setSortDir("asc");
                                  }
                                }}
                              >
                                {label}{" "}
                                {sortKey === key ? (sortDir === "asc" ? "▲" : "▼") : ""}
                              </th>
                            ))}
                          </tr>
                        </thead>

                        <tbody>
                          {screenerRows.map((r) => (
                            <tr key={r.symbol}>
                              <td style={styles.tdSymbol}>
                                <button style={styles.linkBtn} onClick={() => loadSymbol(r.symbol)}>
                                  {r.symbol}
                                </button>
                              </td>
                              <td style={styles.td}>{formatMoney(r.price)}</td>
                              <td style={styles.td}>{formatMoney(r.open)}</td>
                              <td style={styles.td}>{formatMoney(r.high)}</td>
                              <td style={styles.td}>{formatMoney(r.low)}</td>
                              <td style={styles.td}>{formatMoney(r.volume)}</td>
                              <td style={styles.td}>
                                {r.changePct !== null ? `${r.changePct.toFixed(2)}%` : "-"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                ) : (
                  <div style={styles.emptyBox}>Run screener using Run button</div>
                )}
              </>
            )}

            {/* NEWS */}
            {activeTab === "News" && (
              <>
                <div style={styles.sectionTitle}>📰 News + Search Results</div>

                {searchResults ? (
                  <pre style={styles.pre}>{JSON.stringify(searchResults, null, 2)}</pre>
                ) : (
                  <div style={styles.emptyBox}>Search “Tesla” to load news</div>
                )}
              </>
            )}

            {/* INFO */}
            {activeTab === "Info" && (
              <>
                <div style={styles.sectionTitle}>🏢 Company Info</div>
                {info ? (
                  <pre style={styles.pre}>{JSON.stringify(info, null, 2)}</pre>
                ) : (
                  <div style={styles.emptyBox}>Click Info button to load</div>
                )}
              </>
            )}
          </div>
        </div>

        {/* RIGHT: WATCHLIST + LIVE */}
        <div style={styles.sidePanel}>
          <div style={styles.sectionTitle}>⭐ Watchlist</div>

          {watchlist.length === 0 ? (
            <div style={styles.emptyBox}>Add symbols using Watch button</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {watchlist.map((sym) => (
                <div key={sym} style={styles.watchRow}>
                  <button style={styles.linkBtn} onClick={() => loadSymbol(sym)}>
                    {sym}
                  </button>
                  <button
                    style={styles.iconBtn}
                    onClick={() => removeFromWatchlist(sym)}
                    title="Remove"
                  >
                    <StarOff size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div style={{ marginTop: 16 }}>
            <div style={styles.sectionTitle}>⚡ Live Quote</div>

            <div style={styles.statBox}>
              <div style={styles.statRow}>
                <span style={styles.statLabel}>Symbol</span>
                <b>{quote?.symbol || symbol}</b>
              </div>

              <div style={styles.statRow}>
                <span style={styles.statLabel}>Price</span>
                <b style={{ fontSize: 18 }}>
                  {quote?.price ? formatMoney(quote.price) : "-"}
                </b>
              </div>

              <div style={styles.statRow}>
                <span style={styles.statLabel}>Open</span>
                <b>{formatMoney(quote?.open)}</b>
              </div>

              <div style={styles.statRow}>
                <span style={styles.statLabel}>High</span>
                <b>{formatMoney(quote?.high)}</b>
              </div>

              <div style={styles.statRow}>
                <span style={styles.statLabel}>Low</span>
                <b>{formatMoney(quote?.low)}</b>
              </div>

              <div style={styles.statRow}>
                <span style={styles.statLabel}>Volume</span>
                <b>{formatMoney(quote?.volume)}</b>
              </div>
            </div>
          </div>

          <div style={{ marginTop: 14 }}>
            <div style={styles.sectionTitle}>📡 WebSocket Logs</div>
            <pre style={styles.preSmall}>{wsLog.join("\n")}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    padding: 18,
    background: "#0b1220",
    color: "#e5e7eb",
    fontFamily: "Inter, system-ui, Arial",
  },

  header: {
    background: "#111827",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: 14,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 14,
  },

  logoBox: {
    width: 40,
    height: 40,
    borderRadius: 12,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "rgba(59,130,246,0.15)",
    border: "1px solid rgba(59,130,246,0.25)",
  },

  controlsGrid: {
    display: "grid",
    gridTemplateColumns: "1.2fr 1.5fr 1.3fr",
    gap: 12,
    marginBottom: 12,
  },

  panel: {
    background: "#111827",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: 14,
  },

  panelTitle: {
    display: "flex",
    alignItems: "center",
    fontSize: 13,
    color: "#9ca3af",
    marginBottom: 10,
    fontWeight: 600,
  },

  input: {
    width: "100%",
    padding: "10px 12px",
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.04)",
    color: "#e5e7eb",
    outline: "none",
    fontSize: 14,
  },

  btn: {
    padding: "10px 14px",
    borderRadius: 12,
    border: "1px solid rgba(59,130,246,0.5)",
    background: "rgba(59,130,246,0.18)",
    color: "#e5e7eb",
    cursor: "pointer",
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
  },

  btnOutline: {
    padding: "10px 14px",
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.18)",
    background: "rgba(255,255,255,0.06)",
    color: "#e5e7eb",
    cursor: "pointer",
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
  },

  btnSuccess: {
    padding: "10px 14px",
    borderRadius: 12,
    border: "1px solid rgba(34,197,94,0.5)",
    background: "rgba(34,197,94,0.18)",
    color: "#e5e7eb",
    cursor: "pointer",
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
  },

  mainGrid: {
    display: "grid",
    gridTemplateColumns: "2.2fr 1fr",
    gap: 12,
  },

  bigPanel: {
    background: "#111827",
    border: "1px solid rgba(235, 231, 231, 0.08)",
    borderRadius: 16,
    overflow: "hidden",
  },

  sidePanel: {
    background: "#111827",
    border: "1px solid rgba(231, 223, 223, 0.85)",
    borderRadius: 16,
    padding: 14,
  },

  tabs: {
    display: "flex",
    gap: 8,
    padding: 12,
    borderBottom: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(255,255,255,0.02)",
  },

  tab: {
    padding: "8px 12px",
    borderRadius: 999,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.04)",
    color: "#e5e7eb",
    cursor: "pointer",
    fontWeight: 700,
    fontSize: 13,
    display: "flex",
    alignItems: "center",
  },

  tabActive: {
    padding: "8px 12px",
    borderRadius: 999,
    border: "1px solid rgba(59,130,246,0.5)",
    background: "rgba(59,130,246,0.22)",
    color: "#e5e7eb",
    cursor: "pointer",
    fontWeight: 800,
    fontSize: 13,
    display: "flex",
    alignItems: "center",
  },

  sectionTitle: {
    fontSize: 13,
    color: "#9ca3af",
    marginBottom: 10,
    fontWeight: 700,
  },

  emptyBox: {
    padding: 14,
    borderRadius: 14,
    background: "rgba(255,255,255,0.04)",
    border: "1px dashed rgba(255,255,255,0.18)",
    color: "#cbd5e1",
  },

  pre: {
    background: "#0b1220",
    border: "1px solid rgba(255,255,255,0.08)",
    color: "#a7f3d0",
    padding: 12,
    borderRadius: 14,
    overflowX: "auto",
    maxHeight: 420,
    fontSize: 12,
  },

  preSmall: {
    background: "#0b1220",
    border: "1px solid rgba(255,255,255,0.08)",
    color: "#93c5fd",
    padding: 12,
    borderRadius: 14,
    overflowX: "auto",
    maxHeight: 220,
    fontSize: 12,
  },

  statBox: {
    borderRadius: 16,
    padding: 14,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
  },

  statRow: {
    display: "flex",
    justifyContent: "space-between",
    padding: "8px 0",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  },

  statLabel: {
    color: "#9ca3af",
    fontSize: 13,
  },

  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 14,
  },

  th: {
    textAlign: "left",
    padding: "10px 10px",
    borderBottom: "1px solid rgba(255,255,255,0.12)",
    color: "#9ca3af",
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    cursor: "pointer",
  },

  td: {
    padding: "10px 10px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  },

  tdSymbol: {
    padding: "10px 10px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
    fontWeight: 800,
  },

  linkBtn: {
    background: "transparent",
    border: "none",
    color: "#93c5fd",
    cursor: "pointer",
    fontWeight: 800,
    padding: 0,
  },

  dualBox: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 12,
  },

  smallCard: {
    padding: 12,
    borderRadius: 14,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
  },

  smallTitle: {
    fontSize: 12,
    color: "#9ca3af",
    fontWeight: 800,
    marginBottom: 8,
  },

  miniRow: {
    display: "flex",
    justifyContent: "space-between",
    padding: "6px 0",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  },

  watchRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px 12px",
    borderRadius: 14,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
  },

  iconBtn: {
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.04)",
    color: "#e5e7eb",
    borderRadius: 12,
    padding: 8,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
