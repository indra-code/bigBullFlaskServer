import React, { useEffect, useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:5000';

function RiskScore({ symbol }) {
  const [score, setScore] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/api/stock/history/${symbol}?timeframe=1Y`)
      .then(res => {
        const prices = res.data.data.map(d => d.Close);
        const volatility = Math.std ? Math.std(prices) : (Math.max(...prices) - Math.min(...prices)) / prices.length;
        setScore(Math.round(volatility * 100));
      })
      .catch(() => setScore('N/A'));
  }, [symbol]);

  return (
    <div style={{ marginTop: '1rem' }}>
      <h4>Risk Score</h4>
      <p>{score}</p>
    </div>
  );
}

export default RiskScore;
