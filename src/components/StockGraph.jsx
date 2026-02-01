import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

const API_BASE = 'http://localhost:5000';

function StockGraph({ symbol }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get(`${API_BASE}/api/stock/history/${symbol}?timeframe=1M`)
      .then(res => setData(res.data.data.map(d => ({
        date: d.Date.slice(0,10),
        close: d.Close
      }))))
      .catch(console.error);
  }, [symbol]);

  return (
    <div style={{ height: 300, marginTop: '2rem' }}>
      <h3>{symbol} 1M Performance</h3>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid stroke="#ccc" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="close" stroke="#646cff" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default StockGraph;
