import React, { useEffect, useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:5000';

function NewsList({ symbol }) {
  const [news, setNews] = useState([]);

  useEffect(() => {
    axios.get(`${API_BASE}/api/search?query=${symbol}`)
      .then(res => setNews(res.data.results?.quotes?.slice(0,3) || []))
      .catch(console.error);
  }, [symbol]);

  return (
    <div style={{ marginTop: '1rem' }}>
      <h4>News & Trends</h4>
      <ul>
        {news.map((item, i) => (
          <li key={i}>{item.symbol || item.shortname || item.exchange}</li>
        ))}
      </ul>
    </div>
  );
}

export default NewsList;
