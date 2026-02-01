import React, { useEffect, useState } from 'react';
import axios from 'axios';
import StockCard from './StockCard';
import StockGraph from './StockGraph';
import NewsList from './NewsList';
import BuySellForm from './BuySellForm';
import RiskScore from './RiskScore';

const API_BASE = 'http://localhost:5000';

function Portfolio() {
  const [portfolio, setPortfolio] = useState(['AAPL', 'TSLA']);
  const [stockData, setStockData] = useState({});
  const [selected, setSelected] = useState(portfolio[0]);

  useEffect(() => {
    portfolio.forEach(symbol => {
      axios.get(`${API_BASE}/api/stock/quote/${symbol}`)
        .then(res => setStockData(prev => ({ ...prev, [symbol]: res.data })))
        .catch(console.error);
    });
  }, [portfolio]);

  const removeStock = (symbol) => {
    setPortfolio(prev => prev.filter(s => s !== symbol));
  };

  const addStock = (symbol) => {
    if (!portfolio.includes(symbol)) setPortfolio([...portfolio, symbol]);
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        {portfolio.map(sym => (
          <StockCard
            key={sym}
            symbol={sym}
            data={stockData[sym]}
            remove={() => removeStock(sym)}
            select={() => setSelected(sym)}
          />
        ))}
      </div>

      {selected && stockData[selected] && (
        <>
          <StockGraph symbol={selected} />
          <NewsList symbol={selected} />
          <RiskScore symbol={selected} />
          <BuySellForm symbol={selected} />
        </>
      )}

      <div style={{ marginTop: '2rem' }}>
        <input placeholder="Add Symbol" id="new-symbol" />
        <button onClick={() => {
          const val = document.getElementById('new-symbol').value.toUpperCase();
          addStock(val);
          document.getElementById('new-symbol').value = '';
        }}>Add Stock</button>
      </div>
    </div>
  );
}

export default Portfolio;
