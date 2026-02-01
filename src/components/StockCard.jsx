import React from 'react';

function StockCard({ symbol, data, remove, select }) {
  return (
    <div className="card" onClick={select} style={{ cursor: 'pointer' }}>
      <h2>{symbol}</h2>
      {data ? (
        <>
          <p>Price: ${data.price}</p>
          <p>Open: ${data.open}</p>
          <p>High: ${data.high}</p>
          <p>Low: ${data.low}</p>
        </>
      ) : (
        <p>Loading...</p>
      )}
      <button onClick={(e) => { e.stopPropagation(); remove(); }}>Remove</button>
    </div>
  );
}

export default StockCard;
