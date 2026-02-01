import React, { useState } from 'react';

function BuySellForm({ symbol }) {
  const [type, setType] = useState('buy');
  const [quantity, setQuantity] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    alert(`${type.toUpperCase()} ${quantity} shares of ${symbol}`);
    setQuantity('');
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: '1rem' }}>
      <h4>Buy/Sell {symbol}</h4>
      <select value={type} onChange={e => setType(e.target.value)}>
        <option value="buy">Buy</option>
        <option value="sell">Sell</option>
      </select>
      <input
        type="number"
        value={quantity}
        onChange={e => setQuantity(e.target.value)}
        placeholder="Quantity"
        required
      />
      <button type="submit">{type.toUpperCase()}</button>
    </form>
  );
}

export default BuySellForm;
