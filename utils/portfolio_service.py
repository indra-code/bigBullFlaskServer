import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PortfolioService:
    def __init__(self):
        # In a real app, this would be replaced by a Database (SQL/NoSQL)
        self.holdings = {} # Format: { 'SYMBOL': {'units': 0.0, 'avg_price': 0.0} }

    def buy_asset(self, symbol: str, units: float, price: float) -> Dict:
        """Adds units to portfolio and recalculates Weighted Average Cost"""
        if symbol not in self.holdings:
            self.holdings[symbol] = {'units': 0.0, 'avg_price': 0.0}
        
        current = self.holdings[symbol]
        
        # Formula: (Current Total Cost + New Cost) / Total Units
        total_cost = (current['units'] * current['avg_price']) + (units * price)
        new_total_units = current['units'] + units
        
        current['avg_price'] = total_cost / new_total_units if new_total_units > 0 else 0
        current['units'] = new_total_units
        
        return {
            'symbol': symbol,
            'units': current['units'],
            'average_cost': round(current['avg_price'], 2)
        }

    def sell_asset(self, symbol: str, units_to_sell: float, current_market_price: float) -> Dict:
        """Sells units and calculates Profit/Loss based on Average Pricing"""
        if symbol not in self.holdings or self.holdings[symbol]['units'] < units_to_sell:
            raise ValueError(f"Insufficient units to sell for {symbol}")
            
        current = self.holdings[symbol]
        avg_buy_price = current['avg_price']
        
        # Profit = (Selling Price - Average Buying Price) * Units Sold
        realized_pnl = (current_market_price - avg_buy_price) * units_to_sell
        
        current['units'] -= units_to_sell
        
        # If portfolio for this asset is zero, reset avg_price
        if current['units'] == 0:
            current['avg_price'] = 0.0
            
        return {
            'symbol': symbol,
            'units_sold': units_to_sell,
            'realized_pnl': round(realized_pnl, 2),
            'remaining_units': current['units'],
            'cost_basis': round(avg_buy_price, 2)
        }

    def get_portfolio(self) -> Dict:
        return self.holdings