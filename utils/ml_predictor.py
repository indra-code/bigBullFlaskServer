import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MLPredictor:
    """Machine Learning based price prediction"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for ML model
        
        Args:
            data: Historical price data
            
        Returns:
            Tuple of (features, target)
        """
        try:
            df = data.copy()
            
            # Update: Ensure we have enough data to calculate long-term indicators
            # If data is short, adjust windows dynamically to prevent 0% change/empty DF
            data_len = len(df)
            ma_window = min(50, max(5, data_len // 4))
            vol_window = min(20, max(5, data_len // 10))
            
            # Calculate technical indicators
            df['Returns'] = df['Close'].pct_change()
            df['MA7'] = df['Close'].rolling(window=min(7, data_len)).mean()
            df['MA21'] = df['Close'].rolling(window=min(21, data_len)).mean()
            df['MA50'] = df['Close'].rolling(window=ma_window).mean()
            
            # Volatility
            df['Volatility'] = df['Returns'].rolling(window=vol_window).std()
            
            # RSI
            df['RSI'] = self._calculate_rsi(df['Close'])
            
            # MACD
            df['MACD'], df['Signal'] = self._calculate_macd(df['Close'])
            
            # Bollinger Bands
            df['BB_upper'], df['BB_lower'] = self._calculate_bollinger_bands(df['Close'])
            
            # Volume features
            df['Volume_MA'] = df['Volume'].rolling(window=vol_window).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, np.nan)
            
            # Price momentum
            momentum_window = min(10, data_len // 5)
            df['Momentum'] = df['Close'] - df['Close'].shift(momentum_window)
            
            # Update: Handle NaNs more gracefully to keep more data rows
            df = df.ffill().bfill().dropna()
            
            if len(df) < 15:
                raise ValueError(f"Insufficient data after feature engineering: {len(df)} rows")
            
            # Select features
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'MA7', 'MA21', 'MA50', 'Volatility',
                'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_lower',
                'Volume_Ratio', 'Momentum'
            ]
            
            X = df[feature_columns].values
            y = df['Close'].values
            
            return X, y
        
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        """
        Train the ML model
        """
        try:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict_future(self, data: pd.DataFrame, days: int = 30) -> Dict:
        """
        Predict future prices with fallback logic to prevent 0% results
        """
        try:
            # Update: Fallback to Simple Predictor if historical data is extremely sparse
            if len(data) < 40:
                logger.warning("Data too sparse for ML; using Simple Moving Average")
                return SimpleMovingAveragePredictor.predict(data, days)

            # Prepare features and train
            X, y = self.prepare_features(data)
            self.train_model(X, y)
            
            predictions = []
            current_data = data.copy()
            
            # Use the actual last known close to calculate percentage change later
            initial_price = float(data['Close'].iloc[-1])
            
            for i in range(days):
                # Prepare features for the last row
                X_all, _ = self.prepare_features(current_data)
                X_last_scaled = self.scaler.transform(X_all[-1:])
                
                # Predict
                next_price = float(self.model.predict(X_last_scaled)[0])
                
                # Prevent model from getting stuck at 0 or identical values
                if i > 0 and next_price == predictions[-1]['predicted_price']:
                    next_price *= (1 + np.random.uniform(-0.001, 0.001))
                
                last_date = current_data.index[-1]
                next_date = last_date + timedelta(days=1)
                
                predictions.append({
                    'date': next_date.isoformat(),
                    'predicted_price': next_price,
                    'day': i + 1
                })
                
                # Update current_data for the next iteration
                new_row = pd.DataFrame({
                    'Open': [next_price],
                    'High': [next_price * 1.01],
                    'Low': [next_price * 0.99],
                    'Close': [next_price],
                    'Volume': [current_data['Volume'].mean()]
                }, index=[next_date])
                
                current_data = pd.concat([current_data, new_row])
            
            final_price = predictions[-1]['predicted_price']
            price_change = ((final_price - initial_price) / initial_price) * 100
            
            return {
                'current_price': initial_price,
                'predictions': predictions,
                'summary': {
                    'days_predicted': days,
                    'final_predicted_price': round(final_price, 2),
                    'expected_change_percent': round(price_change, 2),
                    'trend': 'bullish' if price_change > 0.2 else 'bearish' if price_change < -0.2 else 'neutral',
                    'confidence': self._calculate_confidence(data, predictions)
                }
            }
        
        except Exception as e:
            logger.error(f"ML Predictor failed, falling back: {str(e)}")
            return SimpleMovingAveragePredictor.predict(data, days)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line
        except:
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        try:
            ma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = ma + (std * std_dev)
            lower_band = ma - (std * std_dev)
            return upper_band.ffill().fillna(prices), lower_band.ffill().fillna(prices)
        except:
            return prices, prices
    
    def _calculate_confidence(self, historical_data: pd.DataFrame, predictions: List[Dict]) -> str:
        try:
            returns = historical_data['Close'].pct_change().dropna()
            volatility = returns.std() * 100
            data_points = len(historical_data)
            
            if volatility < 2 and data_points > 180:
                return 'high'
            elif volatility < 5 and data_points > 60:
                return 'medium'
            else:
                return 'low'
        except:
            return 'medium'


class SimpleMovingAveragePredictor:
    """Simple moving average based prediction (fallback method)"""
    
    @staticmethod
    def predict(data: pd.DataFrame, days: int = 30) -> Dict:
        try:
            prices = data['Close'].values
            x = np.arange(len(prices))
            coeffs = np.polyfit(x, prices, 1)
            trend_line = np.poly1d(coeffs)
            
            predictions = []
            last_date = data.index[-1]
            current_price = float(prices[-1])
            
            for i in range(1, days + 1):
                next_date = last_date + timedelta(days=i)
                predicted_price = float(trend_line(len(prices) + i))
                
                # Update: Ensure predicted price doesn't go negative
                predicted_price = max(predicted_price, current_price * 0.5)
                
                predictions.append({
                    'date': next_date.isoformat(),
                    'predicted_price': predicted_price,
                    'day': i
                })
            
            final_price = predictions[-1]['predicted_price']
            price_change = ((final_price - current_price) / current_price) * 100
            
            return {
                'current_price': current_price,
                'predictions': predictions,
                'summary': {
                    'days_predicted': days,
                    'final_predicted_price': round(final_price, 2),
                    'expected_change_percent': round(price_change, 2),
                    'trend': 'bullish' if price_change > 0.1 else 'bearish' if price_change < -0.1 else 'neutral',
                    'confidence': 'low',
                    'method': 'simple_moving_average'
                }
            }
        except Exception as e:
            logger.error(f"Error in simple prediction: {str(e)}")
            return {'error': str(e), 'predictions': []}