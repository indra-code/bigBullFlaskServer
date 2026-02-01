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
            
            # Calculate technical indicators
            df['Returns'] = df['Close'].pct_change()
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA21'] = df['Close'].rolling(window=21).mean()
            df['MA50'] = df['Close'].rolling(window=min(50, len(df))).mean()
            
            # Volatility
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # RSI
            df['RSI'] = self._calculate_rsi(df['Close'])
            
            # MACD
            df['MACD'], df['Signal'] = self._calculate_macd(df['Close'])
            
            # Bollinger Bands
            df['BB_upper'], df['BB_lower'] = self._calculate_bollinger_bands(df['Close'])
            
            # Volume features
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Price momentum
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 30:
                raise ValueError("Insufficient data after feature engineering")
            
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
        
        Args:
            X: Features
            y: Target values
        """
        try:
            # Use Gradient Boosting for better performance
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict_future(self, data: pd.DataFrame, days: int = 30) -> Dict:
        """
        Predict future prices
        
        Args:
            data: Historical price data
            days: Number of days to predict
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Prepare features
            X, y = self.prepare_features(data)
            
            # Train model
            self.train_model(X, y)
            
            # Make predictions
            predictions = []
            current_data = data.copy()
            
            for i in range(days):
                # Prepare features for prediction
                X_pred, _ = self.prepare_features(current_data)
                X_pred_scaled = self.scaler.transform(X_pred[-1:])
                
                # Predict next day
                next_price = self.model.predict(X_pred_scaled)[0]
                
                # Create prediction date
                last_date = current_data.index[-1]
                next_date = last_date + timedelta(days=1)
                
                # Add prediction to list
                predictions.append({
                    'date': next_date.isoformat(),
                    'predicted_price': float(next_price),
                    'day': i + 1
                })
                
                # Update current_data with prediction for next iteration
                new_row = pd.DataFrame({
                    'Open': [next_price],
                    'High': [next_price * 1.01],
                    'Low': [next_price * 0.99],
                    'Close': [next_price],
                    'Volume': [current_data['Volume'].mean()]
                }, index=[next_date])
                
                current_data = pd.concat([current_data, new_row])
            
            # Calculate prediction statistics
            current_price = float(data['Close'].iloc[-1])
            final_price = predictions[-1]['predicted_price']
            price_change = ((final_price - current_price) / current_price) * 100
            
            return {
                'current_price': current_price,
                'predictions': predictions,
                'summary': {
                    'days_predicted': days,
                    'final_predicted_price': final_price,
                    'expected_change_percent': round(price_change, 2),
                    'trend': 'bullish' if price_change > 0 else 'bearish',
                    'confidence': self._calculate_confidence(data, predictions)
                }
            }
        
        except Exception as e:
            logger.error(f"Error predicting future prices: {str(e)}")
            return {
                'error': str(e),
                'predictions': []
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            return macd, signal_line
        except:
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            ma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = ma + (std * std_dev)
            lower_band = ma - (std * std_dev)
            return upper_band, lower_band
        except:
            return prices, prices
    
    def _calculate_confidence(self, historical_data: pd.DataFrame, predictions: List[Dict]) -> str:
        """
        Calculate confidence level for predictions
        
        Args:
            historical_data: Historical price data
            predictions: List of predictions
            
        Returns:
            Confidence level (low, medium, high)
        """
        try:
            # Calculate historical volatility
            returns = historical_data['Close'].pct_change().dropna()
            volatility = returns.std() * 100
            
            # Calculate data sufficiency
            data_points = len(historical_data)
            
            # Determine confidence
            if volatility < 2 and data_points > 90:
                return 'high'
            elif volatility < 4 and data_points > 60:
                return 'medium'
            else:
                return 'low'
        
        except:
            return 'medium'


class SimpleMovingAveragePredictor:
    """Simple moving average based prediction (fallback method)"""
    
    @staticmethod
    def predict(data: pd.DataFrame, days: int = 30) -> Dict:
        """
        Simple MA-based prediction
        
        Args:
            data: Historical price data
            days: Number of days to predict
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Calculate trend using linear regression
            prices = data['Close'].values
            x = np.arange(len(prices))
            
            # Fit linear trend
            coeffs = np.polyfit(x, prices, 1)
            trend_line = np.poly1d(coeffs)
            
            # Generate predictions
            predictions = []
            last_date = data.index[-1]
            current_price = float(prices[-1])
            
            for i in range(1, days + 1):
                next_date = last_date + timedelta(days=i)
                predicted_price = float(trend_line(len(prices) + i))
                
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
                    'final_predicted_price': final_price,
                    'expected_change_percent': round(price_change, 2),
                    'trend': 'bullish' if price_change > 0 else 'bearish',
                    'confidence': 'low',
                    'method': 'simple_moving_average'
                }
            }
        
        except Exception as e:
            logger.error(f"Error in simple prediction: {str(e)}")
            return {
                'error': str(e),
                'predictions': []
            }