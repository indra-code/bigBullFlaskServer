import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Risk analysis and scoring for assets"""
    
    @staticmethod
    def calculate_risk_score(historical_data: pd.DataFrame, 
                            current_price: float,
                            news_sentiment: Optional[Dict] = None) -> Dict:
        """
        Calculate comprehensive risk score for an asset
        
        Args:
            historical_data: Historical price data
            current_price: Current asset price
            news_sentiment: News sentiment data (optional)
            
        Returns:
            Dictionary with risk analysis
        """
        try:
            if historical_data.empty:
                return {
                    'risk_score': 50,
                    'risk_level': 'medium',
                    'error': 'Insufficient data'
                }
            
            # Calculate various risk metrics
            volatility = RiskAnalyzer._calculate_volatility(historical_data)
            sharpe_ratio = RiskAnalyzer._calculate_sharpe_ratio(historical_data)
            max_drawdown = RiskAnalyzer._calculate_max_drawdown(historical_data)
            beta = RiskAnalyzer._calculate_beta(historical_data)
            var_95 = RiskAnalyzer._calculate_var(historical_data, confidence=0.95)
            
            # Calculate trend strength
            trend = RiskAnalyzer._calculate_trend(historical_data)
            
            # Base risk score (0-100, higher = more risky)
            risk_score = 0
            
            # Volatility component (0-35 points)
            if volatility < 15:
                risk_score += 5
            elif volatility < 25:
                risk_score += 15
            elif volatility < 40:
                risk_score += 25
            else:
                risk_score += 35
            
            # Sharpe ratio component (0-20 points, inverse)
            if sharpe_ratio > 1.5:
                risk_score += 0
            elif sharpe_ratio > 1.0:
                risk_score += 5
            elif sharpe_ratio > 0.5:
                risk_score += 10
            elif sharpe_ratio > 0:
                risk_score += 15
            else:
                risk_score += 20
            
            # Max drawdown component (0-25 points)
            if abs(max_drawdown) < 10:
                risk_score += 5
            elif abs(max_drawdown) < 20:
                risk_score += 10
            elif abs(max_drawdown) < 30:
                risk_score += 15
            elif abs(max_drawdown) < 40:
                risk_score += 20
            else:
                risk_score += 25
            
            # Trend component (0-10 points)
            if trend == 'strong_uptrend':
                risk_score += 0
            elif trend == 'uptrend':
                risk_score += 2
            elif trend == 'sideways':
                risk_score += 5
            elif trend == 'downtrend':
                risk_score += 7
            else:  # strong_downtrend
                risk_score += 10
            
            # News sentiment component (0-10 points)
            if news_sentiment:
                sentiment_label = news_sentiment.get('sentiment_label', 'neutral')
                if sentiment_label == 'positive':
                    risk_score += 0
                elif sentiment_label == 'neutral':
                    risk_score += 5
                else:  # negative
                    risk_score += 10
            else:
                risk_score += 5  # Neutral if no news data
            
            # Determine risk level
            if risk_score < 30:
                risk_level = 'low'
            elif risk_score < 50:
                risk_level = 'medium-low'
            elif risk_score < 70:
                risk_level = 'medium'
            elif risk_score < 85:
                risk_level = 'medium-high'
            else:
                risk_level = 'high'
            
            return {
                'risk_score': min(100, max(0, risk_score)),
                'risk_level': risk_level,
                'metrics': {
                    'volatility': round(volatility, 2),
                    'sharpe_ratio': round(sharpe_ratio, 3),
                    'max_drawdown': round(max_drawdown, 2),
                    'beta': round(beta, 3),
                    'var_95': round(var_95, 2),
                    'trend': trend
                },
                'recommendation': RiskAnalyzer._get_recommendation(risk_score, trend)
            }
        
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return {
                'risk_score': 50,
                'risk_level': 'medium',
                'error': str(e)
            }
    
    @staticmethod
    def _calculate_volatility(data: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        try:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            return volatility
        except:
            return 0.0
    
    @staticmethod
    def _calculate_sharpe_ratio(data: pd.DataFrame, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        try:
            returns = data['Close'].pct_change().dropna()
            avg_return = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)
            
            if volatility == 0:
                return 0.0
            
            sharpe = (avg_return - risk_free_rate) / volatility
            return sharpe
        except:
            return 0.0
    
    @staticmethod
    def _calculate_max_drawdown(data: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage"""
        try:
            cumulative = (1 + data['Close'].pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            return drawdown.min()
        except:
            return 0.0
    
    @staticmethod
    def _calculate_beta(data: pd.DataFrame, market_return: float = 0.12) -> float:
        """
        Calculate beta (simplified version using historical volatility)
        In production, should compare against market index
        """
        try:
            returns = data['Close'].pct_change().dropna()
            # Simplified beta calculation
            asset_volatility = returns.std()
            beta = asset_volatility / 0.15  # Assuming market volatility of 15%
            return beta
        except:
            return 1.0
    
    @staticmethod
    def _calculate_var(data: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            returns = data['Close'].pct_change().dropna()
            var = np.percentile(returns, (1 - confidence) * 100) * 100
            return abs(var)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_trend(data: pd.DataFrame) -> str:
        """Determine price trend"""
        try:
            if len(data) < 20:
                return 'sideways'
            
            # Calculate moving averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=min(50, len(data))).mean()
            
            current_price = data['Close'].iloc[-1]
            ma20 = data['MA20'].iloc[-1]
            ma50 = data['MA50'].iloc[-1]
            
            # Calculate price change percentage
            price_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            
            # Determine trend
            if current_price > ma20 > ma50 and price_change > 10:
                return 'strong_uptrend'
            elif current_price > ma20:
                return 'uptrend'
            elif current_price < ma20 < ma50 and price_change < -10:
                return 'strong_downtrend'
            elif current_price < ma20:
                return 'downtrend'
            else:
                return 'sideways'
        
        except:
            return 'sideways'
    
    @staticmethod
    def _get_recommendation(risk_score: int, trend: str) -> str:
        """Generate investment recommendation"""
        if risk_score < 30:
            if trend in ['strong_uptrend', 'uptrend']:
                return 'Strong Buy - Low risk with positive trend'
            else:
                return 'Hold - Low risk but weak trend'
        elif risk_score < 50:
            if trend in ['strong_uptrend', 'uptrend']:
                return 'Buy - Moderate risk with positive momentum'
            elif trend == 'sideways':
                return 'Hold - Moderate risk with neutral trend'
            else:
                return 'Caution - Moderate risk with negative trend'
        elif risk_score < 70:
            if trend in ['strong_uptrend', 'uptrend']:
                return 'Hold - Medium-high risk despite positive trend'
            else:
                return 'Sell - High risk with unfavorable trend'
        else:
            return 'Strong Sell - Very high risk profile'