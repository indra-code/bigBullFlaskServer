import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class InsightsGenerator:
    """Generate AI-powered investment insights in natural language"""
    
    @staticmethod
    def generate_asset_insights(
        symbol: str,
        current_price: float,
        historical_data: pd.DataFrame,
        risk_analysis: Dict,
        prediction: Dict,
        news_sentiment: Optional[Dict] = None
    ) -> Dict:
        """
        Generate comprehensive insights for an asset
        
        Args:
            symbol: Asset symbol
            current_price: Current price
            historical_data: Historical price data
            risk_analysis: Risk analysis results
            prediction: Price prediction results
            news_sentiment: News sentiment analysis (optional)
            
        Returns:
            Dictionary with insights
        """
        try:
            insights = []
            
            # Price performance insight
            price_insight = InsightsGenerator._generate_price_insight(
                current_price, historical_data
            )
            insights.append(price_insight)
            
            # Risk insight
            risk_insight = InsightsGenerator._generate_risk_insight(risk_analysis)
            insights.append(risk_insight)
            
            # Prediction insight
            prediction_insight = InsightsGenerator._generate_prediction_insight(
                prediction, current_price
            )
            insights.append(prediction_insight)
            
            # News sentiment insight
            if news_sentiment:
                sentiment_insight = InsightsGenerator._generate_sentiment_insight(
                    news_sentiment
                )
                insights.append(sentiment_insight)
            
            # Volatility insight
            volatility_insight = InsightsGenerator._generate_volatility_insight(
                risk_analysis
            )
            insights.append(volatility_insight)
            
            # Overall recommendation
            overall_recommendation = InsightsGenerator._generate_overall_recommendation(
                risk_analysis, prediction, news_sentiment
            )
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'insights': insights,
                'overall_recommendation': overall_recommendation,
                'summary': InsightsGenerator._generate_summary(
                    symbol, insights, overall_recommendation
                )
            }
        
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'insights': []
            }
    
    @staticmethod
    def _generate_price_insight(current_price: float, data: pd.DataFrame) -> Dict:
        """Generate price performance insight"""
        try:
            # Calculate performance metrics
            week_ago_price = data['Close'].iloc[-5] if len(data) >= 5 else data['Close'].iloc[0]
            month_ago_price = data['Close'].iloc[-20] if len(data) >= 20 else data['Close'].iloc[0]
            quarter_ago_price = data['Close'].iloc[-60] if len(data) >= 60 else data['Close'].iloc[0]
            
            week_change = ((current_price - week_ago_price) / week_ago_price) * 100
            month_change = ((current_price - month_ago_price) / month_ago_price) * 100
            quarter_change = ((current_price - quarter_ago_price) / quarter_ago_price) * 100
            
            # Generate insight text
            if month_change > 10:
                text = f"The asset has shown strong performance with a {abs(month_change):.2f}% gain over the past month. "
                text += "This indicates robust upward momentum and positive investor sentiment."
            elif month_change > 5:
                text = f"The asset has demonstrated moderate growth of {abs(month_change):.2f}% in the last month. "
                text += "This suggests a healthy upward trend with room for further appreciation."
            elif month_change > -5:
                text = f"The asset has remained relatively stable with a {abs(month_change):.2f}% change over the past month. "
                text += "This sideways movement indicates market indecision or consolidation phase."
            elif month_change > -10:
                text = f"The asset has experienced a {abs(month_change):.2f}% decline over the past month. "
                text += "This downward pressure suggests caution and potential for further correction."
            else:
                text = f"The asset has faced significant selling pressure with a {abs(month_change):.2f}% drop in the last month. "
                text += "This sharp decline indicates strong bearish sentiment and high risk."
            
            return {
                'category': 'price_performance',
                'title': 'Price Performance',
                'text': text,
                'metrics': {
                    'week_change': round(week_change, 2),
                    'month_change': round(month_change, 2),
                    'quarter_change': round(quarter_change, 2)
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating price insight: {str(e)}")
            return {
                'category': 'price_performance',
                'title': 'Price Performance',
                'text': 'Unable to analyze price performance due to insufficient data.',
                'metrics': {}
            }
    
    @staticmethod
    def _generate_risk_insight(risk_analysis: Dict) -> Dict:
        """Generate risk analysis insight"""
        try:
            risk_score = risk_analysis.get('risk_score', 50)
            risk_level = risk_analysis.get('risk_level', 'medium')
            metrics = risk_analysis.get('metrics', {})
            
            volatility = metrics.get('volatility', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            
            if risk_level == 'low':
                text = f"This asset exhibits low risk characteristics with a risk score of {risk_score}/100. "
                text += f"The volatility of {volatility:.2f}% indicates stable price movements, making it suitable for conservative investors. "
                text += f"The Sharpe ratio of {sharpe:.2f} suggests good risk-adjusted returns."
            elif risk_level in ['medium-low', 'medium']:
                text = f"This asset presents moderate risk with a score of {risk_score}/100. "
                text += f"With volatility at {volatility:.2f}%, expect some price fluctuations but within manageable limits. "
                text += "This risk level is appropriate for balanced portfolios seeking growth with reasonable safety."
            else:
                text = f"This asset carries high risk with a score of {risk_score}/100. "
                text += f"The elevated volatility of {volatility:.2f}% indicates significant price swings. "
                text += "Only suitable for aggressive investors with high risk tolerance and longer investment horizons."
            
            return {
                'category': 'risk_analysis',
                'title': 'Risk Assessment',
                'text': text,
                'metrics': {
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'volatility': round(volatility, 2)
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating risk insight: {str(e)}")
            return {
                'category': 'risk_analysis',
                'title': 'Risk Assessment',
                'text': 'Unable to analyze risk profile.',
                'metrics': {}
            }
    
    @staticmethod
    def _generate_prediction_insight(prediction: Dict, current_price: float) -> Dict:
        """Generate price prediction insight"""
        try:
            summary = prediction.get('summary', {})
            expected_change = summary.get('expected_change_percent', 0)
            trend = summary.get('trend', 'neutral')
            confidence = summary.get('confidence', 'medium')
            final_price = summary.get('final_predicted_price', current_price)
            
            if expected_change > 10:
                text = f"Our AI model predicts a strong {trend} trend with an expected price increase of {abs(expected_change):.2f}% "
                text += f"to approximately ₹{final_price:.2f} over the next month. "
                text += f"This prediction has {confidence} confidence based on historical patterns and market indicators."
            elif expected_change > 5:
                text = f"The predictive model forecasts moderate growth of {abs(expected_change):.2f}%, "
                text += f"targeting ₹{final_price:.2f} in the coming month. "
                text += f"With {confidence} confidence, this suggests a favorable outlook for investors."
            elif expected_change > -5:
                text = f"Price predictions indicate sideways movement with minimal change of {abs(expected_change):.2f}%. "
                text += f"The asset is expected to trade around ₹{final_price:.2f}, "
                text += "suggesting a consolidation phase before the next directional move."
            else:
                text = f"Our analysis predicts a {trend} trend with a potential decline of {abs(expected_change):.2f}% "
                text += f"to approximately ₹{final_price:.2f}. "
                text += f"Investors should exercise caution. Prediction confidence: {confidence}."
            
            return {
                'category': 'prediction',
                'title': 'Future Outlook',
                'text': text,
                'metrics': {
                    'expected_change': round(expected_change, 2),
                    'predicted_price': round(final_price, 2),
                    'trend': trend,
                    'confidence': confidence
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating prediction insight: {str(e)}")
            return {
                'category': 'prediction',
                'title': 'Future Outlook',
                'text': 'Unable to generate price predictions.',
                'metrics': {}
            }
    
    @staticmethod
    def _generate_sentiment_insight(news_sentiment: Dict) -> Dict:
        """Generate news sentiment insight"""
        try:
            sentiment_label = news_sentiment.get('sentiment_label', 'neutral')
            total_articles = news_sentiment.get('total_articles', 0)
            positive_count = news_sentiment.get('positive_count', 0)
            negative_count = news_sentiment.get('negative_count', 0)
            
            if sentiment_label == 'positive':
                text = f"Recent market news is predominantly positive with {positive_count} out of {total_articles} articles "
                text += "showing favorable sentiment. This suggests strong market confidence and potential for continued growth. "
                text += "Positive news flow often attracts more investors and supports price appreciation."
            elif sentiment_label == 'neutral':
                text = f"News sentiment is balanced with mixed signals from {total_articles} recent articles. "
                text += f"The distribution shows {positive_count} positive and {negative_count} negative stories. "
                text += "This neutral backdrop suggests the market is awaiting clearer catalysts before establishing direction."
            else:
                text = f"Market news carries a negative tone with {negative_count} out of {total_articles} articles "
                text += "expressing concerns. This bearish sentiment could weigh on investor confidence and price action. "
                text += "Negative news often leads to increased selling pressure and volatility."
            
            return {
                'category': 'sentiment',
                'title': 'Market Sentiment',
                'text': text,
                'metrics': {
                    'sentiment': sentiment_label,
                    'total_articles': total_articles,
                    'positive_ratio': round((positive_count / total_articles * 100) if total_articles > 0 else 0, 1)
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating sentiment insight: {str(e)}")
            return {
                'category': 'sentiment',
                'title': 'Market Sentiment',
                'text': 'News sentiment data unavailable.',
                'metrics': {}
            }
    
    @staticmethod
    def _generate_volatility_insight(risk_analysis: Dict) -> Dict:
        """Generate volatility insight"""
        try:
            metrics = risk_analysis.get('metrics', {})
            volatility = metrics.get('volatility', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            
            if volatility < 15:
                text = f"The asset displays low volatility at {volatility:.2f}%, indicating stable and predictable price movements. "
                text += "This stability makes it suitable for risk-averse investors and provides a smoother investment experience."
            elif volatility < 30:
                text = f"Moderate volatility of {volatility:.2f}% suggests normal market fluctuations. "
                text += "Investors should expect periodic price swings but within a manageable range. "
                text += "This is typical for actively traded assets."
            else:
                text = f"High volatility at {volatility:.2f}% indicates significant price swings. "
                text += f"With a maximum drawdown of {abs(max_drawdown):.2f}%, investors must be prepared for substantial fluctuations. "
                text += "Only suitable for those with strong risk tolerance and active portfolio management."
            
            return {
                'category': 'volatility',
                'title': 'Volatility Analysis',
                'text': text,
                'metrics': {
                    'volatility': round(volatility, 2),
                    'max_drawdown': round(max_drawdown, 2)
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating volatility insight: {str(e)}")
            return {
                'category': 'volatility',
                'title': 'Volatility Analysis',
                'text': 'Volatility analysis unavailable.',
                'metrics': {}
            }
    
    @staticmethod
    def _generate_overall_recommendation(
        risk_analysis: Dict,
        prediction: Dict,
        news_sentiment: Optional[Dict]
    ) -> Dict:
        """Generate overall investment recommendation"""
        try:
            risk_score = risk_analysis.get('risk_score', 50)
            prediction_change = prediction.get('summary', {}).get('expected_change_percent', 0)
            sentiment_label = news_sentiment.get('sentiment_label', 'neutral') if news_sentiment else 'neutral'
            
            # Calculate recommendation score
            rec_score = 50  # Neutral starting point
            
            # Adjust based on prediction
            rec_score += prediction_change * 2
            
            # Adjust based on risk (inverse relationship)
            rec_score -= (risk_score - 50) * 0.5
            
            # Adjust based on sentiment
            if sentiment_label == 'positive':
                rec_score += 10
            elif sentiment_label == 'negative':
                rec_score -= 10
            
            # Determine action
            if rec_score > 70:
                action = 'STRONG BUY'
                text = "All indicators align favorably. This asset shows strong growth potential with manageable risk."
            elif rec_score > 55:
                action = 'BUY'
                text = "Positive signals outweigh negatives. Consider accumulating positions for medium to long-term gains."
            elif rec_score > 45:
                action = 'HOLD'
                text = "Mixed signals suggest maintaining current positions while monitoring developments closely."
            elif rec_score > 30:
                action = 'SELL'
                text = "Negative factors dominate. Consider reducing exposure or exiting positions to preserve capital."
            else:
                action = 'STRONG SELL'
                text = "Multiple risk factors present. Immediate action recommended to minimize potential losses."
            
            return {
                'action': action,
                'score': round(rec_score, 1),
                'text': text,
                'timeframe': 'next 30 days'
            }
        
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {
                'action': 'HOLD',
                'score': 50,
                'text': 'Insufficient data for recommendation.',
                'timeframe': 'N/A'
            }
    
    @staticmethod
    def _generate_summary(symbol: str, insights: List[Dict], recommendation: Dict) -> str:
        """Generate executive summary"""
        try:
            action = recommendation.get('action', 'HOLD')
            
            summary = f"Investment Summary for {symbol}: "
            summary += f"Our comprehensive analysis recommends '{action}' based on multiple factors. "
            
            # Extract key points from insights
            for insight in insights[:3]:  # Top 3 insights
                category = insight.get('category', '')
                if category == 'price_performance':
                    metrics = insight.get('metrics', {})
                    month_change = metrics.get('month_change', 0)
                    summary += f"Month performance: {month_change:+.2f}%. "
                elif category == 'risk_analysis':
                    metrics = insight.get('metrics', {})
                    risk_level = metrics.get('risk_level', 'medium')
                    summary += f"Risk level: {risk_level}. "
                elif category == 'prediction':
                    metrics = insight.get('metrics', {})
                    expected_change = metrics.get('expected_change', 0)
                    summary += f"Predicted change: {expected_change:+.2f}%. "
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Analysis summary for {symbol} unavailable."