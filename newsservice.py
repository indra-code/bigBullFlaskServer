import finnhub
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from textblob import TextBlob

logger = logging.getLogger(__name__)


class NewsService:
    """Service for fetching and analyzing news using Finnhub API"""
    
    def __init__(self, api_key: str):
        """
        Initialize news service with Finnhub API key
        
        Args:
            api_key: Finnhub API key
        """
        self.api_key = api_key
        self.client = None
        
        if api_key:
            try:
                self.client = finnhub.Client(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Finnhub client: {str(e)}")
    
    def get_company_news(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """
        Get company news for a stock symbol
        
        Args:
            symbol: Stock symbol (without exchange suffix for Finnhub)
            days_back: Number of days to look back
            
        Returns:
            List of news articles
        """
        if not self.client:
            logger.warning("Finnhub client not initialized")
            return []
        
        try:
            # Remove exchange suffix for Finnhub API
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('-USD', '')
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Fetch news
            news = self.client.company_news(clean_symbol, _from=from_date, to=to_date)
            
            # Process and enrich news
            processed_news = []
            for article in news[:10]:  # Limit to 10 most recent
                processed_article = {
                    'id': article.get('id'),
                    'headline': article.get('headline'),
                    'summary': article.get('summary'),
                    'source': article.get('source'),
                    'url': article.get('url'),
                    'datetime': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                    'image': article.get('image'),
                    'category': article.get('category'),
                    'sentiment': self._analyze_sentiment(article.get('headline', '') + ' ' + article.get('summary', ''))
                }
                processed_news.append(processed_article)
            
            return processed_news
        
        except Exception as e:
            logger.error(f"Error fetching company news for {symbol}: {str(e)}")
            return []
    
    def get_market_news(self, category: str = 'general', count: int = 10) -> List[Dict]:
        """
        Get general market news
        
        Args:
            category: News category (general, forex, crypto, merger)
            count: Number of articles to return
            
        Returns:
            List of news articles
        """
        if not self.client:
            logger.warning("Finnhub client not initialized")
            return []
        
        try:
            news = self.client.general_news(category, min_id=0)
            
            processed_news = []
            for article in news[:count]:
                processed_article = {
                    'id': article.get('id'),
                    'headline': article.get('headline'),
                    'summary': article.get('summary'),
                    'source': article.get('source'),
                    'url': article.get('url'),
                    'datetime': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                    'image': article.get('image'),
                    'category': article.get('category'),
                    'sentiment': self._analyze_sentiment(article.get('headline', '') + ' ' + article.get('summary', ''))
                }
                processed_news.append(processed_article)
            
            return processed_news
        
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'label': label
            }
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'label': 'neutral'
            }
    
    def get_news_sentiment_summary(self, symbol: str, days_back: int = 7) -> Dict:
        """
        Get aggregated news sentiment for a symbol
        
        Args:
            symbol: Asset symbol
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with sentiment summary
        """
        news = self.get_company_news(symbol, days_back)
        
        if not news:
            return {
                'symbol': symbol,
                'total_articles': 0,
                'average_polarity': 0.0,
                'sentiment_label': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        polarities = [article['sentiment']['polarity'] for article in news]
        sentiments = [article['sentiment']['label'] for article in news]
        
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0
        
        # Overall sentiment label
        if avg_polarity > 0.1:
            overall_sentiment = 'positive'
        elif avg_polarity < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'symbol': symbol,
            'total_articles': len(news),
            'average_polarity': round(avg_polarity, 3),
            'sentiment_label': overall_sentiment,
            'positive_count': sentiments.count('positive'),
            'negative_count': sentiments.count('negative'),
            'neutral_count': sentiments.count('neutral'),
            'recent_news': news[:5]  # Include 5 most recent articles
        }