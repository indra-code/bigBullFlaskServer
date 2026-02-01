import finnhub
import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from textblob import TextBlob

logger = logging.getLogger(__name__)

class NewsService:
    """Service for fetching and analyzing news using Finnhub and yfinance fallback"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        
        if api_key:
            try:
                self.client = finnhub.Client(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Finnhub client: {str(e)}")
    
    def get_company_news(self, symbol: str, days_back: int = 7) -> List[Dict]:
        processed_news = []
        
        # 1. Try Finnhub First (Best for Crypto and Global Stocks)
        if self.client:
            try:
                clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('-USD', '')
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                news = self.client.company_news(
                    clean_symbol, 
                    _from=start_date.strftime('%Y-%m-%d'), 
                    to=end_date.strftime('%Y-%m-%d')
                )
                
                for article in news[:10]:
                    processed_news.append({
                        'id': article.get('id'),
                        'headline': article.get('headline'),
                        'summary': article.get('summary'),
                        'source': article.get('source'),
                        'url': article.get('url'),
                        'datetime': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                        'image': article.get('image'),
                        'category': article.get('category'),
                        'sentiment': self._analyze_sentiment(article.get('headline', '') + ' ' + article.get('summary', ''))
                    })
            except Exception as e:
                logger.error(f"Finnhub error for {symbol}: {str(e)}")

        # 2. Update: Fallback to yfinance (Essential for Indian Stocks & Mutual Funds)
        if not processed_news or ('.NS' in symbol or '.BO' in symbol):
            try:
                ticker = yf.Ticker(symbol)
                yf_news = ticker.news
                if yf_news:
                    for article in yf_news[:10]:
                        # Map yfinance fields to your existing structure
                        processed_news.append({
                            'id': article.get('uuid'),
                            'headline': article.get('title'),
                            'summary': article.get('publisher'),
                            'source': article.get('publisher'),
                            'url': article.get('link'),
                            'datetime': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                            'image': None,
                            'category': 'yfinance_fallback',
                            'sentiment': self._analyze_sentiment(article.get('title', ''))
                        })
            except Exception as e:
                logger.error(f"yfinance fallback news error for {symbol}: {str(e)}")
            
        return processed_news

    def get_market_news(self, category: str = 'general', count: int = 10) -> List[Dict]:
        if not self.client:
            return []
        try:
            news = self.client.general_news(category, min_id=0)
            processed_news = []
            for article in news[:count]:
                processed_news.append({
                    'id': article.get('id'),
                    'headline': article.get('headline'),
                    'summary': article.get('summary'),
                    'source': article.get('source'),
                    'url': article.get('url'),
                    'datetime': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                    'image': article.get('image'),
                    'category': article.get('category'),
                    'sentiment': self._analyze_sentiment(article.get('headline', '') + ' ' + article.get('summary', ''))
                })
            return processed_news
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            return []

    def _analyze_sentiment(self, text: str) -> Dict:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            label = 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
            return {'polarity': round(polarity, 3), 'subjectivity': round(subjectivity, 3), 'label': label}
        except Exception as e:
            logger.error(f"Sentiment error: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'neutral'}

    def get_news_sentiment_summary(self, symbol: str, days_back: int = 7) -> Dict:
        news = self.get_company_news(symbol, days_back)
        if not news:
            return {'symbol': symbol, 'total_articles': 0, 'sentiment_label': 'neutral'}
        
        polarities = [a['sentiment']['polarity'] for a in news]
        sentiments = [a['sentiment']['label'] for a in news]
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0
        
        return {
            'symbol': symbol,
            'total_articles': len(news),
            'average_polarity': round(avg_polarity, 3),
            'sentiment_label': 'positive' if avg_polarity > 0.1 else 'negative' if avg_polarity < -0.1 else 'neutral',
            'positive_count': sentiments.count('positive'),
            'negative_count': sentiments.count('negative'),
            'neutral_count': sentiments.count('neutral'),
            'recent_news': news[:5]
        }