"""
Sentiment analysis module using FinBERT and news APIs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from gnews import GNews
import warnings
warnings.filterwarnings('ignore')

# Lazy load transformers to avoid startup delays
_sentiment_model = None
_sentiment_tokenizer = None


def get_sentiment_model():
    """Lazy load FinBERT model"""
    global _sentiment_model, _sentiment_tokenizer
    
    if _sentiment_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            _sentiment_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            _sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            _sentiment_model.eval()
        except Exception as e:
            st.warning(f"Could not load FinBERT model: {str(e)}. Using fallback sentiment.")
            return None, None
    
    return _sentiment_model, _sentiment_tokenizer


def fetch_news_headlines(company_name: str, ticker: str = None, days_back: int = 30) -> list:
    """
    Fetch news headlines using GNews (free, no API key required)
    
    Args:
        company_name: Company name to search
        ticker: Stock ticker (optional)
        days_back: Number of days to search back
    
    Returns:
        List of news articles
    """
    try:
        google_news = GNews(language="en", country="IN", period=f"{days_back}d", max_results=100)
        
        # Search for company name
        articles = google_news.get_news(company_name)
        
        if not articles and ticker:
            # Fallback: search by ticker
            articles = google_news.get_news(ticker)
        
        return articles
        
    except Exception as e:
        st.warning(f"Error fetching news: {str(e)}")
        return []


def analyze_sentiment_finbert(headlines: list) -> dict:
    """
    Analyze sentiment using FinBERT model
    
    Args:
        headlines: List of news headlines/text
    
    Returns:
        Dictionary with sentiment analysis results
    """
    import torch
    
    try:
        model, tokenizer = get_sentiment_model()
        
        if model is None or tokenizer is None:
            return analyze_sentiment_fallback(headlines)
        
        sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        scores = []
        sentiment_details = []
        
        for headline in headlines:
            try:
                # Truncate if too long
                text = headline[:512] if len(headline) > 512 else headline
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # FinBERT outputs: positive, negative, neutral
                positive_prob = probs[0][0].item()
                negative_prob = probs[0][1].item()
                neutral_prob = probs[0][2].item()
                
                # Determine sentiment
                if positive_prob > max(negative_prob, neutral_prob):
                    sentiment = 'positive'
                    sentiments['positive'] += 1
                elif negative_prob > max(positive_prob, neutral_prob):
                    sentiment = 'negative'
                    sentiments['negative'] += 1
                else:
                    sentiment = 'neutral'
                    sentiments['neutral'] += 1
                
                # Score: +1 to -1
                score = positive_prob - negative_prob
                scores.append(score)
                
                sentiment_details.append({
                    'headline': headline[:100],
                    'sentiment': sentiment,
                    'confidence': max(positive_prob, negative_prob, neutral_prob),
                    'score': score
                })
                
            except Exception as e:
                # Handle individual headline errors
                continue
        
        if not scores:
            return {
                'avg_score': 0,
                'sentiment': 'neutral',
                'distribution': sentiments,
                'details': [],
                'total_articles': 0
            }
        
        avg_score = np.mean(scores)
        
        # Classify overall sentiment
        if avg_score > 0.2:
            overall_sentiment = '🟢 Bullish'
        elif avg_score < -0.2:
            overall_sentiment = '🔴 Bearish'
        else:
            overall_sentiment = '🟡 Neutral'
        
        return {
            'avg_score': avg_score,
            'sentiment': overall_sentiment,
            'distribution': sentiments,
            'details': sentiment_details,
            'total_articles': len(sentiment_details)
        }
        
    except Exception as e:
        st.warning(f"Error in sentiment analysis: {str(e)}")
        return analyze_sentiment_fallback(headlines)


def analyze_sentiment_fallback(headlines: list) -> dict:
    """
    Fallback sentiment analysis using keyword matching
    
    Args:
        headlines: List of headlines
    
    Returns:
        Dictionary with sentiment results
    """
    positive_keywords = ['gain', 'profit', 'rise', 'surge', 'jump', 'rally', 'win', 
                         'bullish', 'upgrade', 'outperform', 'growth', 'boost', 'strong']
    negative_keywords = ['loss', 'fall', 'drop', 'decline', 'crash', 'weak', 'risk',
                         'bearish', 'downgrade', 'underperform', 'slump', 'concern', 'fear']
    
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    scores = []
    
    for headline in headlines:
        headline_lower = headline.lower()
        
        pos_count = sum(1 for keyword in positive_keywords if keyword in headline_lower)
        neg_count = sum(1 for keyword in negative_keywords if keyword in headline_lower)
        
        if pos_count > neg_count:
            sentiments['positive'] += 1
            scores.append(0.5)
        elif neg_count > pos_count:
            sentiments['negative'] += 1
            scores.append(-0.5)
        else:
            sentiments['neutral'] += 1
            scores.append(0)
    
    avg_score = np.mean(scores) if scores else 0
    
    if avg_score > 0.2:
        overall_sentiment = '🟢 Bullish'
    elif avg_score < -0.2:
        overall_sentiment = '🔴 Bearish'
    else:
        overall_sentiment = '🟡 Neutral'
    
    return {
        'avg_score': avg_score,
        'sentiment': overall_sentiment,
        'distribution': sentiments,
        'details': [],
        'total_articles': len(headlines)
    }


def get_sentiment_features(company_name: str, ticker: str = None, days_back: int = 30) -> dict:
    """
    Get sentiment features for model input
    
    Args:
        company_name: Company name
        ticker: Stock ticker
        days_back: Days to analyze
    
    Returns:
        Dictionary with sentiment features
    """
    try:
        # Fetch news
        articles = fetch_news_headlines(company_name, ticker, days_back)
        
        if not articles:
            return {
                'sentiment_score': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 1,
                'article_count': 0,
                'sentiment_label': '🟡 Neutral'
            }
        
        # Extract headlines
        headlines = [article.get('title', '') for article in articles if article.get('title')]
        
        # Analyze sentiment
        sentiment_result = analyze_sentiment_finbert(headlines)
        
        total = sentiment_result['total_articles']
        dist = sentiment_result['distribution']
        
        return {
            'sentiment_score': sentiment_result['avg_score'],
            'positive_ratio': dist['positive'] / total if total > 0 else 0,
            'negative_ratio': dist['negative'] / total if total > 0 else 0,
            'neutral_ratio': dist['neutral'] / total if total > 0 else 0,
            'article_count': total,
            'sentiment_label': sentiment_result['sentiment'],
            'details': sentiment_result['details']
        }
        
    except Exception as e:
        st.warning(f"Error getting sentiment features: {str(e)}")
        return {
            'sentiment_score': 0,
            'positive_ratio': 0,
            'negative_ratio': 0,
            'neutral_ratio': 1,
            'article_count': 0,
            'sentiment_label': '🟡 Neutral'
        }
