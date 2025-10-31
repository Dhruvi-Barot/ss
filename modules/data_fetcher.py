"""
Data fetching module for stock market data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st


@st.cache_data(ttl=3600)
def fetch_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE.NS")
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            st.error(f"No data found for ticker: {ticker}")
            return None
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Ensure column names are standard
        df.columns = df.columns.str.capitalize()
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=86400)
def fetch_fundamental_data(ticker: str) -> dict:
    """
    Fetch fundamental data and company information
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with fundamental data
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant fundamental data
        fundamentals = {
            'company_name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'peg_ratio': info.get('pegRatio', None),
            'price_to_book': info.get('priceToBook', None),
            'debt_to_equity': info.get('debtToEquity', None),
            'roe': info.get('returnOnEquity', None),
            'profit_margin': info.get('profitMargins', None),
            'dividend_yield': info.get('dividendYield', None),
            'beta': info.get('beta', None),
            '52_week_high': info.get('fiftyTwoWeekHigh', None),
            '52_week_low': info.get('fiftyTwoWeekLow', None),
            'avg_volume': info.get('averageVolume', None),
            'eps': info.get('trailingEps', None),
            'book_value': info.get('bookValue', None),
            'price': info.get('currentPrice', None),
            'target_price': info.get('targetMeanPrice', None),
            'recommendation': info.get('recommendationKey', 'N/A'),
            'currency': info.get('currency', 'INR')
        }
        
        # Convert percentages to readable format
        if fundamentals['roe']:
            fundamentals['roe'] = fundamentals['roe'] * 100
        if fundamentals['profit_margin']:
            fundamentals['profit_margin'] = fundamentals['profit_margin'] * 100
        if fundamentals['dividend_yield']:
            fundamentals['dividend_yield'] = fundamentals['dividend_yield'] * 100
        
        return fundamentals
        
    except Exception as e:
        st.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
        return {}


@st.cache_data(ttl=3600)
def fetch_recent_price(ticker: str) -> dict:
    """
    Fetch most recent price data
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with current price info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'current_price': info.get('currentPrice', None),
            'previous_close': info.get('previousClose', None),
            'open': info.get('open', None),
            'day_high': info.get('dayHigh', None),
            'day_low': info.get('dayLow', None),
            'volume': info.get('volume', None),
            'change': info.get('currentPrice', 0) - info.get('previousClose', 0),
            'change_percent': ((info.get('currentPrice', 0) - info.get('previousClose', 1)) / 
                             info.get('previousClose', 1) * 100) if info.get('previousClose') else 0
        }
        
    except Exception as e:
        st.error(f"Error fetching recent price for {ticker}: {str(e)}")
        return {}


def validate_ticker(ticker: str) -> bool:
    """
    Validate if ticker exists and has data
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Boolean indicating if ticker is valid
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid data
        if info.get('regularMarketPrice') or info.get('currentPrice'):
            return True
        return False
        
    except:
        return False


def get_stock_info_summary(ticker: str) -> str:
    """
    Get a brief text summary of the stock
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Text summary string
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        summary = info.get('longBusinessSummary', 'No summary available')
        return summary[:500] + "..." if len(summary) > 500 else summary
        
    except:
        return "Summary not available"
