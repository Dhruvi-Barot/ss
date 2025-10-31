"""
Utility functions for the stock analysis platform
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st


def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ['models', 'cache', 'reports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        # Create .gitkeep files
        gitkeep = Path(directory) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()


def format_currency(value, currency='₹'):
    """
    Format number as currency
    
    Args:
        value: Numeric value
        currency: Currency symbol (default: ₹ for INR)
    
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    if abs(value) >= 1e12:
        return f"{currency}{value/1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"{currency}{value/1e9:.2f}B"
    elif abs(value) >= 1e7:
        return f"{currency}{value/1e7:.2f}Cr"
    elif abs(value) >= 1e5:
        return f"{currency}{value/1e5:.2f}L"
    else:
        return f"{currency}{value:,.2f}"


def format_large_number(num):
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string
    """
    if num is None:
        return "N/A"
    
    if abs(num) >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def save_metadata(ticker, metadata):
    """
    Save model metadata to JSON file
    
    Args:
        ticker: Stock ticker symbol
        metadata: Dictionary of metadata
    """
    ensure_directories()
    filepath = Path('models') / f'{ticker}_metadata.json'
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(ticker):
    """
    Load model metadata from JSON file
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary of metadata or None
    """
    filepath = Path('models') / f'{ticker}_metadata.json'
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def get_recommendation_emoji(recommendation):
    """
    Get emoji for recommendation
    
    Args:
        recommendation: String recommendation
    
    Returns:
        Emoji string
    """
    rec_lower = recommendation.lower()
    if 'strong buy' in rec_lower:
        return "🚀"
    elif 'buy' in rec_lower:
        return "📈"
    elif 'hold' in rec_lower:
        return "⏸️"
    elif 'sell' in rec_lower:
        return "📉"
    elif 'strong sell' in rec_lower:
        return "⚠️"
    else:
        return "📊"


def calculate_change_percentage(current, previous):
    """
    Calculate percentage change
    
    Args:
        current: Current value
        previous: Previous value
    
    Returns:
        Percentage change
    """
    if previous == 0 or previous is None:
        return 0
    return ((current - previous) / previous) * 100
