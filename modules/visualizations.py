"""
Visualization module with Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st


def plot_candlestick_chart(df: pd.DataFrame, title: str = "Stock Price Chart") -> go.Figure:
    """
    Create interactive candlestick chart
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Add SMA 50
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='orange', width=1)
        ))
    
    # Add SMA 200
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['SMA_200'],
            name='SMA 200',
            line=dict(color='red', width=1)
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title='Price (₹)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600,
        hovermode='x unified'
    )
    
    return fig


def plot_prediction_chart(df: pd.DataFrame, current_price: float, 
                         predicted_price: float, confidence: float) -> go.Figure:
    """
    Create chart showing prediction
    
    Args:
        df: DataFrame with historical prices
        current_price: Current stock price
        predicted_price: AI predicted price
        confidence: Model confidence percentage
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Historical line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        name='Historical Price',
        line=dict(color='cyan', width=2),
        mode='lines'
    ))
    
    # Current price marker
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[-1]],
        y=[current_price],
        mode='markers+text',
        marker=dict(size=15, color='gold', symbol='circle'),
        text=['Current'],
        textposition='top center',
        name='Current Price',
        hovertext=f"Current: ₹{current_price:.2f}"
    ))
    
    # Prediction marker
    next_date = pd.to_datetime(df['Date'].iloc[-1]) + pd.Timedelta(days=1)
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    marker_color = 'green' if predicted_price > current_price else 'red'
    
    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[predicted_price],
        mode='markers+text',
        marker=dict(size=15, color=marker_color, symbol='star'),
        text=['Prediction'],
        textposition='top center',
        name='AI Prediction',
        hovertext=f"Predicted: ₹{predicted_price:.2f}<br>Change: {change_pct:.2f}%<br>Confidence: {confidence:.1f}%"
    ))
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[-1], next_date],
        y=[current_price, predicted_price],
        mode='lines',
        line=dict(color=marker_color, width=2, dash='dash'),
        name='Trend',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Stock Price Prediction',
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_rsi_indicator(df: pd.DataFrame) -> go.Figure:
    """
    Plot RSI indicator
    
    Args:
        df: DataFrame with RSI data
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if 'RSI_14' not in df.columns:
        return fig
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['RSI_14'],
        name='RSI 14',
        line=dict(color='purple', width=2)
    ))
    
    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Mid")
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        yaxis_title='RSI',
        xaxis_title='Date',
        template='plotly_dark',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def plot_macd_indicator(df: pd.DataFrame) -> go.Figure:
    """
    Plot MACD indicator
    
    Args:
        df: DataFrame with MACD data
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if 'MACD' not in df.columns:
        return fig
    
    # MACD line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    # Signal line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MACD_signal'],
        name='Signal',
        line=dict(color='red', width=2)
    ))
    
    # Histogram
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['MACD_diff'],
        name='Histogram',
        marker=dict(color=df['MACD_diff'].apply(lambda x: 'green' if x > 0 else 'red')),
        opacity=0.3
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        yaxis_title='MACD',
        xaxis_title='Date',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_bollinger_bands(df: pd.DataFrame) -> go.Figure:
    """
    Plot Bollinger Bands
    
    Args:
        df: DataFrame with Bollinger Bands data
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if 'BB_High' not in df.columns:
        return fig
    
    # Upper band
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['BB_High'],
        name='Upper Band',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    # Middle band (SMA)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['BB_Mid'],
        name='Middle Band',
        line=dict(color='orange', width=1)
    ))
    
    # Lower band
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['BB_Low'],
        name='Lower Band',
        line=dict(color='green', width=1, dash='dash'),
        fill='tonexty'
    ))
    
    # Close price
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        name='Close Price',
        line=dict(color='cyan', width=2)
    ))
    
    fig.update_layout(
        title='Bollinger Bands',
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_volume_chart(df: pd.DataFrame) -> go.Figure:
    """
    Plot trading volume
    
    Args:
        df: DataFrame with volume data
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = ['green' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'red' 
              for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker=dict(color=colors)
    ))
    
    fig.update_layout(
        title='Trading Volume',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """
    Plot feature importance
    
    Args:
        importance_df: DataFrame with features and importance scores
    
    Returns:
        Plotly figure
    """
    top_features = importance_df.head(15)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_features['Feature'],
        x=top_features['Importance'],
        orientation='h',
        marker=dict(color='cyan')
    ))
    
    fig.update_layout(
        title='Top 15 Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_dark',
        height=500,
        hovermode='y unified'
    )
    
    return fig


def plot_sentiment_distribution(sentiment_data: dict) -> go.Figure:
    """
    Plot sentiment distribution
    
    Args:
        sentiment_data: Dictionary with sentiment distribution
    
    Returns:
        Plotly figure
    """
    distribution = sentiment_data.get('distribution', {'positive': 0, 'negative': 0, 'neutral': 0})
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Negative', 'Neutral'],
        values=[distribution['positive'], distribution['negative'], distribution['neutral']],
        marker=dict(colors=['green', 'red', 'gray']),
        hole=0.3
    )])
    
    fig.update_layout(
        title='News Sentiment Distribution',
        template='plotly_dark',
        height=400
    )
    
    return fig
