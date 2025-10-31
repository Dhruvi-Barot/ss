"""
Feature engineering module for technical indicators and data preparation
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Safe imports with fallback
try:
    from ta.trend import MACD
    MACD_AVAILABLE = True
except ImportError:
    MACD_AVAILABLE = False

try:
    from ta.momentum import StochasticOscillator
    STOCH_AVAILABLE = True
except ImportError:
    STOCH_AVAILABLE = False

try:
    from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
    VOLUME_AVAILABLE = True
except ImportError:
    VOLUME_AVAILABLE = False


def safe_divide(numerator, denominator, fill_value=0):
    """Safely divide without creating infinities"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator, where=denominator!=0, out=np.full_like(numerator, fill_value, dtype=float))
    return result


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with technical indicators added
    """
    try:
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['Close', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return df
        
        # Convert to lowercase for ta-lib compatibility
        df_ta = df.copy()
        df_ta.columns = df_ta.columns.str.lower()
        
        # Simple Moving Averages
        try:
            df['SMA_20'] = SMAIndicator(close=df_ta['close'], window=20).sma_indicator()
            df['SMA_50'] = SMAIndicator(close=df_ta['close'], window=50).sma_indicator()
            df['SMA_200'] = SMAIndicator(close=df_ta['close'], window=200).sma_indicator()
        except:
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
        
        # Exponential Moving Averages
        try:
            df['EMA_12'] = EMAIndicator(close=df_ta['close'], window=12).ema_indicator()
            df['EMA_26'] = EMAIndicator(close=df_ta['close'], window=26).ema_indicator()
        except:
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI - Relative Strength Index
        try:
            rsi = RSIIndicator(close=df_ta['close'], window=14)
            df['RSI_14'] = rsi.rsi()
        except:
            df['RSI_14'] = 50  # Default neutral RSI
        
        # MACD - Moving Average Convergence Divergence
        if MACD_AVAILABLE:
            try:
                macd = MACD(close=df_ta['close'], window_fast=12, window_slow=26, window_sign=9)
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
                df['MACD_diff'] = macd.macd_diff()
            except:
                df['MACD'] = 0
                df['MACD_signal'] = 0
                df['MACD_diff'] = 0
        else:
            df['MACD'] = 0
            df['MACD_signal'] = 0
            df['MACD_diff'] = 0
        
        # Bollinger Bands
        try:
            bb = BollingerBands(close=df_ta['close'], window=20, window_dev=2)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Mid'] = bb.bollinger_mavg()
            df['BB_Low'] = bb.bollinger_lband()
        except:
            df['BB_High'] = df['Close'] * 1.02
            df['BB_Mid'] = df['Close']
            df['BB_Low'] = df['Close'] * 0.98
        
        # Average True Range (Volatility)
        try:
            atr = AverageTrueRange(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], window=14)
            df['ATR_14'] = atr.average_true_range()
        except:
            df['ATR_14'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # ADX - Average Directional Index (Trend Strength)
        try:
            adx = ADXIndicator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], window=14)
            df['ADX_14'] = adx.adx()
        except:
            df['ADX_14'] = 25  # Default neutral ADX
        
        # Stochastic Oscillator
        if STOCH_AVAILABLE:
            try:
                stoch = StochasticOscillator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], window=14, smooth_window=3)
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
            except:
                df['Stoch_K'] = 50
                df['Stoch_D'] = 50
        else:
            df['Stoch_K'] = 50
            df['Stoch_D'] = 50
        
        # Volume indicators (safe import with fallback)
        if VOLUME_AVAILABLE:
            try:
                obv = OnBalanceVolumeIndicator(close=df_ta['close'], volume=df_ta['volume'])
                df['OBV'] = obv.on_balance_volume()
            except:
                df['OBV'] = 0
            
            try:
                vwap = VolumeWeightedAveragePrice(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], volume=df_ta['volume'], window=14)
                df['VWAP'] = vwap.volume_weighted_average_price()
            except:
                df['VWAP'] = df['Close']
        else:
            df['OBV'] = 0
            df['VWAP'] = df['Close']
        
        # Price rate of change - with safe division
        pct_change = df['Close'].pct_change(periods=12)
        df['ROC'] = pct_change.fillna(0) * 100
        df['ROC'] = df['ROC'].replace([np.inf, -np.inf], 0)
        
        # Volume rate of change - with safe division
        vol_change = df['Volume'].pct_change(periods=5)
        df['Volume_ROC'] = vol_change.fillna(0) * 100
        df['Volume_ROC'] = df['Volume_ROC'].replace([np.inf, -np.inf], 0)
        
        # Volatility (20-day) - with safe division
        rolling_std = df['Close'].rolling(window=20).std()
        rolling_mean = df['Close'].rolling(window=20).mean()
        df['Volatility'] = safe_divide(rolling_std, rolling_mean, fill_value=0) * 100
        df['Volatility'] = df['Volatility'].replace([np.inf, -np.inf], 0)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.fillna(0)
        
        # Replace any remaining inf or -inf values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
        
    except Exception as e:
        print(f"Error adding technical indicators: {str(e)}")
        return df


def prepare_features_for_model(df: pd.DataFrame, fundamentals: dict = None) -> pd.DataFrame:
    """
    Prepare final feature set for model training
    
    Args:
        df: DataFrame with technical indicators
        fundamentals: Dictionary with fundamental data
    
    Returns:
        DataFrame ready for model training
    """
    try:
        df = df.copy()
        
        # Create price-based features - with safe division
        pct_change = df['Close'].pct_change()
        df['Price_Change'] = pct_change.fillna(0) * 100
        df['Price_Change'] = df['Price_Change'].replace([np.inf, -np.inf], 0)
        
        # High/Low ratio - safe division
        df['High_Low_Ratio'] = safe_divide(df['High'], df['Low'], fill_value=1)
        df['High_Low_Ratio'] = df['High_Low_Ratio'].replace([np.inf, -np.inf], 1)
        
        # Close/Open ratio - safe division
        df['Close_Open_Ratio'] = safe_divide(df['Close'], df['Open'], fill_value=1)
        df['Close_Open_Ratio'] = df['Close_Open_Ratio'].replace([np.inf, -np.inf], 1)
        
        # Volume MA ratio - safe division
        vol_ma = df['Volume'].rolling(window=20).mean()
        df['Volume_MA_Ratio'] = safe_divide(df['Volume'], vol_ma, fill_value=1)
        df['Volume_MA_Ratio'] = df['Volume_MA_Ratio'].replace([np.inf, -np.inf], 1)
        
        # Add lag features (previous days' closes)
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        
        # Add fundamental features if provided
        if fundamentals and isinstance(fundamentals, dict):
            if fundamentals.get('pe_ratio') and isinstance(fundamentals.get('pe_ratio'), (int, float)):
                pe = fundamentals['pe_ratio']
                if np.isfinite(pe):
                    df['PE_Ratio'] = pe
                else:
                    df['PE_Ratio'] = 0
            else:
                df['PE_Ratio'] = 0
            
            if fundamentals.get('market_cap'):
                mc = fundamentals['market_cap']
                if isinstance(mc, (int, float)) and mc > 0:
                    df['Market_Cap_Log'] = np.log1p(mc / 1e9)
                else:
                    df['Market_Cap_Log'] = 0
            else:
                df['Market_Cap_Log'] = 0
            
            if fundamentals.get('roe') and isinstance(fundamentals.get('roe'), (int, float)):
                roe = fundamentals['roe']
                if np.isfinite(roe):
                    df['ROE'] = roe
                else:
                    df['ROE'] = 0
            else:
                df['ROE'] = 0
            
            if fundamentals.get('profit_margin') and isinstance(fundamentals.get('profit_margin'), (int, float)):
                pm = fundamentals['profit_margin']
                if np.isfinite(pm):
                    df['Profit_Margin'] = pm
                else:
                    df['Profit_Margin'] = 0
            else:
                df['Profit_Margin'] = 0
            
            if fundamentals.get('debt_to_equity') and isinstance(fundamentals.get('debt_to_equity'), (int, float)):
                de = fundamentals['debt_to_equity']
                if np.isfinite(de):
                    df['Debt_to_Equity'] = de
                else:
                    df['Debt_to_Equity'] = 0
            else:
                df['Debt_to_Equity'] = 0
            
            if fundamentals.get('beta') and isinstance(fundamentals.get('beta'), (int, float)):
                beta = fundamentals['beta']
                if np.isfinite(beta):
                    df['Beta'] = beta
                else:
                    df['Beta'] = 1
            else:
                df['Beta'] = 1
            
            if fundamentals.get('dividend_yield') and isinstance(fundamentals.get('dividend_yield'), (int, float)):
                dy = fundamentals['dividend_yield']
                if np.isfinite(dy):
                    df['Dividend_Yield'] = dy
                else:
                    df['Dividend_Yield'] = 0
            else:
                df['Dividend_Yield'] = 0
        
        # Final cleanup
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        # Replace any NaN values with 0
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        return df


def create_target_variable(df: pd.DataFrame, periods_ahead: int = 1) -> pd.DataFrame:
    """
    Create target variable for prediction
    
    Args:
        df: DataFrame with price data
        periods_ahead: Number of periods to predict ahead
    
    Returns:
        DataFrame with target variable
    """
    try:
        # Target is future close price shifted back
        df['Target'] = df['Close'].shift(-periods_ahead)
        
        # Drop last rows where target is NaN
        df = df.dropna()
        
        # Remove any rows with inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error creating target variable: {str(e)}")
        return df


def get_feature_importance_names(df: pd.DataFrame) -> list:
    """
    Get list of feature column names (excluding target and date)
    
    Args:
        df: DataFrame with features
    
    Returns:
        List of feature names
    """
    exclude = ['Date', 'Target', 'Datetime', 'DateTime', 'Index', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    feature_cols = [col for col in df.columns if col not in exclude]
    
    return feature_cols
