"""
Model training module with XGBoost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import streamlit as st
from datetime import datetime
import json

from modules.data_fetcher import fetch_historical_data, fetch_fundamental_data
from modules.feature_engineering import (add_technical_indicators, prepare_features_for_model, 
                                         create_target_variable, get_feature_importance_names)
from utils.helpers import ensure_directories, save_metadata


class StockPredictor:
    """XGBoost model for stock price prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_data(self, ticker: str, days_back: int = 365) -> tuple:
        """
        Prepare data for model training
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days of historical data
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        try:
            # Fetch historical data
            df = fetch_historical_data(ticker, period="2y")
            
            if df is None or df.empty:
                raise ValueError(f"No data available for {ticker}")
            
            st.info(f"📥 Fetched {len(df)} rows of historical data")
            
            # Fetch fundamental data
            fundamentals = fetch_fundamental_data(ticker)
            
            # Add technical indicators
            df = add_technical_indicators(df)
            st.info(f"✅ Added technical indicators")
            
            # Prepare features
            df = prepare_features_for_model(df, fundamentals)
            st.info(f"✅ Prepared features")
            
            # Create target variable (predict next day's close)
            df = create_target_variable(df, periods_ahead=1)
            st.info(f"✅ Created target variable - {len(df)} valid samples")
            
            # Remove NaN values
            df = df.dropna()
            
            if df.empty:
                raise ValueError("No valid data after feature engineering")
            
            # Get feature names
            self.feature_names = get_feature_importance_names(df)
            st.info(f"✅ Using {len(self.feature_names)} features")
            
            # Prepare X and y
            X = df[self.feature_names].values
            y = df['Target'].values
            
            # CRITICAL: Check for inf and nan values BEFORE scaling
            st.info("🔍 Validating data for inf/nan values...")
            
            # Replace inf with nan
            X = np.where(np.isinf(X), np.nan, X)
            y = np.where(np.isinf(y), np.nan, y)
            
            # Get valid rows (no nan, no inf)
            valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                raise ValueError("No valid data after removing inf/nan values")
            
            st.info(f"✅ Valid samples: {len(X)} (removed {len(df) - len(X)} invalid rows)")
            
            # Normalize features
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            
            # Final validation after scaling
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError("Data contains inf/nan after scaling")
            
            st.info(f"✅ Data normalized and validated")
            
            return X, y, self.feature_names
            
        except Exception as e:
            st.error(f"❌ Error preparing data: {str(e)}")
            raise
    
    def train(self, ticker: str, days_back: int = 365):
        """
        Train the XGBoost model
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days of historical data
        """
        try:
            st.info("📊 Preparing data...")
            X, y, feature_names = self.prepare_data(ticker, days_back)
            
            # Split data
            st.info("📊 Splitting data into train/test...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            st.info(f"📊 Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            st.info("🤖 Training XGBoost model...")
            
            # Train XGBoost
            self.model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=0      # Remove early_stopping_rounds for compat!
            )
            
            # Evaluate
            st.info("📈 Evaluating model...")
            y_pred = self.model.predict(X_test)
            
            # Validate predictions
            if not np.isfinite(y_pred).all():
                raise ValueError("Model predictions contain inf/nan values")
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            st.success(f"✅ Model trained successfully!")
            st.info(f"📈 **MAE:** ₹{mae:.2f} | **RMSE:** ₹{rmse:.2f} | **R²:** {r2:.4f}")
            
            # Save model
            self.save_model(ticker)
            self.is_trained = True
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'samples': len(X_train),
                'features': len(feature_names)
            }
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            raise
    
    def predict(self, X_input: np.ndarray) -> float:
        """
        Make prediction
        
        Args:
            X_input: Input features (unnormalized)
        
        Returns:
            Predicted price
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.scaler is None:
            raise ValueError("Scaler not initialized")
        
        try:
            # Reshape input
            X_input = np.array(X_input).reshape(1, -1)
            
            # Validate input
            if not np.isfinite(X_input).all():
                st.warning("⚠️ Input contains inf/nan, using default prediction")
                return 0
            
            # Normalize input
            X_normalized = self.scaler.transform(X_input)
            
            # Validate normalized input
            if not np.isfinite(X_normalized).all():
                st.warning("⚠️ Normalized input contains inf/nan")
                return 0
            
            # Predict
            prediction = self.model.predict(X_normalized)[0]
            
            # Validate prediction
            if not np.isfinite(prediction):
                st.warning("⚠️ Prediction is inf/nan, using default")
                return 0
            
            return float(prediction)
            
        except Exception as e:
            st.warning(f"⚠️ Prediction error: {str(e)}")
            return 0
    
    def save_model(self, ticker: str):
        """
        Save model to disk
        
        Args:
            ticker: Stock ticker symbol
        """
        try:
            ensure_directories()
            
            model_path = Path('models') / f'{ticker}_model.joblib'
            scaler_path = Path('models') / f'{ticker}_scaler.joblib'
            features_path = Path('models') / f'{ticker}_features.joblib'
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.feature_names, features_path)
            
            # Save metadata
            metadata = {
                'ticker': ticker,
                'last_trained': datetime.now().isoformat(),
                'model_type': 'XGBoost',
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'features': self.feature_names
            }
            
            save_metadata(ticker, metadata)
            st.success(f"✅ Model saved for {ticker}")
            
        except Exception as e:
            st.warning(f"⚠️ Error saving model: {str(e)}")
    
    def load_model(self, ticker: str) -> bool:
        """
        Load model from disk
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Boolean indicating success
        """
        try:
            model_path = Path('models') / f'{ticker}_model.joblib'
            scaler_path = Path('models') / f'{ticker}_scaler.joblib'
            features_path = Path('models') / f'{ticker}_features.joblib'
            
            if not model_path.exists():
                return False
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Error loading model: {str(e)}")
            return False


def get_or_train_model(ticker: str, force_retrain: bool = False) -> StockPredictor:
    """
    Get existing model or train new one
    
    Args:
        ticker: Stock ticker symbol
        force_retrain: Force retraining even if model exists
    
    Returns:
        Trained StockPredictor instance
    """
    predictor = StockPredictor()
    
    # Try to load existing model
    if not force_retrain and predictor.load_model(ticker):
        st.success(f"✅ Loaded cached model for {ticker}")
        return predictor
    
    # Train new model
    st.info(f"🚀 Training new model for {ticker}...")
    predictor.train(ticker)
    return predictor
