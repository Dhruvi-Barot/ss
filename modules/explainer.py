"""
SHAP-based model explainability module
"""

import numpy as np
import pandas as pd
import streamlit as st
import shap


def generate_shap_explanations(model, X_input: np.ndarray, feature_names: list, X_train: np.ndarray = None) -> dict:
    """
    Generate SHAP explanations for model predictions
    
    Args:
        model: Trained XGBoost model
        X_input: Input features for explanation
        feature_names: Names of features
        X_train: Training data for explainer
    
    Returns:
        Dictionary with SHAP values and explanations
    """
    try:
        # Create explainer
        if X_train is not None:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
        
        # Get base value (expected value)
        base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
        
        # Ensure shap_values is 1D (for regression)
        if isinstance(shap_values, list):
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        shap_values = shap_values.flatten() if isinstance(shap_values, np.ndarray) else shap_values
        
        # Get feature importance (mean absolute SHAP values)
        if len(X_input.shape) == 1:
            feature_importance = np.abs(shap_values)
        else:
            feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance ranking
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        return {
            'shap_values': shap_values,
            'base_value': base_value,
            'feature_importance': importance_df,
            'top_features': importance_df.head(10)
        }
        
    except Exception as e:
        st.warning(f"Error generating SHAP explanations: {str(e)}")
        return None


def categorize_features(feature_names: list) -> dict:
    """
    Categorize features into groups
    
    Args:
        feature_names: List of feature names
    
    Returns:
        Dictionary with feature categories
    """
    categories = {
        'Technical_Indicators': [],
        'Price_Action': [],
        'Fundamentals': [],
        'Volume': [],
        'Volatility': [],
        'Momentum': []
    }
    
    technical_keywords = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'ADX', 'Stoch', 'ROC']
    price_keywords = ['Close', 'Open', 'High', 'Low', 'Price', 'Ratio', 'Lag']
    fundamental_keywords = ['PE', 'Market', 'ROE', 'Profit', 'Debt', 'Beta', 'Dividend']
    volume_keywords = ['Volume', 'OBV', 'VWAP']
    volatility_keywords = ['Volatility', 'ATR']
    momentum_keywords = ['RSI', 'MACD', 'ROC', 'Stoch']
    
    for feature in feature_names:
        categorized = False
        
        for keyword in momentum_keywords:
            if keyword in feature:
                categories['Momentum'].append(feature)
                categorized = True
                break
        
        if not categorized:
            for keyword in technical_keywords:
                if keyword in feature:
                    categories['Technical_Indicators'].append(feature)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in volume_keywords:
                if keyword in feature:
                    categories['Volume'].append(feature)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in volatility_keywords:
                if keyword in feature:
                    categories['Volatility'].append(feature)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in fundamental_keywords:
                if keyword in feature:
                    categories['Fundamentals'].append(feature)
                    categorized = True
                    break
        
        if not categorized:
            for keyword in price_keywords:
                if keyword in feature:
                    categories['Price_Action'].append(feature)
                    categorized = True
                    break
        
        if not categorized:
            categories['Price_Action'].append(feature)
    
    return categories


def get_grouped_feature_importance(shap_values: np.ndarray, feature_names: list) -> dict:
    """
    Get feature importance grouped by category
    
    Args:
        shap_values: Array of SHAP values
        feature_names: List of feature names
    
    Returns:
        Dictionary with grouped importance scores
    """
    categories = categorize_features(feature_names)
    
    grouped_importance = {}
    
    for category, features in categories.items():
        if not features:
            continue
        
        # Get indices of features in this category
        indices = [i for i, fname in enumerate(feature_names) if fname in features]
        
        # Calculate total importance
        if indices:
            importance = np.abs(shap_values[indices]).sum()
            grouped_importance[category] = importance
    
    # Normalize to percentage
    total_importance = sum(grouped_importance.values())
    if total_importance > 0:
        grouped_importance = {k: (v/total_importance)*100 for k, v in grouped_importance.items()}
    
    return grouped_importance


def generate_prediction_explanation(prediction: float, current_price: float, 
                                   shap_values: np.ndarray, feature_names: list) -> str:
    """
    Generate human-readable explanation for prediction
    
    Args:
        prediction: Predicted price
        current_price: Current price
        shap_values: SHAP values
        feature_names: Feature names
    
    Returns:
        Explanation string
    """
    price_change = ((prediction - current_price) / current_price) * 100
    
    # Get top contributing features
    top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
    top_features = [(feature_names[i], shap_values[i]) for i in top_indices]
    
    # Build explanation
    if price_change > 2:
        base_text = f"The model predicts a significant **UPWARD** movement of **{price_change:.2f}%**"
    elif price_change < -2:
        base_text = f"The model predicts a significant **DOWNWARD** movement of **{price_change:.2f}%**"
    elif price_change > 0.5:
        base_text = f"The model predicts a slight **UPWARD** movement of **{price_change:.2f}%**"
    elif price_change < -0.5:
        base_text = f"The model predicts a slight **DOWNWARD** movement of **{price_change:.2f}%**"
    else:
        base_text = f"The model predicts a **STABLE** trend with **{price_change:.2f}%** change"
    
    explanation = base_text + "\n\n**Key Factors:**\n"
    
    for i, (feature, value) in enumerate(top_features, 1):
        direction = "↑" if value > 0 else "↓"
        explanation += f"{i}. {feature} {direction}\n"
    
    return explanation
