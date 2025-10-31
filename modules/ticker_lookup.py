"""
Ticker symbol lookup using Perplexity API
Fallback to direct search if API not configured
"""

import os
import re
from dotenv import load_dotenv
import requests
import streamlit as st

# Load environment variables
load_dotenv()


def search_ticker_perplexity(company_name: str) -> dict:
    """
    Search for ticker symbol using Perplexity API
    
    Args:
        company_name: Company name (e.g., "Reliance", "Bank of Baroda")
    
    Returns:
        Dictionary with ticker, company name, and confidence
    """
    api_key = os.getenv('PERPLEXITY_API_KEY')
    
    if not api_key:
        st.warning("⚠️ Perplexity API key not configured. Using fallback method.")
        return search_ticker_fallback(company_name)
    
    try:
        url = "https://api.perplexity.ai/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Find the exact Yahoo Finance ticker symbol for "{company_name}" 
        on the Indian stock market (NSE or BSE). Return ONLY the ticker symbol with 
        .NS or .BO suffix in this exact format: TICKER.NS or TICKER.BO
        
        Example responses:
        - RELIANCE.NS
        - BANKBARODA.NS
        - TATAMOTORS.NS
        
        Return only the ticker, nothing else."""
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial data assistant. Return only ticker symbols."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        ticker_raw = result['choices'][0]['message']['content'].strip()
        
        # Extract ticker using regex
        ticker_match = re.search(r'([A-Z0-9]+\.(NS|BO))', ticker_raw)
        
        if ticker_match:
            ticker = ticker_match.group(1)
            return {
                'ticker': ticker,
                'company': company_name,
                'confidence': 'high',
                'source': 'Perplexity API'
            }
        else:
            st.warning("Could not parse ticker from API response. Using fallback.")
            return search_ticker_fallback(company_name)
            
    except Exception as e:
        st.warning(f"Perplexity API error: {str(e)}. Using fallback method.")
        return search_ticker_fallback(company_name)


def search_ticker_fallback(company_name: str) -> dict:
    """
    Fallback method: Direct ticker mapping for common Indian stocks
    
    Args:
        company_name: Company name
    
    Returns:
        Dictionary with ticker information
    """
    # Common Indian stock mappings
    ticker_map = {
        'reliance': 'RELIANCE.NS',
        'tcs': 'TCS.NS',
        'infosys': 'INFY.NS',
        'hdfc bank': 'HDFCBANK.NS',
        'hdfcbank': 'HDFCBANK.NS',
        'icici bank': 'ICICIBANK.NS',
        'icicibank': 'ICICIBANK.NS',
        'state bank': 'SBIN.NS',
        'sbi': 'SBIN.NS',
        'bajaj finance': 'BAJFINANCE.NS',
        'bharti airtel': 'BHARTIARTL.NS',
        'airtel': 'BHARTIARTL.NS',
        'itc': 'ITC.NS',
        'larsen': 'LT.NS',
        'lt': 'LT.NS',
        'asian paints': 'ASIANPAINT.NS',
        'maruti': 'MARUTI.NS',
        'tata motors': 'TATAMOTORS.NS',
        'tatamotors': 'TATAMOTORS.NS',
        'mahindra': 'M&M.NS',
        'wipro': 'WIPRO.NS',
        'hcl': 'HCLTECH.NS',
        'axis bank': 'AXISBANK.NS',
        'axisbank': 'AXISBANK.NS',
        'kotak': 'KOTAKBANK.NS',
        'sun pharma': 'SUNPHARMA.NS',
        'dr reddy': 'DRREDDY.NS',
        'titan': 'TITAN.NS',
        'ultratech': 'ULTRACEMCO.NS',
        'nestle': 'NESTLEIND.NS',
        'hindustan unilever': 'HINDUNILVR.NS',
        'hul': 'HINDUNILVR.NS',
        'adani': 'ADANIENT.NS',
        'power grid': 'POWERGRID.NS',
        'ntpc': 'NTPC.NS',
        'ongc': 'ONGC.NS',
        'coal india': 'COALINDIA.NS',
        'ioc': 'IOC.NS',
        'bpcl': 'BPCL.NS',
        'gail': 'GAIL.NS',
        'bank of baroda': 'BANKBARODA.NS',
        'bankbaroda': 'BANKBARODA.NS',
        'punjab national': 'PNB.NS',
        'pnb': 'PNB.NS',
        'canara bank': 'CANBK.NS',
        'bank of india': 'BANKINDIA.NS',
        'union bank': 'UNIONBANK.NS',
        'jindal steel': 'JINDALSTEL.NS',
        'tata steel': 'TATASTEEL.NS',
        'vedanta': 'VEDL.NS',
        'hindalco': 'HINDALCO.NS',
        'grasim': 'GRASIM.NS',
        'upl': 'UPL.NS',
        'shree cement': 'SHREECEM.NS',
        'ambuja': 'AMBUJACEM.NS',
        'acc': 'ACC.NS',
        'britannia': 'BRITANNIA.NS',
        'dabur': 'DABUR.NS',
        'godrej': 'GODREJCP.NS',
        'bajaj auto': 'BAJAJ-AUTO.NS',
        'hero motocorp': 'HEROMOTOCO.NS',
        'eicher': 'EICHERMOT.NS',
        'tvsmotor': 'TVSMOTOR.NS',
        'havells': 'HAVELLS.NS',
        'siemens': 'SIEMENS.NS',
        'abb': 'ABB.NS',
        'bhel': 'BHEL.NS',
        'divi': 'DIVISLAB.NS',
        'cipla': 'CIPLA.NS',
        'lupin': 'LUPIN.NS',
        'biocon': 'BIOCON.NS',
        'apollo hospitals': 'APOLLOHOSP.NS',
        'fortis': 'FORTIS.NS',
        'indigo': 'INDIGO.NS',
        'spicejet': 'SPICEJET.NS',
        'irctc': 'IRCTC.NS',
        'dlf': 'DLF.NS',
        'oberoi': 'OBEROIRLTY.NS',
        'godrej properties': 'GODREJPROP.NS',
        'brigade': 'BRIGADE.NS',
        'tech mahindra': 'TECHM.NS',
        'mindtree': 'LTIM.NS',
        'mphasis': 'MPHASIS.NS',
        'persistent': 'PERSISTENT.NS',
        'zee': 'ZEEL.NS',
        'pvr': 'PVR.NS',
        'dish tv': 'DISHTV.NS'
    }
    
    # Normalize company name
    company_lower = company_name.lower().strip()
    
    # Direct lookup
    if company_lower in ticker_map:
        return {
            'ticker': ticker_map[company_lower],
            'company': company_name,
            'confidence': 'high',
            'source': 'Direct Mapping'
        }
    
    # Partial matching
    for key, ticker in ticker_map.items():
        if key in company_lower or company_lower in key:
            return {
                'ticker': ticker,
                'company': company_name,
                'confidence': 'medium',
                'source': 'Partial Match'
            }
    
    # If no match, assume user entered ticker directly
    ticker = company_name.strip().upper()
    if not ticker.endswith(('.NS', '.BO')):
        ticker = f"{ticker}.NS"
    
    return {
        'ticker': ticker,
        'company': company_name,
        'confidence': 'assumed',
        'source': 'Direct Input'
    }


def get_ticker_info(search_query: str) -> dict:
    """
    Main function to get ticker information
    
    Args:
        search_query: Company name or ticker
    
    Returns:
        Dictionary with ticker information
    """
    if not search_query or not search_query.strip():
        return None
    
    # Try Perplexity first, falls back automatically if needed
    return search_ticker_perplexity(search_query)
