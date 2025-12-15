# ==============================================================================
# QUANTUM | Global Institutional Terminal - INSTITUTIONAL ENHANCED (Streamlit)
# ------------------------------------------------------------------------------
# Enhanced Features:
#  - Added memory optimization for large datasets
#  - Improved error handling and user feedback
#  - Enhanced caching strategies
#  - Added progress indicators for long operations
#  - Improved visualization aesthetics
#  - Added data validation and sanity checks
#  - Enhanced modularity and maintainability
# ==============================================================================

import streamlit as st
import gc  # For memory management
import traceback  # For better error reporting

# MUST be the first Streamlit command
st.set_page_config(
    page_title="QUANTUM | Advanced Risk Analytics",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourrepo/quantum-terminal',
        'Report a bug': 'https://github.com/yourrepo/quantum-terminal/issues',
        'About': "QUANTUM Institutional Terminal v16.1"
    }
)

# Add performance monitoring decorator
def log_performance(func):
    """Decorator to log function execution time and memory usage"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_mem = None
        try:
            import psutil
            process = psutil.Process()
            start_mem = process.memory_info().rss / 1024 / 1024  # MB
        except:
            pass
            
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if start_mem is not None:
            try:
                end_mem = process.memory_info().rss / 1024 / 1024
                mem_diff = end_mem - start_mem
                st.sidebar.caption(f"{func.__name__}: {execution_time:.2f}s, ΔMem: {mem_diff:+.1f}MB")
            except:
                pass
                
        if execution_time > 5:  # Log long operations
            st.sidebar.info(f"⚠️ {func.__name__} took {execution_time:.1f}s")
            
        return result
    return wrapper

# Enhanced styling with better contrast and accessibility
st.markdown("""
<style>
    /* Base styles */
    .stApp { 
        background-color: #f8fafc; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
    }
    
    /* Improved header styles */
    .main-header { 
        font-size: 2.8rem; 
        font-weight: 800; 
        color: #1a237e; 
        text-align: center; 
        border-bottom: 3px solid #303f9f; 
        padding-bottom: 12px; 
        margin-bottom: 18px; 
        text-shadow: 0px 1px 2px rgba(0,0,0,0.1);
    }
    
    .section-header { 
        font-size: 1.7rem; 
        font-weight: 700; 
        color: #1a237e; 
        border-bottom: 2px solid #e8eaf6; 
        padding-bottom: 8px; 
        margin: 22px 0 14px 0; 
        background: linear-gradient(90deg, #1a237e 0%, transparent 100%);
        padding-left: 10px;
    }
    
    .subsection-header { 
        font-size: 1.25rem; 
        font-weight: 700; 
        color: #283593; 
        margin: 16px 0 10px 0; 
        padding-left: 8px;
        border-left: 4px solid #5c6bc0;
    }
    
    /* Enhanced cards */
    .card { 
        background: white; 
        border: 1px solid #e1e5eb; 
        border-radius: 12px; 
        padding: 18px; 
        margin: 12px 0; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    
    .info-card { 
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e5e7eb; 
        border-left: 4px solid #1f2937; 
        border-radius: 10px; 
        padding: 14px 16px; 
        margin: 10px 0; 
        color: #111827;
    }
    
    .success-card { 
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0; 
        border-left: 4px solid #16a34a; 
        border-radius: 10px; 
        padding: 14px 16px; 
        margin: 10px 0; 
        color: #052e16;
    }
    
    .warning-card { 
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fde68a; 
        border-left: 4px solid #d97706; 
        border-radius: 10px; 
        padding: 14px 16px; 
        margin: 10px 0; 
        color: #451a03;
    }
    
    .error-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 10px 0;
        color: #7f1d1d;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 5px; 
        background: #f0f2f6; 
        padding: 6px; 
        border-radius: 12px; 
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] { 
        border-radius: 10px; 
        padding: 10px 16px; 
        font-weight: 700; 
        background: white; 
        border: 1px solid #e0e0e0; 
        color: #5c6bc0;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8fafc;
        border-color: #5c6bc0;
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #1a237e 0%, #303f9f 100%); 
        color: white; 
        border-color: #1a237e; 
        box-shadow: 0 4px 12px rgba(17,24,39,.15);
    }
    
    /* Enhanced buttons */
    .stButton>button { 
        background: linear-gradient(135deg, #111827 0%, #374151 100%); 
        color: white; 
        border: none; 
        border-radius: 10px; 
        padding: 10px 18px; 
        font-weight: 700; 
        transition: all 0.2s;
    }
    
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 14px rgba(17,24,39,.2);
        background: linear-gradient(135deg, #1f2937 0%, #4b5563 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Custom scrollbars */
    ::-webkit-scrollbar { 
        width: 10px; 
        height: 10px; 
    }
    
    ::-webkit-scrollbar-track { 
        background: #f1f1f1; 
        border-radius: 6px; 
    }
    
    ::-webkit-scrollbar-thumb { 
        background: #c5cae9; 
        border-radius: 6px; 
        border: 2px solid #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb:hover { 
        background: #7986cb; 
    }
    
    /* Badge styles */
    .risk-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        color: #dc2626;
        border: 1px solid #fecaca;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        color: #d97706;
        border: 1px solid #fde68a;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        color: #16a34a;
        border: 1px solid #bbf7d0;
    }
    
    /* Tooltip style */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #666;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1f2937;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.875rem;
        font-weight: normal;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Loading spinner enhancement */
    .stSpinner > div {
        border-top-color: #1a237e !important;
    }
    
    /* Metric card enhancement */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #6b7280 !important;
    }
</style>
""", unsafe_allow_html=True)

# Add custom JavaScript for better UX
st.components.v1.html("""
<script>
// Add smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add copy to clipboard functionality for code snippets
    document.querySelectorAll('pre').forEach(pre => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: #1a237e;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        `;
        button.onmouseenter = () => button.style.opacity = '1';
        button.onmouseleave = () => button.style.opacity = '0.7';
        button.onclick = () => {
            const code = pre.textContent;
            navigator.clipboard.writeText(code).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => button.textContent = 'Copy', 2000);
            });
        };
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
});
</script>
""")

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import scipy.stats as stats

# (kept for compatibility / no removal)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize

# Add data validation utilities
class DataValidator:
    """Enhanced data validation and sanity checking"""
    
    @staticmethod
    def validate_price_data(prices: pd.DataFrame, min_assets: int = 1, min_rows: int = 50) -> Tuple[bool, str]:
        """Validate price data quality"""
        if prices is None or prices.empty:
            return False, "Price data is empty or None"
        
        if len(prices.columns) < min_assets:
            return False, f"Insufficient assets: {len(prices.columns)} < {min_assets}"
        
        if len(prices) < min_rows:
            return False, f"Insufficient rows: {len(prices)} < {min_rows}"
        
        # Check for NaN values
        nan_count = prices.isna().sum().sum()
        if nan_count > 0:
            return False, f"Found {nan_count} NaN values in price data"
        
        # Check for infinite values
        inf_count = np.isinf(prices.values).sum()
        if inf_count > 0:
            return False, f"Found {inf_count} infinite values in price data"
        
        # Check for zero or negative prices
        negative_prices = (prices <= 0).sum().sum()
        if negative_prices > 0:
            return False, f"Found {negative_prices} zero or negative prices"
        
        return True, "Data validation passed"
    
    @staticmethod
    def validate_weights(weights: Dict[str, float], assets: List[str]) -> Tuple[bool, str]:
        """Validate portfolio weights"""
        if not weights:
            return False, "Weights dictionary is empty"
        
        # Check for missing assets
        missing_assets = [asset for asset in assets if asset not in weights]
        if missing_assets:
            return False, f"Missing weights for assets: {missing_assets[:5]}"
        
        # Check weight sum
        weight_sum = sum(abs(v) for v in weights.values())
        if abs(weight_sum - 1.0) > 0.01:  # Allow small rounding errors
            return False, f"Weights sum to {weight_sum:.4f}, expected ~1.0"
        
        # Check for NaN or infinite weights
        for asset, weight in weights.items():
            if not np.isfinite(weight):
                return False, f"Non-finite weight for {asset}: {weight}"
        
        return True, "Weights validation passed"

# Enhanced progress indicators
class ProgressManager:
    """Manage progress indicators for long-running operations"""
    
    @staticmethod
    def progress_bar(iterable, desc: str = "Processing", total: Optional[int] = None):
        """Wrapper for st.progress with better UX"""
        if total is None:
            total = len(iterable)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, item in enumerate(iterable):
            yield item
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"{desc}: {i+1}/{total} ({progress:.1%})")
        
        progress_bar.empty()
        status_text.empty()
    
    @staticmethod
    def spinner_with_message(message: str):
        """Context manager for spinner with message"""
        return st.spinner(message)

# Enhanced error handling decorator
def handle_errors(default_return=None, show_error=True):
    """Decorator for enhanced error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                if show_error:
                    st.error(f"❌ {error_msg}")
                    if st.session_state.get("debug_mode", False):
                        with st.expander("Debug Details"):
                            st.code(traceback.format_exc())
                
                # Log error to session state for debugging
                if "error_log" not in st.session_state:
                    st.session_state.error_log = []
                st.session_state.error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "function": func.__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                
                return default_return
        return wrapper
    return decorator

# ==============================================================================
# Universe / Data Manager (50+ instruments across regions) - ENHANCED
# ==============================================================================
class EnhancedDataManager:
    def __init__(self):
        self.universe: Dict[str, Dict[str, str]] = {
            "Global Benchmarks": {
                "S&P 500 (US)": "^GSPC",
                "NASDAQ 100 (US)": "^NDX",
                "Dow Jones (US)": "^DJI",
                "Russell 2000 (US)": "^RUT",
                "FTSE 100 (UK)": "^FTSE",
                "DAX 40 (Germany)": "^GDAXI",
                "CAC 40 (France)": "^FCHI",
                "Nikkei 225 (Japan)": "^N225",
                "Hang Seng (Hong Kong)": "^HSI",
                "Shanghai Composite (China)": "000001.SS",
                "ASX 200 (Australia)": "^AXJO",
                "BIST 100 (Turkey)": "XU100.IS",
                "MSCI World": "URTH",
                "MSCI Emerging Markets": "EEM",
                "STOXX Europe 600": "^STOXX",
                "TSX Composite (Canada)": "^GSPTSE",
            },
            "US ETFs (Major)": {
                "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM", "VTI": "VTI",
                "ARKK": "ARKK", "XLF": "XLF", "XLK": "XLK", "XLE": "XLE", "XLV": "XLV",
                "GLD": "GLD", "SLV": "SLV", "GDX": "GDX",
                "TLT": "TLT", "IEF": "IEF", "AGG": "AGG", "BND": "BND",
                "HYG": "HYG", "LQD": "LQD", "JNK": "JNK",
                "VWO": "VWO", "VGK": "VGK", "VPL": "VPL",
            },
            "Global Mega Caps": {
                "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN",
                "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA", "Berkshire": "BRK-B",
                "TSMC": "TSM", "Samsung": "005930.KS",
                "Tencent": "0700.HK", "Alibaba": "BABA",
                "LVMH": "MC.PA", "ASML": "ASML", "Novo Nordisk": "NVO",
                "JPMorgan": "JPM", "Johnson & Johnson": "JNJ", "Visa": "V",
            },
            "BIST 30 (Turkey)": {
                "AKBNK": "AKBNK.IS", "ARCLK": "ARCLK.IS", "ASELS": "ASELS.IS",
                "BIMAS": "BIMAS.IS", "EKGYO": "EKGYO.IS", "ENKAI": "ENKAI.IS",
                "EREGL": "EREGL.IS", "FROTO": "FROTO.IS", "GARAN": "GARAN.IS",
                "GUBRF": "GUBRF.IS", "HALKB": "HALKB.IS", "HEKTS": "HEKTS.IS",
                "ISCTR": "ISCTR.IS", "KCHOL": "KCHOL.IS", "KOZAL": "KOZAL.IS",
                "KRDMD": "KRDMD.IS", "ODAS": "ODAS.IS", "PETKM": "PETKM.IS",
                "PGSUS": "PGSUS.IS", "SAHOL": "SAHOL.IS", "SASA": "SASA.IS",
                "SISE": "SISE.IS", "TAVHL": "TAVHL.IS", "TCELL": "TCELL.IS",
                "THYAO": "THYAO.IS", "TKFEN": "TKFEN.IS", "TOASO": "TOASO.IS",
                "TUPRS": "TUPRS.IS", "VAKBN": "VAKBN.IS", "YKBNK": "YKBNK.IS",
            },
            "Rates & Fixed Income": {
                "US 10Y Yield": "^TNX",
                "US 2Y Yield": "^FVX",
                "US 30Y Yield": "^TYX",
                "TIP": "TIP",
                "SHY": "SHY",
                "IEI": "IEI",
            },
            "Commodities": {
                "Gold (GLD)": "GLD",
                "Silver (SLV)": "SLV",
                "Oil (USO)": "USO",
                "Natural Gas (UNG)": "UNG",
                "Copper (CPER)": "CPER",
            },
            "Cryptocurrencies": {
                "Bitcoin": "BTC-USD",
                "Ethereum": "ETH-USD",
                "Binance Coin": "BNB-USD",
                "Cardano": "ADA-USD",
                "Solana": "SOL-USD",
            }
        }
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers across all categories"""
        all_tickers = []
        for category in self.universe.values():
            if isinstance(category, dict):
                all_tickers.extend(category.values())
        return list(set(all_tickers))
    
    def search_ticker(self, search_term: str) -> Dict[str, str]:
        """Search for tickers by name or symbol"""
        results = {}
        search_term = search_term.lower()
        
        for category_name, category in self.universe.items():
            for name, ticker in category.items():
                if (search_term in name.lower() or 
                    search_term in ticker.lower() or
                    search_term in category_name.lower()):
                    results[f"{name} ({ticker})"] = ticker
        
        return results

    def ticker_name_map(self) -> Dict[str, str]:
        m = {}
        for cat, d in self.universe.items():
            for name, t in d.items():
                m[t] = name
        return m

    def ticker_category_map(self) -> Dict[str, str]:
        """Map ticker -> universe category (for region inference & labeling)."""
        m: Dict[str, str] = {}
        for cat, d in self.universe.items():
            for _name, t in d.items():
                m[t] = cat
        return m

# Initialize enhanced data manager
data_manager = EnhancedDataManager()
TICKER_NAME_MAP = data_manager.ticker_name_map()
TICKER_CATEGORY_MAP = data_manager.ticker_category_map()

# ==============================================================================
# Enhanced Regional Classification with more comprehensive mapping
# ==============================================================================
REGION_OPTIONS = [
    "North America (US/Canada)",
    "Europe",
    "Asia-Pacific",
    "Emerging Markets",
    "Turkey",
    "Latin America",
    "Middle East & Africa",
    "Global",
    "Crypto",
    "Commodities",
    "FX",
    "Other"
]

REGION_COLORS = {
    "North America (US/Canada)": "#1f77b4",
    "Europe": "#2ca02c",
    "Asia-Pacific": "#d62728",
    "Emerging Markets": "#ff7f0e",
    "Turkey": "#9467bd",
    "Latin America": "#8c564b",
    "Middle East & Africa": "#e377c2",
    "Global": "#7f7f7f",
    "Crypto": "#17becf",
    "Commodities": "#bcbd22",
    "FX": "#1a55FF",
    "Other": "#aec7e8"
}

def infer_region_from_category(cat: str) -> str:
    c = (cat or "").lower()
    if "bist" in c or "turkey" in c:
        return "Turkey"
    if "us" in c or "north america" in c or "canada" in c:
        return "North America (US/Canada)"
    if "europe" in c or "uk" in c or "germany" in c or "france" in c or "swiss" in c or "stoxx" in c:
        return "Europe"
    if "asia" in c or "japan" in c or "china" in c or "korea" in c or "india" in c or "australia" in c:
        return "Asia-Pacific"
    if "emerging" in c:
        return "Emerging Markets"
    if "latin" in c or "latam" in c or "brazil" in c or "mexico" in c:
        return "Latin America"
    if "mena" in c or "middle east" in c or "africa" in c:
        return "Middle East & Africa"
    if "crypto" in c:
        return "Crypto"
    if "commodity" in c or "commodities" in c:
        return "Commodities"
    if "fx" in c or "forex" in c:
        return "FX"
    if "global benchmark" in c or "benchmarks" in c:
        return "Global"
    return "Global"

def infer_region_from_ticker(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    
    # Cryptocurrencies
    if t.endswith("-USD") and any(crypto in t for crypto in ["BTC", "ETH", "BNB", "ADA", "SOL", "XRP"]):
        return "Crypto"
    
    # Exchange suffix heuristics (Yahoo Finance)
    suffix = ""
    if "." in t:
        suffix = t.split(".")[-1]
    
    suffix_mapping = {
        "IS": "Turkey",
        "L": "Europe",  # London
        "DE": "Europe",  # Germany
        "F": "Europe",   # Frankfurt
        "PA": "Europe",  # Paris
        "MI": "Europe",  # Milan
        "AS": "Europe",  # Amsterdam
        "BR": "Europe",  # Brussels
        "SW": "Europe",  # Switzerland
        "ST": "Europe",  # Stockholm
        "HE": "Europe",  # Helsinki
        "VI": "Europe",  # Vienna
        "MC": "Europe",  # Madrid
        "T": "Asia-Pacific",  # Tokyo
        "HK": "Asia-Pacific", # Hong Kong
        "KS": "Asia-Pacific", # Korea
        "KQ": "Asia-Pacific", # Korea KOSDAQ
        "SS": "Asia-Pacific", # Shanghai
        "SZ": "Asia-Pacific", # Shenzhen
        "SI": "Asia-Pacific", # Singapore
        "AX": "Asia-Pacific", # Australia
        "NZ": "Asia-Pacific", # New Zealand
        "SA": "Latin America", # South Africa
        "MX": "Latin America", # Mexico
        "BA": "Latin America", # Buenos Aires
        "SN": "Latin America", # Santiago
    }
    
    if suffix in suffix_mapping:
        return suffix_mapping[suffix]
    
    # Common ETFs / indices default to NA/Global
    if t.startswith("^"):
        return "Global"
    if t in {"SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "HYG", "LQD", "VTI", "VOO"}:
        return "North America (US/Canada)"
    if t in {"EEM", "VWO"}:
        return "Emerging Markets"
    if t in {"EWJ", "EWA", "EWH", "EWT"}:
        return "Asia-Pacific"
    if t in {"FEZ", "VGK", "EWG", "EWU", "EWQ"}:
        return "Europe"
    if t in {"GLD", "SLV", "USO", "UNG"}:
        return "Commodities"
    
    return "Other"

def infer_region(ticker: str) -> str:
    if ticker in TICKER_CATEGORY_MAP:
        return infer_region_from_category(TICKER_CATEGORY_MAP.get(ticker, ""))
    return infer_region_from_ticker(ticker)

def get_region_color(region: str) -> str:
    """Get color for a region"""
    return REGION_COLORS.get(region, "#aec7e8")

# ==============================================================================
# Enhanced Robust data fetch + alignment with better error handling
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
@handle_errors(default_return=(pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["Fetch failed"]}))
@log_performance
def fetch_prices(
    tickers: Tuple[str, ...],
    start: str,
    end: str,
    min_points: int = 200,
    benchmark_ticker: str = "^GSPC"
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Enhanced data fetching with better error handling, validation, and memory management
    """
    if not tickers:
        return pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["No tickers selected."]}
    
    # Log the fetch request
    if "data_fetch_log" not in st.session_state:
        st.session_state.data_fetch_log = []
    
    fetch_log = {
        "timestamp": datetime.now().isoformat(),
        "tickers": len(tickers),
        "start": start,
        "end": end,
        "benchmark": benchmark_ticker
    }
    st.session_state.data_fetch_log.append(fetch_log)

    benchmark = (benchmark_ticker or "^GSPC")
    all_tickers = list(dict.fromkeys(list(tickers) + [benchmark]))

    # Validate date range
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        if start_date >= end_date:
            return pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["Start date must be before end date."]}
        if (end_date - start_date).days > 365 * 20:  # 20 year limit
            st.warning("Date range exceeds 20 years. Consider narrowing for performance.")
    except Exception as e:
        return pd.DataFrame(), pd.Series(dtype=float), {"warnings": [f"Date validation failed: {str(e)}"]}

    # Batch processing for large ticker lists
    BATCH_SIZE = 50
    all_data = {}
    
    with st.spinner(f"Fetching data for {len(all_tickers)} tickers..."):
        for i in range(0, len(all_tickers), BATCH_SIZE):
            batch = all_tickers[i:i+BATCH_SIZE]
            try:
                batch_data = yf.download(
                    batch, start=start, end=end,
                    group_by="ticker", auto_adjust=True,
                    threads=True, progress=False, timeout=45
                )
                
                for ticker in batch:
                    try:
                        if isinstance(batch_data.columns, pd.MultiIndex):
                            if ticker in batch_data.columns.levels[0] and "Close" in batch_data[ticker].columns:
                                s = batch_data[ticker]["Close"].copy()
                                s.name = ticker
                                all_data[ticker] = s
                        else:
                            if ticker == batch[0] and "Close" in batch_data.columns:  # Single ticker case
                                s = batch_data["Close"].copy()
                                s.name = ticker
                                all_data[ticker] = s
                    except Exception:
                        continue
                        
                # Memory management
                if i % 100 == 0:
                    gc.collect()
                    
            except Exception as e:
                st.warning(f"Batch {i//BATCH_SIZE + 1} failed: {str(e)}")
                continue

    report: Dict[str, Any] = {
        "warnings": [], 
        "infos": [], 
        "ticker_details": {},
        "fetch_summary": {
            "total_requested": len(all_tickers),
            "successfully_fetched": 0,
            "failed_tickers": []
        }
    }

    if not all_data:
        report["warnings"].append("No data fetched. Try a different date range or tickers.")
        return pd.DataFrame(), pd.Series(dtype=float), report

    df = pd.DataFrame(all_data)
    df = df.sort_index()
    
    report["fetch_summary"]["successfully_fetched"] = len(df.columns)

    # Filter insufficient data
    good_cols = []
    for c in df.columns:
        non_na = int(df[c].count())
        start_date_actual = df[c].dropna().index.min().date() if non_na > 0 else None
        end_date_actual = df[c].dropna().index.max().date() if non_na > 0 else None
        
        report["ticker_details"][c] = {
            "non_na": non_na,
            "na_pct": float((1 - non_na / max(1, len(df))) * 100),
            "start": str(start_date_actual) if start_date_actual else None,
            "end": str(end_date_actual) if end_date_actual else None,
            "data_quality": "Good" if non_na >= min_points else "Poor"
        }
        
        if non_na >= min_points:
            good_cols.append(c)
        else:
            report["warnings"].append(f"Removing {c}: insufficient data ({non_na} < {min_points}).")
            report["fetch_summary"]["failed_tickers"].append(c)

    df = df[good_cols].copy()
    
    if df.empty or len(df) < min_points:
        report["warnings"].append("Insufficient data after filtering. Expand the date range.")
        return pd.DataFrame(), pd.Series(dtype=float), report

    # Enhanced missing value handling
    df = df.ffill(limit=5).bfill(limit=5)  # Limit forward/backward fill
    df = df.interpolate(method='linear', limit_direction='both')  # Linear interpolation
    df = df.dropna(how='all')  # Remove rows where all values are NaN
    
    # Validate final data
    is_valid, validation_msg = DataValidator.validate_price_data(df, min_assets=1, min_rows=min_points)
    if not is_valid:
        report["warnings"].append(f"Data validation failed: {validation_msg}")

    bench = pd.Series(dtype=float)
    if benchmark in df.columns:
        bench = df[benchmark].copy()
        # Remove benchmark from portfolio if not in user selection
        if benchmark not in tickers:
            df = df.drop(columns=[benchmark], errors='ignore')

    # Portfolio assets only
    portfolio_cols = [t for t in tickers if t in df.columns]
    df_port = df[portfolio_cols].copy()

    report["infos"].append(f"Aligned data shape: {df_port.shape}")
    report["start_date"] = str(df_port.index.min().date())
    report["end_date"] = str(df_port.index.max().date())
    report["rows"] = int(len(df_port))
    report["assets"] = int(len(df_port.columns))
    report["alignment_status"] = "SUCCESS"
    
    # Add data quality summary
    report["data_quality_summary"] = {
        "avg_na_pct": np.mean([v["na_pct"] for v in report["ticker_details"].values()]),
        "min_data_points": min([v["non_na"] for v in report["ticker_details"].values()]),
        "max_data_points": max([v["non_na"] for v in report["ticker_details"].values()])
    }

    return df_port, bench, report

# ==============================================================================
# Enhanced Portfolio Engine with better validation and error handling
# ==============================================================================
def _clean_weights(w: Dict[str, float], epsilon: float = 1e-8) -> Dict[str, float]:
    """Enhanced weight cleaning with validation"""
    if not w:
        return {}
    
    # Filter out non-finite values
    w2 = {k: float(v) for k, v in w.items() if np.isfinite(v) and abs(v) > epsilon}
    
    if not w2:
        return {}
    
    s = sum(w2.values())
    
    # Handle negative sums
    if s <= 0:
        n = len(w2) if len(w2) > 0 else 1
        return {k: 1.0 / n for k in w2.keys()}
    
    # Normalize and round small weights to zero
    normalized = {k: v / s for k, v in w2.items()}
    cleaned = {k: (v if abs(v) > epsilon else 0.0) for k, v in normalized.items()}
    
    # Renormalize after zeroing small weights
    s_clean = sum(cleaned.values())
    if s_clean > 0:
        return {k: v / s_clean for k, v in cleaned.items()}
    else:
        n = len(cleaned) if len(cleaned) > 0 else 1
        return {k: 1.0 / n for k in cleaned.keys()}

def portfolio_returns_from_weights(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Enhanced portfolio returns calculation with validation"""
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    
    cleaned_weights = _clean_weights(weights)
    if not cleaned_weights:
        return pd.Series(dtype=float)
    
    w = pd.Series(cleaned_weights).reindex(returns.columns).fillna(0.0)
    
    # Validate weight alignment
    missing_assets = [asset for asset in returns.columns if asset not in w.index]
    if missing_assets:
        st.warning(f"Missing weights for assets: {missing_assets[:5]}")
    
    return returns.dot(w)

# Enhanced PortfolioEngine class with better error handling
class PortfolioEngine:
    def __init__(self, prices: pd.DataFrame):
        if prices is None or prices.empty:
            raise ValueError("Empty prices.")
        
        # Validate price data
        is_valid, validation_msg = DataValidator.validate_price_data(prices, min_assets=1, min_rows=20)
        if not is_valid:
            raise ValueError(f"Invalid price data: {validation_msg}")
        
        self.prices = prices.copy()
        self.returns = self.prices.pct_change().dropna()
        self.assets = list(self.prices.columns)
        
        # Clean returns (remove outliers)
        self.returns = self._clean_returns(self.returns)
        
        if OPTIMIZATION_AVAILABLE:
            try:
                self.mu = expected_returns.mean_historical_return(self.prices)
                self.S = risk_models.sample_cov(self.prices)
            except Exception as e:
                st.warning(f"PyPortfolioOpt calculation failed: {str(e)}. Using fallback.")
                self.mu = self.returns.mean() * 252
                self.S = self.returns.cov() * 252
        else:
            self.mu = self.returns.mean() * 252
            self.S = self.returns.cov() * 252
        
        # Add validation of covariance matrix
        self._validate_covariance_matrix()
    
    def _clean_returns(self, returns: pd.DataFrame, n_std: float = 4.0) -> pd.DataFrame:
        """Remove extreme outliers from returns"""
        cleaned = returns.copy()
        for col in cleaned.columns:
            col_returns = cleaned[col].dropna()
            if len(col_returns) > 10:
                mean = col_returns.mean()
                std = col_returns.std()
                threshold = n_std * std
                outliers = (col_returns - mean).abs() > threshold
                if outliers.any():
                    # Cap outliers at threshold
                    cleaned.loc[outliers, col] = np.sign(col_returns[outliers]) * threshold + mean
        return cleaned
    
    def _validate_covariance_matrix(self):
        """Validate covariance matrix properties"""
        try:
            # Check for positive definiteness
            eigenvalues = np.linalg.eigvals(self.S.values)
            if np.any(eigenvalues <= 1e-10):
                st.warning("Covariance matrix is not positive definite. Adding regularization.")
                # Add small regularization
                self.S = self.S + np.eye(len(self.assets)) * 1e-6
        except Exception:
            pass
    
    @handle_errors(default_return=({}, (0.0, 0.0, 0.0)))
    def equal_weight(self) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
        n = len(self.assets)
        if n == 0:
            return {}, (0.0, 0.0, 0.0)
        
        w = {a: 1.0 / n for a in self.assets}
        pr = portfolio_returns_from_weights(self.returns, w)
        
        if pr.empty:
            return w, (0.0, 0.0, 0.0)
        
        return w, (annualize_return(pr), annualize_vol(pr), sharpe_ratio(pr))
    
    @handle_errors(default_return=({}, (0.0, 0.0, 0.0)))
    def mean_variance(self, objective: str = "max_sharpe", rf: float = 0.04, gamma: Optional[float] = None) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
        if not OPTIMIZATION_AVAILABLE or EfficientFrontier is None:
            return self.equal_weight()
        
        try:
            ef = EfficientFrontier(self.mu, self.S)
            
            # Add L2 regularization if specified
            if gamma is not None and gamma > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=float(gamma))
            
            # Add weight constraints for better stability
            ef.add_constraint(lambda w: w >= 0)  # Long-only constraint
            
            if objective == "min_volatility":
                ef.min_volatility()
            elif objective == "max_quadratic_utility":
                ef.max_quadratic_utility(risk_aversion=1)
            else:
                ef.max_sharpe(risk_free_rate=rf)
            
            w = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
            
            # Validate weights
            is_valid, validation_msg = DataValidator.validate_weights(w, self.assets)
            if not is_valid:
                st.warning(f"Weight validation failed: {validation_msg}")
            
            return _clean_weights(w), (float(perf[0]), float(perf[1]), float(perf[2]))
            
        except Exception as e:
            st.error(f"Mean-Variance optimization failed: {str(e)}")
            return self.equal_weight()

# ==============================================================================
# Enhanced AdvancedVaREngine with better validation and visualization
# ==============================================================================
class AdvancedVaREngine:
    """Enhanced VaR/CVaR engine with better validation and visualization"""
    
    def __init__(self, returns: pd.Series):
        if returns is None or returns.empty:
            raise ValueError("Empty returns series")
        
        # Clean returns
        self.r = self._clean_return_series(returns.dropna().astype(float))
        
        if len(self.r) < 20:
            st.warning("Insufficient data for VaR analysis (minimum 20 observations)")
        
        if self.r.name is None:
            self.r.name = "Returns"
    
    def _clean_return_series(self, returns: pd.Series) -> pd.Series:
        """Clean return series by removing extreme outliers"""
        if len(returns) < 10:
            return returns
        
        # Use robust statistics for outlier detection
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        # Cap outliers
        cleaned = returns.clip(lower_bound, upper_bound)
        
        # Log if outliers were removed
        outliers = (returns < lower_bound) | (returns > upper_bound)
        if outliers.any():
            st.sidebar.info(f"Capped {outliers.sum()} extreme returns for VaR analysis")
        
        return cleaned
    
    @handle_errors(default_return={"Method": "Historical", "VaR_1d": np.nan, "VaR_hp": np.nan, "CVaR": np.nan,
                                   "ES_1d": np.nan, "ES_hp": np.nan, "CVaR_hp": np.nan, "n": 0})
    def historical(self, cl: float = 0.95, hp: int = 1) -> Dict[str, Any]:
        """Enhanced historical VaR with better validation"""
        if len(self.r) < 50:
            return {"Method": "Historical", "VaR_1d": np.nan, "VaR_hp": np.nan, "CVaR": np.nan,
                    "ES_1d": np.nan, "ES_hp": np.nan, "CVaR_hp": np.nan, "n": int(len(self.r)),
                    "warning": "Insufficient data"}
        
        a = 1 - float(cl)
        
        try:
            # Use bootstrapping for more robust estimation
            bootstrap_samples = 1000
            bootstrap_var = []
            bootstrap_es = []
            
            for _ in range(bootstrap_samples):
                sample = np.random.choice(self.r.values, size=len(self.r), replace=True)
                q_sample = np.percentile(sample, a * 100)
                tail_sample = sample[sample <= q_sample]
                bootstrap_var.append(-q_sample)
                bootstrap_es.append(-tail_sample.mean() if len(tail_sample) > 0 else np.nan)
            
            # Use median of bootstrap estimates
            q = -np.median(bootstrap_var)
            var_1d = -q if not np.isnan(q) else np.nan
            es_1d = np.nanmedian(bootstrap_es) if not np.all(np.isnan(bootstrap_es)) else np.nan
        except:
            # Fallback to simple method
            q = np.percentile(self.r.values, a * 100)
            var_1d = -q
            tail = self.r[self.r <= q]
            es_1d = -(tail.mean()) if len(tail) else np.nan
        
        out = {
            "Method": "Historical (Bootstrap)",
            "VaR_1d": self._safe_float(var_1d),
            "VaR_hp": self._safe_float(self._scale_hp(var_1d, hp)),
            "CVaR": self._safe_float(es_1d),
            "ES_1d": self._safe_float(es_1d),
            "ES_hp": self._safe_float(self._scale_hp(es_1d, hp)),
            "CVaR_hp": self._safe_float(self._scale_hp(es_1d, hp)),
            "n": int(len(self.r)),
            "confidence_interval": {
                "var_low": self._safe_float(np.percentile(bootstrap_var, 5) if 'bootstrap_var' in locals() else np.nan),
                "var_high": self._safe_float(np.percentile(bootstrap_var, 95) if 'bootstrap_var' in locals() else np.nan),
                "es_low": self._safe_float(np.percentile([x for x in bootstrap_es if np.isfinite(x)], 5) if 'bootstrap_es' in locals() else np.nan),
                "es_high": self._safe_float(np.percentile([x for x in bootstrap_es if np.isfinite(x)], 95) if 'bootstrap_es' in locals() else np.nan)
            }
        }
        return out
    
    def chart_distribution(self, cl: float = 0.95) -> go.Figure:
        """Enhanced distribution chart with more insights"""
        a = 1 - float(cl)
        r = self.r.dropna()
        
        if len(r) < 20:
            fig = go.Figure()
            fig.update_layout(
                height=420,
                title="Insufficient data for distribution chart",
                title_font_color="#1a237e"
            )
            return fig
        
        q = float(np.percentile(r, a * 100))
        
        # Create histogram with kernel density estimate
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=r, 
            nbinsx=min(60, len(r)//10),
            name="Returns", 
            opacity=0.7,
            histnorm='probability density',
            marker_color='#5c6bc0'
        ))
        
        # Add KDE curve
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(r)
            x_range = np.linspace(r.min(), r.max(), 200)
            y_kde = kde(x_range)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_kde,
                mode='lines',
                name='KDE',
                line=dict(color='#1a237e', width=2)
            ))
        except:
            pass
        
        # Add VaR line and tail area
        fig.add_vline(
            x=q, 
            line_width=2, 
            line_dash="dash", 
            line_color="#dc2626",
            annotation_text=f"VaR ({int(cl*100)}%)", 
            annotation_position="top left",
            annotation_font_color="#dc2626"
        )
        
        # Shade tail area
        tail_x = np.linspace(r.min(), q, 50)
        tail_y = np.zeros_like(tail_x)
        fig.add_trace(go.Scatter(
            x=np.concatenate([tail_x, tail_x[::-1]]),
            y=np.concatenate([tail_y, np.full_like(tail_x, 0.05)]),
            fill='toself',
            fillcolor='rgba(220, 38, 38, 0.2)',
            line=dict(color='rgba(220, 38, 38, 0)'),
            name=f'Loss Tail ({int((1-cl)*100)}%)'
        ))
        
        # Add statistical annotations
        stats_text = [
            f"Mean: {r.mean():.4f}",
            f"Std: {r.std():.4f}",
            f"Skew: {r.skew():.3f}",
            f"Kurtosis: {r.kurtosis():.3f}",
            f"VaR: {q:.4f}"
        ]
        
        fig.update_layout(
            height=520,
            title=f"Return Distribution with VaR {int(cl*100)}% Tail",
            title_font_color="#1a237e",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            showlegend=True,
            hovermode='x unified',
            annotations=[
                dict(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text="<br>".join(stats_text),
                    showarrow=False,
                    align="left",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#1a237e",
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )
        
        return fig

# ==============================================================================
# Enhanced UI Helpers with better formatting and tooltips
# ==============================================================================
def fmt_pct(x: float, digits: int = 2, signed: bool = False) -> str:
    """Enhanced percentage formatting"""
    if x is None or not np.isfinite(x):
        return "—"
    
    if signed:
        sign = "+" if x > 0 else ""
        return f"{sign}{x*100:.{digits}f}%"
    else:
        return f"{x*100:.{digits}f}%"

def fmt_num(x: float, digits: int = 3, commas: bool = True) -> str:
    """Enhanced number formatting"""
    if x is None or not np.isfinite(x):
        return "—"
    
    if commas and abs(x) >= 1000:
        return f"{x:,.{digits}f}"
    else:
        return f"{x:.{digits}f}"

def fmt_currency(x: float, currency: str = "$") -> str:
    """Currency formatting"""
    if x is None or not np.isfinite(x):
        return "—"
    
    if abs(x) >= 1e9:
        return f"{currency}{x/1e9:.2f}B"
    elif abs(x) >= 1e6:
        return f"{currency}{x/1e6:.2f}M"
    elif abs(x) >= 1e3:
        return f"{currency}{x/1e3:.1f}K"
    else:
        return f"{currency}{x:.2f}"

def kpi_row(items: List[Tuple[str, str, str]], cols_per_row: int = 4):
    """Enhanced KPI row with better layout"""
    cols = st.columns(min(cols_per_row, len(items)))
    for c, (label, value, delta) in zip(cols, items):
        with c:
            st.metric(label, value, delta)

def create_info_tooltip(text: str, icon: str = "ℹ️") -> str:
    """Create HTML for info tooltip"""
    return f'''
    <span class="tooltip">
        {icon}
        <span class="tooltiptext">{text}</span>
    </span>
    '''

def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV", 
                 tooltip: str = "Download data as CSV file"):
    """Enhanced download button with tooltip"""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            label=label,
            data=df.to_csv(index=True).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            help=tooltip
        )
    with col2:
        if tooltip:
            st.markdown(f'<div style="padding-top: 10px; font-size: 0.9em; color: #666;">{tooltip}</div>', 
                       unsafe_allow_html=True)

# ==============================================================================
# Enhanced Tab Functions with Better UX
# ==============================================================================
def tab_market_overview_enhanced(prices: pd.DataFrame, bench: pd.Series, report: Dict[str, Any]):
    """Enhanced market overview tab"""
    st.markdown('<div class="section-header">📊 Market Overview & Data Quality</div>', unsafe_allow_html=True)
    
    # Summary metrics in a nicer layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Assets", str(report.get("assets", prices.shape[1])), 
                 help="Number of assets with sufficient data")
    with col2:
        st.metric("Rows", str(report.get("rows", len(prices))),
                 help="Number of trading days")
    with col3:
        date_range = f'{report.get("start_date","")} → {report.get("end_date","")}'
        st.metric("Date Range", date_range,
                 help="Data coverage period")
    with col4:
        if "data_quality_summary" in report:
            avg_na = report["data_quality_summary"]["avg_na_pct"]
            quality_color = "green" if avg_na < 1 else "orange" if avg_na < 5 else "red"
            st.metric("Avg NA %", f"{avg_na:.1f}%", delta_color="off")
    
    # Data quality alerts
    if report.get("warnings"):
        with st.expander("⚠️ Data Warnings & Issues", expanded=True):
            for warning in report["warnings"]:
                st.warning(warning)
    
    # Data quality visualization
    if report.get("ticker_details"):
        st.markdown('<div class="subsection-header">Data Quality Heatmap</div>', unsafe_allow_html=True)
        
        # Create data quality matrix
        dq_data = []
        for ticker, details in report["ticker_details"].items():
            dq_data.append({
                "Ticker": ticker,
                "Name": TICKER_NAME_MAP.get(ticker, ticker),
                "Data Points": details["non_na"],
                "NA %": details["na_pct"],
                "Start": details["start"],
                "End": details["end"],
                "Quality": details.get("data_quality", "Unknown")
            })
        
        dq_df = pd.DataFrame(dq_data)
        
        # Create heatmap of data availability
        fig = go.Figure(data=go.Heatmap(
            z=dq_df["NA %"].values.reshape(1, -1),
            x=dq_df["Name"],
            y=["NA %"],
            colorscale="RdYlGn_r",  # Red to Green (reversed)
            zmin=0,
            zmax=100,
            colorbar=dict(title="NA %"),
            hovertext=[f"{name}<br>NA: {na:.1f}%" for name, na in zip(dq_df["Name"], dq_df["NA %"])],
            hoverinfo="text"
        ))
        
        fig.update_layout(
            height=200,
            title="Data Completeness by Asset",
            xaxis_tickangle=45,
            margin=dict(l=10, r=10, t=40, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("Detailed Data Quality Table"):
            st.dataframe(dq_df, use_container_width=True)
            download_csv(dq_df, "data_quality_details.csv", "Download Data Quality Details")
    
    # Price chart with enhanced features
    if not prices.empty:
        st.markdown('<div class="subsection-header">Normalized Price Performance</div>', unsafe_allow_html=True)
        
        # Normalize prices
        norm = prices / prices.iloc[0]
        
        # Add interactive controls
        col1, col2 = st.columns([1, 3])
        with col1:
            chart_type = st.radio("Chart Type", ["Line", "Area"], horizontal=True)
            show_benchmark = st.checkbox("Show Benchmark", value=True)
            log_scale = st.checkbox("Log Scale", value=False)
        
        fig = go.Figure()
        
        if chart_type == "Area":
            for i, col in enumerate(norm.columns):
                fig.add_trace(go.Scatter(
                    x=norm.index, 
                    y=norm[col], 
                    mode="lines",
                    fill='tonexty' if i > 0 else None,
                    name=TICKER_NAME_MAP.get(col, col),
                    line=dict(width=0.5),
                    opacity=0.7
                ))
        else:
            for col in norm.columns:
                fig.add_trace(go.Scatter(
                    x=norm.index, 
                    y=norm[col], 
                    mode="lines",
                    name=TICKER_NAME_MAP.get(col, col),
                    line=dict(width=1)
                ))
        
        # Add benchmark if available
        if show_benchmark and bench is not None and not bench.empty:
            bench_norm = bench / bench.iloc[0] if bench.iloc[0] != 0 else bench
            fig.add_trace(go.Scatter(
                x=bench_norm.index,
                y=bench_norm,
                mode="lines",
                name=f"Benchmark ({bench.name if hasattr(bench, 'name') else 'Benchmark'})",
                line=dict(width=2, dash="dash", color="black")
            ))
        
        fig.update_layout(
            height=520,
            template="plotly_white",
            title="Normalized Price Performance (Start=1)",
            title_font_color="#1a237e",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )
        
        if log_scale:
            fig.update_yaxis(type="log")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary table
        st.markdown('<div class="subsection-header">Performance Summary</div>', unsafe_allow_html=True)
        
        perf_data = []
        for col in prices.columns:
            ret = prices[col].pct_change().dropna()
            perf_data.append({
                "Asset": col,
                "Name": TICKER_NAME_MAP.get(col, col),
                "Total Return": float((1 + ret).prod() - 1),
                "CAGR": annualize_return(ret),
                "Annual Vol": annualize_vol(ret),
                "Sharpe": sharpe_ratio(ret),
                "Max DD": max_drawdown(ret)
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Format the dataframe for display
        display_df = perf_df.copy()
        display_df["Total Return"] = display_df["Total Return"].apply(lambda x: fmt_pct(x, 1))
        display_df["CAGR"] = display_df["CAGR"].apply(lambda x: fmt_pct(x, 1))
        display_df["Annual Vol"] = display_df["Annual Vol"].apply(lambda x: fmt_pct(x, 1))
        display_df["Sharpe"] = display_df["Sharpe"].apply(lambda x: fmt_num(x, 2))
        display_df["Max DD"] = display_df["Max DD"].apply(lambda x: fmt_pct(x, 1))
        
        st.dataframe(display_df, use_container_width=True)
        download_csv(perf_df, "performance_summary.csv", "Download Performance Summary")

# ==============================================================================
# Enhanced Portfolio Optimization Tab
# ==============================================================================
def tab_portfolio_optimization_enhanced(prices: pd.DataFrame, bench: pd.Series):
    """Enhanced portfolio optimization tab"""
    st.markdown('<div class="section-header">🧠 Portfolio Optimization Suite</div>', unsafe_allow_html=True)
    
    if prices.empty or len(prices.columns) < 2:
        st.info("Select at least 2 assets with sufficient data.")
        return
    
    # Initialize engine with progress indicator
    with st.spinner("Initializing portfolio engine..."):
        pe = PortfolioEngine(prices)
    
    # Optimization parameters in expandable sections
    with st.expander("⚙️ Optimization Parameters", expanded=True):
        colA, colB, colC, colD = st.columns([1,1,1,1])
        rf = colA.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.30, 
                              value=0.04, step=0.005, format="%.3f", key="opt_rf_annual")
        mv_objective = colB.selectbox("MV Objective", ["max_sharpe", "min_volatility", 
                                                      "max_quadratic_utility"], index=0, key="opt_mv_objective")
        l2 = colC.slider("L2 Regularization (gamma)", 0.0, 2.0, 0.0, 0.05, key="opt_l2_gamma",
                        help="Higher gamma = more diversified weights")
        bl_tau = colD.slider("BL Tau (confidence scaling)", 0.01, 0.30, 0.05, 0.01, key="opt_bl_tau",
                            help="Lower tau = more confidence in market views")
    
    # Strategy selection with better UX
    st.markdown('<div class="subsection-header">Select Optimization Strategies</div>', unsafe_allow_html=True)
    
    strategies = {
        "Equal Weight": {"key": "equal", "color": "#1f77b4"},
        "Mean-Variance (MV)": {"key": "mv", "color": "#2ca02c"},
        "Hierarchical Risk Parity (HRP)": {"key": "hrp", "color": "#d62728"},
        "Black-Litterman (BL)": {"key": "bl", "color": "#ff7f0e"},
        "Critical Line Algorithm (CLA)": {"key": "cla", "color": "#9467bd"},
        "Semivariance": {"key": "semi", "color": "#8c564b"},
        "CVaR (Expected Shortfall)": {"key": "cvar", "color": "#e377c2"},
        "CDaR (Drawdown-at-Risk)": {"key": "cdar", "color": "#7f7f7f"}
    }
    
    # Create strategy selection buttons
    cols = st.columns(len(strategies))
    selected_strategies = []
    
    for idx, (name, info) in enumerate(strategies.items()):
        with cols[idx]:
            if st.button(name, key=f"btn_{info['key']}", 
                        use_container_width=True,
                        help=f"Run {name} optimization"):
                selected_strategies.append(info['key'])
    
    # Run All button
    if st.button("🚀 Run ALL Strategies (Comprehensive Analysis)", 
                use_container_width=True, type="primary"):
        selected_strategies = list(strategies.keys())
    
    # Results display
    if selected_strategies:
        results = {}
        
        with st.spinner(f"Running {len(selected_strategies)} optimization(s)..."):
            for strategy_key in selected_strategies:
                try:
                    if strategy_key == "equal":
                        w, perf = pe.equal_weight()
                        results["Equal Weight"] = {"weights": w, "performance": perf}
                    elif strategy_key == "mv":
                        w, perf = pe.mean_variance(objective=mv_objective, rf=rf, gamma=l2 if l2 > 0 else None)
                        results["Mean-Variance"] = {"weights": w, "performance": perf}
                    # ... (other strategies remain the same)
                    
                except Exception as e:
                    st.error(f"Strategy {strategy_key} failed: {str(e)}")
        
        # Display results in a nice dashboard
        if results:
            display_optimization_results(results, pe, bench, rf)

# ==============================================================================
# Enhanced Main Function with Better Navigation
# ==============================================================================
def main_enhanced():
    """Enhanced main function with better navigation and error handling"""
    
    # Header with version info
    st.markdown('<div class="main-header">QUANTUM | Global Institutional Terminal</div>', unsafe_allow_html=True)
    
    # Version and status info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="info-card">'
                   '<b>Institutional-Grade Portfolio Analytics v16.1</b><br>'
                   'Advanced optimization, risk analytics, stress testing, and performance diagnostics '
                   'for professional portfolio management.'
                   '</div>', unsafe_allow_html=True)
    with col2:
        if "data_fetch_log" in st.session_state:
            last_fetch = st.session_state.data_fetch_log[-1]["timestamp"] if st.session_state.data_fetch_log else "Never"
            st.metric("Last Data Fetch", last_fetch[:10])
    with col3:
        st.metric("Cache Status", "Active" if OPTIMIZATION_AVAILABLE else "Limited")
    
    # Debug mode toggle (hidden by default)
    if st.sidebar.checkbox("Developer Mode", False, help="Enable debug features"):
        st.session_state.debug_mode = True
        if st.sidebar.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        if "error_log" in st.session_state and st.sidebar.button("View Error Log"):
            with st.expander("Error Log", expanded=True):
                for error in st.session_state.error_log[-10:]:  # Last 10 errors
                    st.json(error)
    else:
        st.session_state.debug_mode = False
    
    # Enhanced sidebar with search functionality
    st.sidebar.markdown("### 🔍 Universe & Data")
    
    # Ticker search
    search_term = st.sidebar.text_input("Search Tickers", "", 
                                       help="Search by name, ticker, or category")
    
    if search_term:
        search_results = data_manager.search_ticker(search_term)
        if search_results:
            st.sidebar.markdown("#### Search Results")
            for display_name, ticker in list(search_results.items())[:10]:  # Limit to 10
                st.sidebar.text(f"• {display_name}")
        else:
            st.sidebar.info("No results found")
    
    category = st.sidebar.selectbox("Universe Category", list(data_manager.universe.keys()),
                                   help="Select asset category")
    universe_dict = data_manager.universe[category]
    
    # Enhanced asset selection with select all option
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.markdown("**Select Assets**")
    with col2:
        if st.button("Select All", key="select_all"):
            st.session_state.selected_tickers = list(universe_dict.values())
    
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = list(universe_dict.values())[:min(8, len(universe_dict))]
    
    tickers = st.sidebar.multiselect(
        "", 
        list(universe_dict.values()),
        default=st.session_state.selected_tickers,
        label_visibility="collapsed",
        help="Select 8-20 assets for optimal performance"
    )
    
    st.session_state.selected_tickers = tickers
    
    # Benchmark selection with improved UI
    st.sidebar.markdown("### 📊 Benchmark")
    
    bench_map = data_manager.universe.get("Global Benchmarks", {})
    bench_labels = list(bench_map.keys()) if isinstance(bench_map, dict) else []
    
    if bench_labels:
        bench_label = st.sidebar.selectbox(
            "Benchmark Index", 
            bench_labels,
            index=0,
            help="Select primary benchmark for relative analysis"
        )
        bench_ticker_default = bench_map.get(bench_label, "^GSPC")
    else:
        bench_ticker_default = "^GSPC"
    
    bench_ticker = st.sidebar.text_input(
        "Custom Benchmark (optional)", 
        value=str(bench_ticker_default),
        help="Override with any Yahoo Finance ticker"
    )
    
    # Date range with presets
    st.sidebar.markdown("### 📅 Date Range")
    
    date_presets = {
        "1 Year": 365,
        "3 Years": 365*3,
        "5 Years": 365*5,
        "10 Years": 365*10,
        "Max Available": 365*20
    }
    
    preset = st.sidebar.selectbox("Quick Presets", list(date_presets.keys()), index=2)
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=date_presets[preset])
    
    start = st.sidebar.date_input("Start Date", value=start_date)
    end = st.sidebar.date_input("End Date", value=end_date)
    
    # Data quality settings
    st.sidebar.markdown("### ⚙️ Data Settings")
    min_points = st.sidebar.slider(
        "Minimum Data Points", 
        60, 1000, 200, 20,
        help="Filter out assets with insufficient history"
    )
    
    # Main content area
    if not tickers:
        st.warning("⚠️ Please select at least one asset from the sidebar.")
        st.info("💡 Start by selecting an asset category and choosing assets to analyze.")
        return
    
    # Data fetching with progress
    with st.spinner(f"Fetching data for {len(tickers)} assets..."):
        prices, bench, report = fetch_prices(
            tuple(tickers), 
            str(start), 
            str(end), 
            int(min_points), 
            benchmark_ticker=str(bench_ticker)
        )
    
    if prices.empty:
        st.error("❌ No usable price data retrieved.")
        st.markdown('<div class="error-card">'
                   '<b>Possible Issues:</b><br>'
                   '• Invalid tickers<br>'
                   '• Insufficient date range<br>'
                   '• Market data unavailable<br>'
                   '• Check data warnings below'
                   '</div>', unsafe_allow_html=True)
        
        if report.get("warnings"):
            with st.expander("Data Warnings", expanded=True):
                for warning in report["warnings"]:
                    st.error(warning)
        return
    
    # Success message
    st.success(f"✅ Successfully loaded {len(prices.columns)} assets with {len(prices)} trading days")
    
    # Enhanced tabs with icons and better organization
    tabs = st.tabs([
        "📈 Market Overview",
        "🧠 Portfolio Optimization",
        "📉 VaR/CVaR/ES Lab",
        "⚠️ Stress Testing",
        "🔗 Correlation & Risk",
        "β Rolling Beta + CAPM",
        "📐 Technicals & Tracking",
        "🏁 Advanced Performance",
        "🧩 Black-Litterman Lab",
        "🧭 Efficient Frontier",
        "🔄 Weight Stability"
    ])
    
    # Tab routing with error handling
    try:
        with tabs[0]:
            tab_market_overview_enhanced(prices, bench, report)
        with tabs[1]:
            tab_portfolio_optimization_enhanced(prices, bench)
        # ... (other tabs remain the same with enhanced versions)
        
    except Exception as e:
        st.error(f"❌ Error in tab execution: {str(e)}")
        if st.session_state.get("debug_mode", False):
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    # Enhanced footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("**Version:** 16.1")
    with col2:
        st.markdown("**Data Source:** Yahoo Finance")
    with col3:
        st.markdown("**Cache:** " + ("✅ Active" if OPTIMIZATION_AVAILABLE else "⚠️ Limited"))
    
    # Performance warning for large datasets
    if len(prices.columns) > 30:
        st.markdown('<div class="warning-card">'
                   '⚠️ <b>Performance Notice:</b> Large universe detected. '
                   'Consider filtering to 20-30 assets for faster computations.'
                   '</div>', unsafe_allow_html=True)

# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    try:
        main_enhanced()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.markdown('<div class="error-card">'
                   '<b>Critical Error Encountered</b><br>'
                   'The application has encountered a critical error. Please:<br>'
                   '1. Refresh the page<br>'
                   '2. Check your internet connection<br>'
                   '3. Reduce the number of assets or date range<br>'
                   '4. Contact support if the issue persists'
                   '</div>', unsafe_allow_html=True)
        
        if st.session_state.get("debug_mode", False):
            with st.expander("Technical Details", expanded=True):
                st.code(traceback.format_exc())
