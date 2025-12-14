# ==============================================================================
# QUANTUM | Global Institutional Terminal - COMPLETE ENHANCED VERSION
# Advanced VaR/CVaR/ES + Stress Testing + Performance Metrics + Black-Litterman + 3D Frontier
# PLUS: Bollinger Bands, OHLC Charts, Enhanced Stress Testing, Shock Simulator
# ==============================================================================

import streamlit as st

# --- 1) STREAMLIT PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="QUANTUM | Advanced Risk Analytics",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# --- 2) STANDARD LIBRARIES & THIRD-PARTY IMPORTS ---
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import jarque_bera
from scipy import optimize
import json
import base64
from collections import defaultdict


# --- 2B) OPTIONAL QUANT LIBRARIES (PyPortfolioOpt) ---
# We keep the app fully functional even if PyPortfolioOpt is missing (Streamlit Cloud requirements.txt).
try:
    from pypfopt import expected_returns, risk_models, objective_functions
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import (
        BlackLittermanModel,
        market_implied_prior_returns,
        market_implied_risk_aversion,
    )
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.efficient_semivariance import EfficientSemivariance
    from pypfopt.efficient_cvar import EfficientCVaR
    from pypfopt.efficient_cdar import EfficientCDaR
    PYPFOPT_AVAILABLE = True
    PYPFOPT_IMPORT_ERROR = ""
except Exception as _e:
    PYPFOPT_AVAILABLE = False
    PYPFOPT_IMPORT_ERROR = str(_e)[:300]

# --- 3) UPDATED CSS WITH LIGHT GREY BACKGROUND & DARK BLUE TITLES ---
st.markdown("""
    <style>
        /* Main background - Light Grey */
        .stApp {
            background-color: #f5f7fa;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Main container styling */
        .main-container {
            background-color: white;
            border-radius: 12px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid #e1e5eb;
        }
        
        /* Main header - Dark Blue */
        .main-header {
            font-size: 2.8rem;
            font-weight: 800;
            color: #1a237e;
            padding-bottom: 15px;
            margin-bottom: 30px;
            text-align: center;
            border-bottom: 3px solid #303f9f;
            letter-spacing: -0.5px;
        }
        
        /* Section headers - Dark Blue */
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1a237e;
            margin-top: 25px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e8eaf6;
        }
        
        /* Sub-section headers */
        .subsection-header {
            font-size: 1.4rem;
            font-weight: 600;
            color: #283593;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        
        /* Metric cards */
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #1a237e;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            margin: 0.5rem 0;
        }
        
        .metric-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #424242;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1a237e;
            margin: 5px 0;
        }
        
        .metric-change {
            font-size: 0.85rem;
            font-weight: 500;
            color: #666;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background-color: #f0f2f6;
            padding: 5px;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            color: #5c6bc0;
            transition: all 0.3s ease;
            background-color: white;
            border: 1px solid #e0e0e0;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
            color: white;
            border-color: #1a237e;
            box-shadow: 0 4px 12px rgba(26, 35, 126, 0.15);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #303f9f 0%, #1a237e 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(26, 35, 126, 0.25);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
        }
        
        .css-1lcbmhc {
            border-right: 1px solid #e0e0e0;
        }
        
        /* Expander headers */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            color: #1a237e;
            font-weight: 600;
        }
        
        /* Card-like containers */
        .card-container {
            background: white;
            border: 1px solid #e1e5eb;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }
        
        /* Warning/Info cards */
        .warning-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #9ca3af;
            padding: 1rem;
            border-radius: 12px;
            margin: 0.75rem 0;
        }
        
        .success-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #10b981;
            padding: 1rem;
            border-radius: 12px;
            margin: 0.75rem 0;
        }
        
        .info-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #1a237e;
            padding: 1rem;
            border-radius: 12px;
            margin: 0.75rem 0;
        }
        
        /* Historical event styling */
        .historical-event {
            background: #f8f9fa;
            border-left: 4px solid #303f9f;
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        
        .historical-event:hover {
            background: #e8eaf6;
            transform: translateX(2px);
        }
        
        /* Input field styling */
        .stSelectbox, .stNumberInput, .stDateInput, .stSlider {
            background-color: white;
            border-radius: 6px;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #303f9f 0%, #7986cb 100%);
        }
        
        /* Risk badge styling */
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .risk-low {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        
        .risk-medium {
            background-color: #fff3e0;
            color: #ef6c00;
        }
        
        .risk-high {
            background-color: #ffebee;
            color: #c62828;
        }
        
        .risk-extreme {
            background-color: #f3e5f5;
            color: #6a1b9a;
        }
        
        /* Chart containers */
        .plotly-chart-container {
            background: white;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            margin: 15px 0;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
        }
        
        /* Spinner styling */
        .stSpinner > div {
            border-top-color: #303f9f;
        }
        
        /* Metric containers in columns */
        .stMetric {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius= 8px;
            padding: 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        
        .stMetric > div > div {
            color: #1a237e;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius= 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c5cae9;
            border-radius= 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #7986cb;
        }
        
        /* Performance radar specific */
        .radar-metric {
            font-size: 0.85rem;
            color: #424242;
        }
        
        /* 3D chart controls */
        .modebar {
            background-color: white !important;
            border: 1px solid #e0e0e0 !important;
            border-radius= 8px !important;
        }
        
        /* Bollinger band alerts */
        .bb-alert-up {
            background-color: #ffebee !important;
            border-left: 4px solid #d32f2f !important;
        }
        
        .bb-alert-down {
            background-color: #e8f5e9 !important;
            border-left: 4px solid #388e3c !important;
        }
        
        /* OHLC chart specific */
        .candle-up {
            background-color: #388e3c;
        }
        
        .candle-down {
            background-color: #d32f2f;
        }
        
        /* Shock simulator specific */
        .shock-param-card {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border: 1px solid #ce93d8;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4) ORIGINAL ENHANCED DATA MANAGER (FIXED)
# ==============================================================================

class EnhancedDataManager:
    def __init__(self):
        self.universe = {
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
                "MSCI Emerging Markets": "EEM"
            },
            "US ETFs (Major & Active)": {
                "SPDR S&P 500 ETF (SPY)": "SPY",
                "Invesco QQQ Trust (QQQ)": "QQQ",
                "iShares Russell 2000 (IWM)": "IWM",
                "Vanguard Total Stock Market (VTI)": "VTI",
                "ARK Innovation ETF (ARKK)": "ARKK",
                "Financial Select Sector SPDR (XLF)": "XLF",
                "Technology Select Sector SPDR (XLK)": "XLK",
                "Energy Select Sector SPDR (XLE)": "XLE",
                "Health Care Select Sector SPDR (XLV)": "XLV",
                "SPDR Gold Shares (GLD)": "GLD",
                "iShares Silver Trust (SLV)": "SLV",
                "United States Copper Index (CPER)": "CPER",
                "VanEck Gold Miners (GDX)": "GDX",
                "iShares 20+ Year Treasury Bond (TLT)": "TLT",
                "iShares iBoxx $ High Yield Corp Bond (HYG)": "HYG"
            },
            "BIST 30 (Turkey - Top Active)": {
                "Akbank": "AKBNK.IS", "Arcelik": "ARCLK.IS", "Aselsan": "ASELS.IS",
                "BIM Magazalar": "BIMAS.IS", "Emlak Konut": "EKGYO.IS", "Enka Insaat": "ENKAI.IS",
                "Eregli Demir Celik": "EREGL.IS", "Ford Otosan": "FROTO.IS", "Garanti BBVA": "GARAN.IS",
                "Gubre Fabrikalari": "GUBRF.IS", "Halkbank": "HALKB.IS", "Hektas": "HEKTS.IS",
                "Is Bankasi (C)": "ISCTR.IS", "Koc Holding": "KCHOL.IS", "Koza Altin": "KOZAL.IS",
                "Kardemir (D)": "KRDMD.IS", "Odas Elektrik": "ODAS.IS", "Petkim": "PETKM.IS",
                "Pegasus": "PGSUS.IS", "Sabanci Holding": "SAHOL.IS", "SASA Polyester": "SASA.IS",
                "Sise Cam": "SISE.IS", "TAV Havalimanlari": "TAVHL.IS", "Turkcell": "TCELL.IS",
                "Turk Hava Yollari": "THYAO.IS", "Tekfen Holding": "TKFEN.IS", "Tofas Oto": "TOASO.IS",
                "Tupras": "TUPRS.IS", "Vakifbank": "VAKBN.IS", "Yapi Kredi": "YKBNK.IS"
            },
            "Japan (Major Financials & Industrials)": {
                "Mitsubishi UFJ Financial Group": "8306.T",
                "Sumitomo Mitsui Financial Group": "8316.T",
                "Mizuho Financial Group": "8411.T",
                "Nomura Holdings": "8604.T",
                "Daiwa Securities Group": "8601.T",
                "Toyota Motor": "7203.T",
                "Sony Group": "6758.T",
                "Hitachi": "6501.T",
                "Mitsubishi Corporation": "8058.T",
                "Honda Motor": "7267.T",
                "Nintendo": "7974.T",
                "Panasonic": "6752.T",
                "Canon": "7751.T",
                "SoftBank Group": "9984.T",
                "Mitsubishi Estate": "8802.T",
                "Mitsui & Co.": "8031.T"
            },
            "Australia (Major Stocks)": {
                "Commonwealth Bank of Australia": "CBA.AX",
                "Westpac Banking Corporation": "WBC.AX",
                "Australia and New Zealand Banking Group": "ANZ.AX",
                "National Australia Bank": "NAB.AX",
                "Macquarie Group": "MQG.AX",
                "BHP Group": "BHP.AX",
                "Rio Tinto": "RIO.AX",
                "Fortescue Metals Group": "FMG.AX",
                "CSL Limited": "CSL.AX",
                "Wesfarmers": "WES.AX",
                "Woolworths Group": "WOW.AX",
                "Transurban Group": "TCL.AX",
                "Newcrest Mining": "NCM.AX",
                "South32": "S32.AX",
                "Woodside Energy Group": "WDS.AX"
            },
            "Global Mega Caps": {
                "Apple (US)": "AAPL",
                "Microsoft (US)": "MSFT",
                "Nvidia (US)": "NVDA",
                "Amazon (US)": "AMZN",
                "Alphabet (US)": "GOOGL",
                "Meta Platforms (US)": "META",
                "Tesla (US)": "TSLA",
                "Berkshire Hathaway (US)": "BRK-B",
                "LVMH (France)": "MC.PA",
                "ASML (Netherlands)": "ASML",
                "Novo Nordisk (Denmark)": "NVO",
                "TSMC (Taiwan)": "TSM",
                "Samsung (Korea)": "005930.KS",
                "Alibaba (China)": "BABA",
                "Tencent (China)": "0700.HK"
            },
            "Rates & Fixed Income": {
                "US 10Y Treasury Yield": "^TNX",
                "US 2Y Treasury Yield": "^FVX",
                "iShares 20+ Year Treasury Bond ETF": "TLT",
                "iShares 7-10 Year Treasury Bond ETF": "IEF",
                "iShares Core US Aggregate Bond ETF": "AGG",
                "Vanguard Total Bond Market ETF": "BND",
                "SPDR Bloomberg High Yield Bond ETF": "JNK",
                "iShares iBoxx $ Investment Grade Corp Bond ETF": "LQD"
            }
        }

        # Regional classification mapping
        self.regional_classification = {
            "North America": ["^GSPC", "^NDX", "^DJI", "^RUT", "SPY", "QQQ", "IWM", "VTI", "ARKK",
                            "XLF", "XLK", "XLE", "XLV", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
                            "META", "TSLA", "BRK-B", "^TNX", "^FVX", "TLT", "IEF", "AGG", "BND",
                            "JNK", "LQD", "GLD", "SLV", "CPER", "GDX", "HYG"],
            "Europe": ["^FTSE", "^GDAXI", "^FCHI", "MC.PA", "ASML", "NVO"],
            "Asia Pacific": ["^N225", "^HSI", "000001.SS", "^AXJO", "TSM", "005930.KS", "BABA", "0700.HK",
                           "8306.T", "8316.T", "8411.T", "8604.T", "8601.T", "7203.T", "6758.T", "6501.T",
                           "8058.T", "7267.T", "7974.T", "6752.T", "7751.T", "9984.T", "8802.T", "8031.T",
                           "CBA.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "MQG.AX", "BHP.AX", "RIO.AX", "FMG.AX",
                           "CSL.AX", "WES.AX", "WOW.AX", "TCL.AX", "NCM.AX", "S32.AX", "WDS.AX"],
            "Emerging Markets": ["XU100.IS", "EEM", "URTH"] + [v for v in self.universe["BIST 30 (Turkey - Top Active)"].values()]
        }

    def get_ticker_name_map(self) -> Dict[str, str]:
        mapping = {}
        for category, assets in self.universe.items():
            for name, ticker in assets.items():
                mapping[ticker] = name
        return mapping

    def get_regional_exposure(self, tickers: List[str]) -> Dict[str, float]:
        """Calculate regional exposure for selected tickers."""
        exposure = {}
        tickers = list(dict.fromkeys(tickers))  # de-duplicate while preserving order
        total = len(tickers)
        if total == 0:
            return exposure

        for region, region_tickers in self.regional_classification.items():
            region_count = sum(1 for t in tickers if t in region_tickers)
            if region_count > 0:
                exposure[region] = region_count / total * 100
        return exposure

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_and_align_data_cached(
    selected_tickers: Tuple[str, ...],
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    min_data_length: int = 100
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Robust data fetching with proper alignment and forward-filling.
    """
    warnings_list: List[str] = []
    infos_list: List[str] = []

    if not selected_tickers:
        return pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["No tickers selected."]}

    benchmark_ticker = "^GSPC"
    all_tickers = list(set(list(selected_tickers) + [benchmark_ticker]))

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        data = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
            timeout=30
        )
    except TypeError:
        data = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False
        )

    prices_dict: Dict[str, pd.Series] = {}
    data_quality: Dict[str, Dict] = {}

    for ticker in all_tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.levels[0] and ("Close" in data[ticker].columns):
                    prices_dict[ticker] = data[ticker]["Close"].rename(ticker)
            else:
                if "Close" in data.columns:
                    prices_dict[ticker] = data["Close"].rename(ticker)

            if ticker in prices_dict:
                series = prices_dict[ticker]
                original_length = len(series)
                non_na_count = series.count()
                na_count = original_length - non_na_count
                na_percentage = (na_count / original_length * 100) if original_length > 0 else 100.0
                start_dt = series.dropna().index.min() if non_na_count > 0 else None
                end_dt = series.dropna().index.max() if non_na_count > 0 else None
                data_days = (end_dt - start_dt).days if (start_dt is not None and end_dt is not None and non_na_count > 1) else 0

                data_quality[ticker] = {
                    "original_length": int(original_length),
                    "non_na_count": int(non_na_count),
                    "na_count": int(na_count),
                    "na_percentage": float(na_percentage),
                    "start_date": start_dt.strftime("%Y-%m-%d") if start_dt is not None else None,
                    "end_date": end_dt.strftime("%Y-%m-%d") if end_dt is not None else None,
                    "data_days": int(data_days)
                }

        except Exception as e:
            warnings_list.append(f"Could not extract data for {ticker}: {str(e)[:120]}")
            continue

    if not prices_dict:
        return pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["No data could be fetched for any ticker."]}

    df_raw = pd.DataFrame(prices_dict)

    valid_tickers = []
    removed_insufficient = []
    for ticker in df_raw.columns:
        non_na_count = int(df_raw[ticker].count())
        if non_na_count >= int(min_data_length):
            valid_tickers.append(ticker)
        else:
            removed_insufficient.append((ticker, non_na_count))

    for ticker, npts in removed_insufficient:
        warnings_list.append(f"Removing {ticker}: insufficient data ({npts} points, min={min_data_length}).")

    if not valid_tickers:
        return pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["No tickers have sufficient data."]}

    df_filtered = df_raw[valid_tickers]

    infos_list.append(f"Aligning time series... Original shape: {df_filtered.shape}")

    df_filled = df_filtered.ffill().bfill()
    df_clean = df_filled.dropna()

    if len(df_clean) < int(min_data_length):
        return pd.DataFrame(), pd.Series(dtype=float), {
            "warnings": [f"Insufficient data after cleaning. Only {len(df_clean)} data points available."],
            "total_rows": int(len(df_clean))
        }

    selected = list(dict.fromkeys(selected_tickers))
    portfolio_tickers = [t for t in selected if t in df_clean.columns]

    benchmark_data = pd.Series(dtype=float)
    if benchmark_ticker in df_clean.columns:
        benchmark_data = df_clean[benchmark_ticker].copy()
        if benchmark_ticker not in selected:
            df_portfolio = df_clean[portfolio_tickers].copy()
        else:
            cols = list(dict.fromkeys(portfolio_tickers + [benchmark_ticker]))
            df_portfolio = df_clean[cols].copy()
    else:
        df_portfolio = df_clean[portfolio_tickers].copy()
        warnings_list.append("Benchmark data not available for comparison (^GSPC).")

    final_report = {
        "initial_tickers": int(len(all_tickers)),
        "valid_tickers": int(len(valid_tickers)),
        "final_tickers": int(len(df_portfolio.columns)),
        "total_rows": int(len(df_clean)),
        "start_date": df_clean.index.min().strftime("%Y-%m-%d"),
        "end_date": df_clean.index.max().strftime("%Y-%m-%d"),
        "data_range_days": int((df_clean.index.max() - df_clean.index.min()).days),
        "ticker_details": data_quality,
        "alignment_status": "SUCCESS" if len(df_clean) >= int(min_data_length) else "FAILED",
        "missing_data_summary": {
            ticker: float(data_quality.get(ticker, {}).get("na_percentage", 100.0))
            for ticker in valid_tickers if ticker in data_quality
        },
        "warnings": warnings_list,
        "infos": infos_list,
    }

    return df_portfolio, benchmark_data, final_report

# ==============================================================================
# 5) BOLLINGER BANDS ANALYZER (NEW)
# ==============================================================================

class BollingerBandsAnalyzer:
    """Advanced Bollinger Bands analysis with interactive visualization"""
    
    def __init__(self, df_prices: pd.DataFrame, window: int = 20, num_std: float = 2.0):
        self.df_prices = df_prices
        self.window = window
        self.num_std = num_std
        self.results = {}
        
    def calculate_bollinger_bands(self, ticker: str) -> pd.DataFrame:
        """Calculate Bollinger Bands for a specific ticker"""
        if ticker not in self.df_prices.columns:
            return pd.DataFrame()
        
        prices = self.df_prices[ticker].dropna()
        
        # Calculate moving average
        sma = prices.rolling(window=self.window).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        
        # Calculate band width and %B
        band_width = (upper_band - lower_band) / sma * 100
        percent_b = (prices - lower_band) / (upper_band - lower_band) * 100
        
        # Identify signals
        above_upper = prices > upper_band
        below_lower = prices < lower_band
        in_band = ~above_upper & ~below_lower
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Price': prices,
            'SMA': sma,
            'Upper Band': upper_band,
            'Lower Band': lower_band,
            'Band Width': band_width,
            '%B': percent_b,
            'Above Upper': above_upper,
            'Below Lower': below_lower,
            'In Band': in_band
        })
        
        # Store summary statistics
        self.results[ticker] = {
            'current_price': float(prices.iloc[-1]) if len(prices) > 0 else 0,
            'current_sma': float(sma.iloc[-1]) if len(sma.dropna()) > 0 else 0,
            'current_upper': float(upper_band.iloc[-1]) if len(upper_band.dropna()) > 0 else 0,
            'current_lower': float(lower_band.iloc[-1]) if len(lower_band.dropna()) > 0 else 0,
            'current_band_width': float(band_width.iloc[-1]) if len(band_width.dropna()) > 0 else 0,
            'current_percent_b': float(percent_b.iloc[-1]) if len(percent_b.dropna()) > 0 else 0,
            'times_above_upper': int(above_upper.sum()),
            'times_below_lower': int(below_lower.sum()),
            'avg_time_outside_bands': float((above_upper.sum() + below_lower.sum()) / len(prices) * 100) if len(prices) > 0 else 0,
            'recent_signal': 'ABOVE UPPER' if above_upper.iloc[-1] else ('BELOW LOWER' if below_lower.iloc[-1] else 'IN BAND')
        }
        
        return results_df
    
    def create_bollinger_chart(self, ticker: str, ticker_name: str = None) -> go.Figure:
        """Create interactive Bollinger Bands chart"""
        if ticker not in self.df_prices.columns:
            return go.Figure()
        
        results_df = self.calculate_bollinger_bands(ticker)
        
        if results_df.empty:
            return go.Figure()
        
        # Get name for display
        display_name = ticker_name if ticker_name else ticker
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Price'],
            mode='lines',
            name='Price',
            line=dict(color='#1a237e', width=2),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add SMA line
        fig.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['SMA'],
            mode='lines',
            name=f'SMA ({self.window})',
            line=dict(color='#ff9800', width=1.5, dash='dash'),
            hovertemplate='Date: %{x}<br>SMA: $%{y:.2f}<extra></extra>'
        ))
        
        # Add upper band
        fig.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Upper Band'],
            mode='lines',
            name=f'Upper Band ({self.num_std}σ)',
            line=dict(color='#d32f2f', width=1),
            hovertemplate='Date: %{x}<br>Upper Band: $%{y:.2f}<extra></extra>'
        ))
        
        # Add lower band
        fig.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Lower Band'],
            mode='lines',
            name=f'Lower Band ({self.num_std}σ)',
            line=dict(color='#388e3c', width=1),
            hovertemplate='Date: %{x}<br>Lower Band: $%{y:.2f}<extra></extra>'
        ))
        
        # Fill between bands
        fig.add_trace(go.Scatter(
            x=results_df.index.tolist() + results_df.index.tolist()[::-1],
            y=results_df['Upper Band'].tolist() + results_df['Lower Band'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(41, 98, 255, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Bollinger Band',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add markers for signals
        above_mask = results_df['Above Upper']
        below_mask = results_df['Below Lower']
        
        if above_mask.any():
            fig.add_trace(go.Scatter(
                x=results_df.index[above_mask],
                y=results_df.loc[above_mask, 'Price'],
                mode='markers',
                name='Above Upper Band',
                marker=dict(color='#d32f2f', size=8, symbol='triangle-up'),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<br>Signal: ABOVE UPPER BAND<extra></extra>'
            ))
        
        if below_mask.any():
            fig.add_trace(go.Scatter(
                x=results_df.index[below_mask],
                y=results_df.loc[below_mask, 'Price'],
                mode='markers',
                name='Below Lower Band',
                marker=dict(color='#388e3c', size=8, symbol='triangle-down'),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<br>Signal: BELOW LOWER BAND<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Bollinger Bands Analysis: {display_name} (Window: {self.window}, Std: {self.num_std})",
            title_font_color="#1a237e",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            hovermode="x unified",
            height=600,
            font_color="#424242",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add annotations for current status
        current_status = self.results.get(ticker, {}).get('recent_signal', 'N/A')
        status_color = '#d32f2f' if current_status == 'ABOVE UPPER' else ('#388e3c' if current_status == 'BELOW LOWER' else '#ff9800')
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Current Status: <b>{current_status}</b>",
            showarrow=False,
            font=dict(size=12, color=status_color),
            bgcolor="white",
            bordercolor=status_color,
            borderwidth=1,
            borderpad=4
        )
        
        return fig
    
    def create_band_width_chart(self, ticker: str, ticker_name: str = None) -> go.Figure:
        """Create band width chart"""
        if ticker not in self.df_prices.columns:
            return go.Figure()
        
        results_df = self.calculate_bollinger_bands(ticker)
        
        if results_df.empty:
            return go.Figure()
        
        display_name = ticker_name if ticker_name else ticker
        
        fig = go.Figure()
        
        # Add band width line
        fig.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Band Width'],
            mode='lines',
            name='Band Width',
            line=dict(color='#7b1fa2', width=2),
            hovertemplate='Date: %{x}<br>Band Width: %{y:.1f}%<extra></extra>'
        ))
        
        # Add %B line on secondary y-axis
        fig.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['%B'],
            mode='lines',
            name='%B',
            line=dict(color='#ff9800', width=1.5),
            yaxis="y2",
            hovertemplate='Date: %{x}<br>%B: %{y:.1f}%<extra></extra>'
        ))
        
        # Add horizontal lines for %B thresholds
        fig.add_hline(y=80, line_dash="dash", line_color="#d32f2f", opacity=0.5, annotation_text="80% - Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="#388e3c", opacity=0.5, annotation_text="20% - Oversold")
        
        fig.update_layout(
            title=f"Band Width & %B Analysis: {display_name}",
            title_font_color="#1a237e",
            xaxis_title="Date",
            yaxis_title="Band Width (%)",
            yaxis2=dict(
                title="%B",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            template="plotly_white",
            hovermode="x unified",
            height=400,
            font_color="#424242",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def get_summary_statistics(self, ticker: str) -> pd.DataFrame:
        """Get summary statistics for Bollinger Bands analysis"""
        if ticker not in self.results:
            self.calculate_bollinger_bands(ticker)
        
        stats = self.results.get(ticker, {})
        
        summary_data = [
            {"Metric": "Current Price", "Value": f"${stats.get('current_price', 0):.2f}"},
            {"Metric": f"{self.window}-day SMA", "Value": f"${stats.get('current_sma', 0):.2f}"},
            {"Metric": f"Upper Band ({self.num_std}σ)", "Value": f"${stats.get('current_upper', 0):.2f}"},
            {"Metric": f"Lower Band ({self.num_std}σ)", "Value": f"${stats.get('current_lower', 0):.2f}"},
            {"Metric": "Band Width", "Value": f"{stats.get('current_band_width', 0):.1f}%"},
            {"Metric": "%B", "Value": f"{stats.get('current_percent_b', 0):.1f}%"},
            {"Metric": "Current Signal", "Value": stats.get('recent_signal', 'N/A')},
            {"Metric": "Times Above Upper", "Value": f"{stats.get('times_above_upper', 0)}"},
            {"Metric": "Times Below Lower", "Value": f"{stats.get('times_below_lower', 0)}"},
            {"Metric": "% Time Outside Bands", "Value": f"{stats.get('avg_time_outside_bands', 0):.1f}%"}
        ]
        
        return pd.DataFrame(summary_data)

# ==============================================================================
# 6) OHLC CHART AND TRACKING ERROR ANALYZER (NEW) - FIXED
# ==============================================================================

class OHLCAndTrackingErrorAnalyzer:
    """OHLC chart visualization and tracking error analysis"""
    
    def __init__(self):
        self.ohlc_data = {}
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_ohlc_data(_self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLC data for a specific ticker"""
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=30
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                # Extract the first level if it exists
                data = data.droplevel(level=0, axis=1) if data.columns.nlevels > 1 else data
            
            # Ensure we have OHLC columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Check if we have the required columns
            available_cols = [col for col in required_cols if col in data.columns]
            
            if not available_cols:
                return pd.DataFrame()
            
            return data[available_cols]
            
        except Exception as e:
            st.warning(f"Could not fetch OHLC data for {ticker}: {e}")
            return pd.DataFrame()
    
    def create_ohlc_chart(self, ticker: str, start_date: str, end_date: str, 
                         ticker_name: str = None) -> go.Figure:
        """Create interactive OHLC candlestick chart - FIXED VERSION"""
        ohlc_df = self.fetch_ohlc_data(ticker, start_date, end_date)
        
        if ohlc_df.empty:
            return go.Figure()
        
        display_name = ticker_name if ticker_name else ticker
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=ohlc_df.index,
            open=ohlc_df['Open'],
            high=ohlc_df['High'],
            low=ohlc_df['Low'],
            close=ohlc_df['Close'],
            name='Price',
            increasing_line_color='#388e3c',
            decreasing_line_color='#d32f2f'
        )])
        
        # Add volume as bar chart on secondary y-axis
        if 'Volume' in ohlc_df.columns:
            fig.add_trace(go.Bar(
                x=ohlc_df.index,
                y=ohlc_df['Volume'],
                name='Volume',
                marker_color='rgba(158, 158, 158, 0.5)',
                yaxis='y2'
            ))
        
        # Calculate and add moving averages
        for window in [20, 50, 200]:
            if len(ohlc_df) >= window:
                ma = ohlc_df['Close'].rolling(window=window).mean()
                fig.add_trace(go.Scatter(
                    x=ohlc_df.index,
                    y=ma,
                    mode='lines',
                    name=f'MA{window}',
                    line=dict(width=1),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title=f"OHLC Chart: {display_name}",
            title_font_color="#1a237e",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            ) if 'Volume' in ohlc_df.columns else None,
            template="plotly_white",
            height=600,
            font_color="#424242",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Add technical indicators summary - FIXED
        if len(ohlc_df) > 0:
            # Extract scalar values properly
            current_close = ohlc_df['Close'].iloc[-1]
            if not isinstance(current_close, (int, float, np.number)):
                # Handle case where it's a Series or other type
                current_close = float(current_close.iloc[0]) if hasattr(current_close, 'iloc') else float(current_close)
            
            if len(ohlc_df) > 1:
                prev_close = ohlc_df['Close'].iloc[-2]
                if not isinstance(prev_close, (int, float, np.number)):
                    # Handle case where it's a Series or other type
                    prev_close = float(prev_close.iloc[0]) if hasattr(prev_close, 'iloc') else float(prev_close)
                
                # Check if prev_close is valid and not zero
                if isinstance(prev_close, (int, float, np.number)) and prev_close > 0:
                    change_pct = ((current_close - prev_close) / prev_close * 100)
                else:
                    change_pct = 0
            else:
                change_pct = 0
            
            # Format the change percentage
            change_text = f"Latest: ${current_close:.2f} ({change_pct:+.2f}%)"
            change_color = '#388e3c' if change_pct >= 0 else '#d32f2f'
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=change_text,
                showarrow=False,
                font=dict(size=12, color=change_color),
                bgcolor="white",
                bordercolor=change_color,
                borderwidth=1,
                borderpad=4
            )
        
        return fig
    
    def calculate_tracking_error(self, portfolio_returns: pd.Series, 
                                benchmark_returns: pd.Series) -> Dict:
        """Calculate tracking error and related metrics"""
        
        # Align returns
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if aligned.empty:
            return {}
        
        portfolio_aligned = aligned.iloc[:, 0]
        benchmark_aligned = aligned.iloc[:, 1]
        
        # Calculate tracking error (annualized)
        tracking_error_daily = (portfolio_aligned - benchmark_aligned).std()
        tracking_error_annual = tracking_error_daily * np.sqrt(252)
        
        # Calculate active return
        active_return_daily = portfolio_aligned.mean() - benchmark_aligned.mean()
        active_return_annual = active_return_daily * 252
        
        # Calculate information ratio
        information_ratio = active_return_annual / tracking_error_annual if tracking_error_annual > 0 else 0
        
        # Calculate correlation
        correlation = portfolio_aligned.corr(benchmark_aligned)
        
        # Calculate R-squared
        r_squared = correlation ** 2
        
        # Calculate beta
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha
        benchmark_annual_return = benchmark_aligned.mean() * 252
        risk_free_rate = 0.04  # Default 4%
        expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
        actual_return = portfolio_aligned.mean() * 252
        alpha = actual_return - expected_return
        
        # Calculate up/down capture ratios
        up_market = benchmark_aligned > 0
        down_market = benchmark_aligned < 0
        
        up_capture = 0
        down_capture = 0
        
        if up_market.any():
            up_capture = (portfolio_aligned[up_market].mean() / 
                         benchmark_aligned[up_market].mean() * 100)
        
        if down_market.any():
            down_capture = (portfolio_aligned[down_market].mean() / 
                           benchmark_aligned[down_market].mean() * 100)
        
        return {
            'tracking_error_annual': float(tracking_error_annual * 100),  # as percentage
            'tracking_error_daily': float(tracking_error_daily * 100),
            'active_return_annual': float(active_return_annual * 100),
            'information_ratio': float(information_ratio),
            'correlation': float(correlation),
            'r_squared': float(r_squared * 100),  # as percentage
            'beta': float(beta),
            'alpha': float(alpha * 100),  # as percentage
            'up_capture': float(up_capture),
            'down_capture': float(down_capture),
            'capture_ratio': float(up_capture / down_capture) if down_capture != 0 else 0,
            'observations': len(aligned)
        }
    
    def create_tracking_error_chart(self, portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series,
                                  portfolio_name: str = "Portfolio",
                                  benchmark_name: str = "Benchmark") -> go.Figure:
        """Create tracking error visualization"""
        
        metrics = self.calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        if not metrics:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cumulative Returns", "Tracking Error Over Time",
                          "Rolling Correlation", "Active Return Distribution"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Align returns for plotting
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio_aligned = aligned.iloc[:, 0]
        benchmark_aligned = aligned.iloc[:, 1]
        
        # 1. Cumulative Returns
        cum_portfolio = (1 + portfolio_aligned).cumprod()
        cum_benchmark = (1 + benchmark_aligned).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=cum_portfolio.index,
                y=cum_portfolio.values,
                mode='lines',
                name=portfolio_name,
                line=dict(color='#1a237e', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=cum_benchmark.index,
                y=cum_benchmark.values,
                mode='lines',
                name=benchmark_name,
                line=dict(color='#ff9800', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Tracking Error Over Time (rolling 30-day)
        if len(portfolio_aligned) >= 30:
            rolling_te = (portfolio_aligned - benchmark_aligned).rolling(30).std() * np.sqrt(252) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_te.index,
                    y=rolling_te.values,
                    mode='lines',
                    name='Tracking Error',
                    line=dict(color='#d32f2f', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(239, 83, 80, 0.1)'
                ),
                row=1, col=2
            )
            
            # Add average line
            fig.add_hline(
                y=metrics['tracking_error_annual'],
                line_dash="dash",
                line_color="#d32f2f",
                annotation_text=f"Avg: {metrics['tracking_error_annual']:.2f}%",
                row=1, col=2
            )
        
        # 3. Rolling Correlation (30-day)
        if len(portfolio_aligned) >= 30:
            rolling_corr = portfolio_aligned.rolling(30).corr(benchmark_aligned)
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    name='Rolling Correlation',
                    line=dict(color='#7b1fa2', width=2)
                ),
                row=2, col=1
            )
            
            # Add average line
            fig.add_hline(
                y=metrics['correlation'],
                line_dash="dash",
                line_color="#7b1fa2",
                annotation_text=f"Avg: {metrics['correlation']:.2f}",
                row=2, col=1
            )
        
        # 4. Active Return Distribution
        active_returns = (portfolio_aligned - benchmark_aligned) * 100  # as percentage
        
        fig.add_trace(
            go.Histogram(
                x=active_returns.values,
                nbinsx=50,
                name='Active Returns',
                marker_color='rgba(41, 98, 255, 0.7)',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Add mean line
        fig.add_vline(
            x=metrics['active_return_annual'] / 252,  # Convert annual to daily
            line_dash="dash",
            line_color="#1a237e",
            annotation_text=f"Mean: {metrics['active_return_annual']:.2f}% (annual)",
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white",
            title_text=f"Tracking Error Analysis: {portfolio_name} vs {benchmark_name}",
            title_font_size=16,
            title_font_color="#1a237e",
            font_color="#424242"
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1, title_font_color="#424242")
        fig.update_xaxes(title_text="Date", row=1, col=2, title_font_color="#424242")
        fig.update_yaxes(title_text="Tracking Error (%)", row=1, col=2, title_font_color="#424242")
        fig.update_xaxes(title_text="Date", row=2, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Correlation", row=2, col=1, title_font_color="#424242")
        fig.update_xaxes(title_text="Daily Active Return (%)", row=2, col=2, title_font_color="#424242")
        fig.update_yaxes(title_text="Frequency", row=2, col=2, title_font_color="#424242")
        
        return fig

# ==============================================================================
# 7) ENHANCED STRESS TESTING ENGINE WITH SHOCK SIMULATOR (NEW)
# ==============================================================================

class EnhancedStressTestEngine:
    """Enhanced stress testing with historical events and custom scenarios"""
    
    # Expanded historical crisis database
    HISTORICAL_CRISES = {
        "Global Financial Crisis (2008)": {
            "start": "2007-10-09", 
            "end": "2009-03-09",
            "description": "Subprime mortgage crisis leading to global banking collapse",
            "severity": "Extreme",
            "max_drawdown_global": -56.78,
            "recovery_days": 1465,
            "asset_class_impact": {
                "equities": -55.0,
                "bonds": 20.0,
                "real_estate": -40.0,
                "commodities": -50.0,
                "credit_spread": 600  # bps
            }
        },
        "COVID-19 Pandemic Crash (2020)": {
            "start": "2020-02-19", 
            "end": "2020-03-23",
            "description": "Global pandemic causing fastest bear market in history",
            "severity": "Severe",
            "max_drawdown_global": -33.92,
            "recovery_days": 153,
            "asset_class_impact": {
                "equities": -34.0,
                "bonds": 15.0,
                "real_estate": -25.0,
                "commodities": -60.0,
                "credit_spread": 300  # bps
            }
        },
        "Dot-com Bubble Burst (2000)": {
            "start": "2000-03-10", 
            "end": "2002-10-09",
            "description": "Tech stock bubble collapse",
            "severity": "Severe",
            "max_drawdown_global": -49.15,
            "recovery_days": 1825,
            "asset_class_impact": {
                "equities": -49.0,
                "bonds": 25.0,
                "real_estate": -15.0,
                "commodities": -20.0,
                "credit_spread": 200  # bps
            }
        },
        "2022 Inflation & Rate Hikes": {
            "start": "2022-01-03", 
            "end": "2022-10-12",
            "description": "Aggressive central bank tightening to combat inflation",
            "severity": "Moderate",
            "max_drawdown_global": -25.44,
            "recovery_days": 280,
            "asset_class_impact": {
                "equities": -25.0,
                "bonds": -20.0,
                "real_estate": -30.0,
                "commodities": 15.0,
                "credit_spread": 150  # bps
            }
        },
        "European Debt Crisis (2011)": {
            "start": "2011-05-02", 
            "end": "2011-10-03",
            "description": "Sovereign debt crisis in Eurozone countries",
            "severity": "Moderate",
            "max_drawdown_global": -21.58,
            "recovery_days": 155,
            "asset_class_impact": {
                "equities": -22.0,
                "bonds": 5.0,
                "real_estate": -15.0,
                "commodities": -25.0,
                "credit_spread": 400  # bps
            }
        },
        "Turkey 2018 Currency Crisis": {
            "start": "2018-08-01", 
            "end": "2018-09-01",
            "description": "Lira depreciation and economic sanctions",
            "severity": "Severe",
            "max_drawdown_global": -35.67,
            "recovery_days": 420,
            "asset_class_impact": {
                "equities": -36.0,
                "bonds": -50.0,
                "real_estate": -40.0,
                "commodities": -30.0,
                "credit_spread": 800  # bps
            }
        },
        "Russia-Ukraine War (2022)": {
            "start": "2022-02-24", 
            "end": "2022-03-08",
            "description": "Geopolitical conflict causing energy and commodity shocks",
            "severity": "High",
            "max_drawdown_global": -12.44,
            "recovery_days": 45,
            "asset_class_impact": {
                "equities": -12.0,
                "bonds": -8.0,
                "real_estate": -5.0,
                "commodities": 50.0,
                "credit_spread": 100  # bps
            }
        },
        "China Market Crash (2015)": {
            "start": "2015-06-12", 
            "end": "2015-08-26",
            "description": "Chinese stock market bubble burst",
            "severity": "High",
            "max_drawdown_global": -43.34,
            "recovery_days": 1050,
            "asset_class_impact": {
                "equities": -43.0,
                "bonds": 2.0,
                "real_estate": -20.0,
                "commodities": -30.0,
                "credit_spread": 120  # bps
            }
        },
        "US Debt Ceiling Crisis (2011)": {
            "start": "2011-07-22", 
            "end": "2011-08-08",
            "description": "Political standoff over US debt limit",
            "severity": "Moderate",
            "max_drawdown_global": -16.77,
            "recovery_days": 60,
            "asset_class_impact": {
                "equities": -17.0,
                "bonds": -5.0,
                "real_estate": -8.0,
                "commodities": -12.0,
                "credit_spread": 80  # bps
            }
        },
        "Brexit Referendum (2016)": {
            "start": "2016-06-23", 
            "end": "2016-06-27",
            "description": "UK votes to leave European Union",
            "severity": "Moderate",
            "max_drawdown_global": -5.85,
            "recovery_days": 15,
            "asset_class_impact": {
                "equities": -6.0,
                "bonds": 8.0,
                "real_estate": -15.0,
                "commodities": -10.0,
                "credit_spread": 50  # bps
            }
        },
        "US Banking Turmoil (2023)": {
            "start": "2023-03-01", 
            "end": "2023-03-31",
            "description": "Regional bank failures (SVB, Signature, Credit Suisse)",
            "severity": "Moderate",
            "max_drawdown_global": -8.76,
            "recovery_days": 90,
            "asset_class_impact": {
                "equities": -9.0,
                "bonds": 12.0,
                "real_estate": -5.0,
                "commodities": -8.0,
                "credit_spread": 200  # bps
            }
        },
        "Flash Crash (2010)": {
            "start": "2010-05-06", 
            "end": "2010-05-06",
            "description": "Intraday market crash of ~9% in minutes",
            "severity": "High",
            "max_drawdown_global": -9.03,
            "recovery_days": 1,
            "asset_class_impact": {
                "equities": -9.0,
                "bonds": 1.0,
                "real_estate": 0.0,
                "commodities": -5.0,
                "credit_spread": 20  # bps
            }
        },
        "Oil Price War (2020)": {
            "start": "2020-03-06", 
            "end": "2020-04-20",
            "description": "Saudi-Russia oil price war during COVID pandemic",
            "severity": "High",
            "max_drawdown_global": -65.23,
            "recovery_days": 180,
            "asset_class_impact": {
                "equities": -30.0,
                "bonds": 10.0,
                "real_estate": -15.0,
                "commodities": -65.0,
                "credit_spread": 400  # bps
            }
        },
        "Emerging Markets Crisis (2018)": {
            "start": "2018-01-26", 
            "end": "2018-12-24",
            "description": "Fed tightening causing EM capital outflows",
            "severity": "Moderate",
            "max_drawdown_global": -19.78,
            "recovery_days": 210,
            "asset_class_impact": {
                "equities": -20.0,
                "bonds": -25.0,
                "real_estate": -18.0,
                "commodities": -22.0,
                "credit_spread": 350  # bps
            }
        },
        "Taper Tantrum (2013)": {
            "start": "2013-05-22", 
            "end": "2013-09-05",
            "description": "Fed announces QE taper, bond market selloff",
            "severity": "Moderate",
            "max_drawdown_global": -5.76,
            "recovery_days": 45,
            "asset_class_impact": {
                "equities": -6.0,
                "bonds": -8.0,
                "real_estate": -4.0,
                "commodities": -10.0,
                "credit_spread": 100  # bps
            }
        },
        "Asian Financial Crisis (1997)": {
            "start": "1997-07-02", 
            "end": "1998-12-31",
            "description": "Currency and financial crisis across Asia",
            "severity": "Severe",
            "max_drawdown_global": -42.0,
            "recovery_days": 730,
            "asset_class_impact": {
                "equities": -42.0,
                "bonds": -60.0,
                "real_estate": -50.0,
                "commodities": -35.0,
                "credit_spread": 1000  # bps
            }
        },
        "LTCM Collapse (1998)": {
            "start": "1998-08-17", 
            "end": "1998-10-15",
            "description": "Hedge fund collapse causing liquidity crisis",
            "severity": "High",
            "max_drawdown_global": -19.3,
            "recovery_days": 60,
            "asset_class_impact": {
                "equities": -19.0,
                "bonds": 5.0,
                "real_estate": -10.0,
                "commodities": -15.0,
                "credit_spread": 150  # bps
            }
        },
        "9/11 Attacks (2001)": {
            "start": "2001-09-11", 
            "end": "2001-09-21",
            "description": "Terrorist attacks causing market shutdown",
            "severity": "High",
            "max_drawdown_global": -11.6,
            "recovery_days": 30,
            "asset_class_impact": {
                "equities": -12.0,
                "bonds": 8.0,
                "real_estate": -5.0,
                "commodities": 10.0,
                "credit_spread": 80  # bps
            }
        },
        "Volmageddon (2018)": {
            "start": "2018-02-02", 
            "end": "2018-02-08",
            "description": "Volatility spike causing ETF liquidation",
            "severity": "Moderate",
            "max_drawdown_global": -10.2,
            "recovery_days": 15,
            "asset_class_impact": {
                "equities": -10.0,
                "bonds": -2.0,
                "real_estate": -3.0,
                "commodities": -8.0,
                "credit_spread": 30  # bps
            }
        }
    }
    
    class ShockSimulator:
        """Interactive shock simulator for stress testing"""
        
        def __init__(self, portfolio_returns: pd.Series):
            self.portfolio_returns = portfolio_returns
            self.base_volatility = portfolio_returns.std()
            self.base_return = portfolio_returns.mean()
            
        def apply_shock(self, shock_params: Dict) -> Dict:
            """Apply shock to portfolio returns based on parameters"""
            
            shock_type = shock_params.get("type", "market_crash")
            magnitude = shock_params.get("magnitude", 0.3)  # 30% shock
            duration = shock_params.get("duration", 30)  # 30 days
            volatility_multiplier = shock_params.get("volatility_multiplier", 3.0)
            recovery_speed = shock_params.get("recovery_speed", "medium")
            sector_bias = shock_params.get("sector_bias", "broad")
            
            # Generate shocked returns
            shocked_returns = []
            
            if shock_type == "market_crash":
                # Sharp initial drop followed by recovery
                crash_days = int(duration * 0.3)
                recovery_days = duration - crash_days
                
                # Crash phase
                crash_mean = -magnitude / crash_days
                crash_vol = self.base_volatility * volatility_multiplier
                crash_returns = np.random.normal(crash_mean, crash_vol, crash_days)
                
                # Recovery phase
                if recovery_speed == "fast":
                    recovery_mean = 0.002
                elif recovery_speed == "medium":
                    recovery_mean = 0.001
                else:  # slow
                    recovery_mean = 0.0005
                
                recovery_vol = self.base_volatility * (volatility_multiplier * 0.7)
                recovery_returns = np.random.normal(recovery_mean, recovery_vol, recovery_days)
                
                shocked_returns = np.concatenate([crash_returns, recovery_returns])
                
            elif shock_type == "volatility_spike":
                # Sustained high volatility
                shock_mean = 0.0
                shock_vol = self.base_volatility * volatility_multiplier
                shocked_returns = np.random.normal(shock_mean, shock_vol, duration)
                
            elif shock_type == "slow_bleed":
                # Gradual decline
                daily_decline = -magnitude / duration
                shock_vol = self.base_volatility * volatility_multiplier * 0.5
                shocked_returns = np.random.normal(daily_decline, shock_vol, duration)
                
            elif shock_type == "flash_crash":
                # Single day crash with partial recovery
                crash_day = np.array([-magnitude])
                recovery_days = duration - 1
                recovery_returns = np.random.normal(0.001, self.base_volatility * 2, recovery_days)
                shocked_returns = np.concatenate([crash_day, recovery_returns])
                
            else:  # generic shock
                shock_mean = -magnitude / duration
                shock_vol = self.base_volatility * volatility_multiplier
                shocked_returns = np.random.normal(shock_mean, shock_vol, duration)
            
            # Calculate statistics
            cumulative_shock = (1 + shocked_returns).prod() - 1
            max_drawdown = self._calculate_max_drawdown(shocked_returns)
            volatility_during = np.std(shocked_returns)
            
            # Apply sector bias multiplier
            sector_multiplier = 1.0
            if sector_bias == "tech":
                sector_multiplier = 1.5
            elif sector_bias == "financial":
                sector_multiplier = 1.3
            elif sector_bias == "defensive":
                sector_multiplier = 0.7
            
            cumulative_shock *= sector_multiplier
            max_drawdown *= sector_multiplier
            
            return {
                "shock_type": shock_type,
                "magnitude": magnitude,
                "duration": duration,
                "cumulative_impact": float(cumulative_shock * 100),  # as percentage
                "max_drawdown_shock": float(max_drawdown * 100),  # as percentage
                "volatility_during": float(volatility_during),
                "volatility_multiplier": volatility_multiplier,
                "recovery_speed": recovery_speed,
                "sector_bias": sector_bias,
                "shocked_returns": shocked_returns
            }
        
        def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
            """Calculate maximum drawdown from return series"""
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        
        def create_shock_comparison_chart(self, shock_results: List[Dict]) -> go.Figure:
            """Create comparison chart for multiple shock scenarios"""
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Cumulative Impact", "Maximum Drawdown",
                              "Return Distribution", "Volatility During Shock"),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "box"}, {"type": "bar"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            scenario_names = [f"{r['shock_type'].replace('_', ' ').title()}" 
                            for r in shock_results]
            cumulative_impacts = [r['cumulative_impact'] for r in shock_results]
            max_drawdowns = [r['max_drawdown_shock'] for r in shock_results]
            volatilities = [r['volatility_during'] * 100 for r in shock_results]  # as percentage
            
            # 1. Cumulative Impact
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=cumulative_impacts,
                    name='Cumulative Impact',
                    marker_color='rgba(239, 83, 80, 0.8)',
                    text=[f"{x:.1f}%" for x in cumulative_impacts],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # 2. Maximum Drawdown
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=max_drawdowns,
                    name='Max Drawdown',
                    marker_color='rgba(41, 98, 255, 0.8)',
                    text=[f"{x:.1f}%" for x in max_drawdowns],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. Return Distribution (box plots)
            for i, result in enumerate(shock_results):
                fig.add_trace(
                    go.Box(
                        y=result['shocked_returns'] * 100,  # as percentage
                        name=scenario_names[i],
                        boxpoints='outliers',
                        marker_color='rgba(255, 152, 0, 0.7)',
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # 4. Volatility During Shock
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=volatilities,
                    name='Volatility',
                    marker_color='rgba(76, 175, 80, 0.8)',
                    text=[f"{x:.1f}%" for x in volatilities],
                    textposition='auto'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                template="plotly_white",
                title_text="Shock Scenario Comparison Analysis",
                title_font_size=16,
                title_font_color="#1a237e",
                font_color="#424242"
            )
            
            fig.update_xaxes(title_text="Scenario", row=1, col=1, tickangle=45, title_font_color="#424242")
            fig.update_yaxes(title_text="Cumulative Impact (%)", row=1, col=1, title_font_color="#424242")
            fig.update_xaxes(title_text="Scenario", row=1, col=2, tickangle=45, title_font_color="#424242")
            fig.update_yaxes(title_text="Max Drawdown (%)", row=1, col=2, title_font_color="#424242")
            fig.update_xaxes(title_text="Scenario", row=2, col=1, tickangle=45, title_font_color="#424242")
            fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1, title_font_color="#424242")
            fig.update_xaxes(title_text="Scenario", row=2, col=2, tickangle=45, title_font_color="#424242")
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=2, title_font_color="#424242")
            
            return fig

# ==============================================================================
# 8) NEW UI TABS FOR ADDED FEATURES
# ==============================================================================

def create_bollinger_bands_tab():
    """Create Bollinger Bands analysis tab"""
    st.markdown('<div class="section-header">📊 Bollinger Bands Technical Analysis</div>', unsafe_allow_html=True)
    
    if "df_prices" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please load data first from the Data Overview tab.</div>', unsafe_allow_html=True)
        return
    
    df_prices = st.session_state["df_prices"]
    ticker_map = st.session_state.get("ticker_map", {})
    
    # Configuration
    st.markdown('<div class="subsection-header">⚙️ Bollinger Bands Configuration</div>', unsafe_allow_html=True)
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        selected_ticker = st.selectbox(
            "Select Instrument",
            options=list(df_prices.columns),
            format_func=lambda x: ticker_map.get(x, x),
            key="bb_ticker"
        )
    
    with col_config2:
        window_size = st.slider(
            "Moving Average Window",
            min_value=10,
            max_value=50,
            value=20,
            step=1,
            key="bb_window"
        )
    
    with col_config3:
        num_std = st.slider(
            "Standard Deviations",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1,
            key="bb_std"
        )
    
    # Initialize analyzer
    bb_analyzer = BollingerBandsAnalyzer(df_prices, window=window_size, num_std=num_std)
    
    # Calculate and display
    with st.spinner("🔬 Calculating Bollinger Bands..."):
        # Get summary statistics
        summary_df = bb_analyzer.get_summary_statistics(selected_ticker)
        
        # Display summary metrics
        st.markdown('<div class="subsection-header">📈 Current Band Status</div>', unsafe_allow_html=True)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        current_stats = bb_analyzer.results.get(selected_ticker, {})
        
        with col_stat1:
            st.metric("Current Price", f"${current_stats.get('current_price', 0):.2f}")
        
        with col_stat2:
            st.metric(f"SMA ({window_size})", f"${current_stats.get('current_sma', 0):.2f}")
        
        with col_stat3:
            current_signal = current_stats.get('recent_signal', 'N/A')
            signal_color = "#d32f2f" if current_signal == "ABOVE UPPER" else \
                         "#388e3c" if current_signal == "BELOW LOWER" else "#ff9800"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Current Signal</div>
                    <div class="metric-value" style="color: {signal_color};">{current_signal}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_stat4:
            st.metric("%B", f"{current_stats.get('current_percent_b', 0):.1f}%")
    
    # Create charts
    st.markdown('<div class="subsection-header">📊 Interactive Bollinger Bands Charts</div>', unsafe_allow_html=True)
    
    # Main Bollinger Bands chart
    ticker_name = ticker_map.get(selected_ticker, selected_ticker)
    bb_chart = bb_analyzer.create_bollinger_chart(selected_ticker, ticker_name)
    st.plotly_chart(bb_chart, use_container_width=True, use_container_height=True)
    
    # Band width and %B chart
    st.markdown('<div class="subsection-header">📈 Band Width & %B Analysis</div>', unsafe_allow_html=True)
    width_chart = bb_analyzer.create_band_width_chart(selected_ticker, ticker_name)
    st.plotly_chart(width_chart, use_container_width=True)
    
    # Detailed statistics table
    st.markdown('<div class="subsection-header">📋 Detailed Statistics</div>', unsafe_allow_html=True)
    st.dataframe(summary_df, use_container_width=True, height=400)
    
    # Multi-instrument comparison (optional)
    with st.expander("🔍 Compare Multiple Instruments"):
        selected_tickers = st.multiselect(
            "Select instruments to compare",
            options=list(df_prices.columns),
            default=[selected_ticker],
            format_func=lambda x: ticker_map.get(x, x),
            key="bb_compare_tickers"
        )
        
        if len(selected_tickers) > 1:
            comparison_data = []
            for ticker in selected_tickers:
                stats = bb_analyzer.get_summary_statistics(ticker)
                if not stats.empty:
                    name = ticker_map.get(ticker, ticker)
                    comparison_data.append({
                        "Instrument": name,
                        "Current Signal": stats.iloc[6]["Value"],
                        "%B": stats.iloc[5]["Value"],
                        "Band Width": stats.iloc[4]["Value"]
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

def create_ohlc_tracking_error_tab():
    """Create OHLC chart and tracking error analysis tab"""
    st.markdown('<div class="section-header">📈 OHLC Charts & Tracking Error Analysis</div>', unsafe_allow_html=True)
    
    if "df_prices" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please load data first from the Data Overview tab.</div>', unsafe_allow_html=True)
        return
    
    df_prices = st.session_state["df_prices"]
    ticker_map = st.session_state.get("ticker_map", {})
    
    # Create tabs for OHLC and Tracking Error
    tab_ohlc, tab_tracking = st.tabs(["📊 OHLC Charts", "📈 Tracking Error Analysis"])
    
    with tab_ohlc:
        st.markdown('<div class="subsection-header">📊 OHLC Candlestick Charts</div>', unsafe_allow_html=True)
        
        col_ohlc1, col_ohlc2, col_ohlc3 = st.columns(3)
        
        with col_ohlc1:
            selected_ticker = st.selectbox(
                "Select Instrument",
                options=list(df_prices.columns),
                format_func=lambda x: ticker_map.get(x, x),
                key="ohlc_ticker"
            )
        
        with col_ohlc2:
            # Date range for OHLC
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Default 1 year
            date_range = st.date_input(
                "Date Range",
                value=(start_date, end_date),
                key="ohlc_date_range"
            )
        
        with col_ohlc3:
            show_volume = st.checkbox("Show Volume", value=True, key="ohlc_show_volume")
            show_moving_averages = st.checkbox("Show Moving Averages", value=True, key="ohlc_show_ma")
        
        # Initialize analyzer
        ohlc_analyzer = OHLCAndTrackingErrorAnalyzer()
        
        # Create OHLC chart
        if len(date_range) == 2:
            start_str = date_range[0].strftime("%Y-%m-%d")
            end_str = date_range[1].strftime("%Y-%m-%d")
            
            with st.spinner("📈 Generating OHLC chart..."):
                ticker_name = ticker_map.get(selected_ticker, selected_ticker)
                ohlc_chart = ohlc_analyzer.create_ohlc_chart(
                    selected_ticker, start_str, end_str, ticker_name
                )
                st.plotly_chart(ohlc_chart, use_container_width=True, use_container_height=True)
        
        # Technical indicators summary
        st.markdown('<div class="subsection-header">📊 Technical Indicators</div>', unsafe_allow_html=True)
        
        if len(date_range) == 2:
            # Fetch OHLC data for calculations
            start_str = date_range[0].strftime("%Y-%m-%d")
            end_str = date_range[1].strftime("%Y-%m-%d")
            ohlc_data = ohlc_analyzer.fetch_ohlc_data(selected_ticker, start_str, end_str)
            
            if not ohlc_data.empty:
                # Calculate basic technical indicators
                prices = ohlc_data['Close']
                
                col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)
                
                with col_tech1:
                    # RSI (14-day)
                    if len(prices) >= 14:
                        delta = prices.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]
                        st.metric("RSI (14)", f"{current_rsi:.1f}")
                
                with col_tech2:
                    # MACD
                    if len(prices) >= 26:
                        exp1 = prices.ewm(span=12, adjust=False).mean()
                        exp2 = prices.ewm(span=26, adjust=False).mean()
                        macd = exp1 - exp2
                        signal = macd.ewm(span=9, adjust=False).mean()
                        current_macd = macd.iloc[-1]
                        st.metric("MACD", f"{current_macd:.2f}")
                
                with col_tech3:
                    # Average True Range
                    if len(ohlc_data) >= 14:
                        high_low = ohlc_data['High'] - ohlc_data['Low']
                        high_close = np.abs(ohlc_data['High'] - ohlc_data['Close'].shift())
                        low_close = np.abs(ohlc_data['Low'] - ohlc_data['Close'].shift())
                        ranges = pd.concat([high_low, high_close, low_close], axis=1)
                        true_range = np.max(ranges, axis=1)
                        atr = true_range.rolling(14).mean()
                        current_atr = atr.iloc[-1]
                        st.metric("ATR (14)", f"{current_atr:.2f}")
                
                with col_tech4:
                    # Volume analysis
                    avg_volume = ohlc_data['Volume'].mean()
                    current_volume = ohlc_data['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    st.metric("Volume Ratio", f"{volume_ratio:.1f}x")
    
    with tab_tracking:
        st.markdown('<div class="subsection-header">📈 Tracking Error Analysis</div>', unsafe_allow_html=True)
        
        # Check if we have portfolio returns
        if "portfolio_returns" not in st.session_state:
            st.markdown('<div class="warning-card">⚠️ Please run portfolio optimization first to generate portfolio returns.</div>', unsafe_allow_html=True)
            return
        
        portfolio_returns = st.session_state["portfolio_returns"]
        
        # Benchmark selection
        col_track1, col_track2 = st.columns(2)
        
        with col_track1:
            # Get available benchmark tickers
            benchmark_options = list(df_prices.columns)
            benchmark_options.insert(0, "^GSPC")  # Add S&P 500 as default
            
            selected_benchmark = st.selectbox(
                "Select Benchmark",
                options=benchmark_options,
                format_func=lambda x: ticker_map.get(x, x) if x in ticker_map else x,
                key="tracking_benchmark"
            )
        
        with col_track2:
            portfolio_name = st.text_input("Portfolio Name", value="My Portfolio", key="tracking_port_name")
            benchmark_name = st.text_input("Benchmark Name", value="S&P 500", key="tracking_bench_name")
        
        # Fetch benchmark returns
        with st.spinner("📊 Calculating benchmark returns..."):
            if selected_benchmark in df_prices.columns:
                benchmark_prices = df_prices[selected_benchmark]
            else:
                # Fetch benchmark data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*5)  # 5 years
                try:
                    bench_data = yf.download(
                        selected_benchmark,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        progress=False
                    )
                    benchmark_prices = bench_data['Close'] if 'Close' in bench_data.columns else pd.Series()
                except:
                    st.error(f"Could not fetch data for benchmark: {selected_benchmark}")
                    return
            
            if benchmark_prices.empty:
                st.error("No benchmark data available")
                return
            
            benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # Initialize analyzer
        tracking_analyzer = OHLCAndTrackingErrorAnalyzer()
        
        # Calculate tracking error
        with st.spinner("🔬 Calculating tracking error metrics..."):
            metrics = tracking_analyzer.calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        if metrics:
            # Display key metrics
            st.markdown('<div class="subsection-header">🎯 Key Tracking Metrics</div>', unsafe_allow_html=True)
            
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            
            with col_metric1:
                st.metric("Tracking Error", f"{metrics['tracking_error_annual']:.2f}%")
            
            with col_metric2:
                st.metric("Information Ratio", f"{metrics['information_ratio']:.2f}")
            
            with col_metric3:
                st.metric("Beta", f"{metrics['beta']:.2f}")
            
            with col_metric4:
                st.metric("Alpha", f"{metrics['alpha']:.2f}%")
            
            # Create tracking error chart
            st.markdown('<div class="subsection-header">📊 Tracking Error Visualization</div>', unsafe_allow_html=True)
            tracking_chart = tracking_analyzer.create_tracking_error_chart(
                portfolio_returns, benchmark_returns, portfolio_name, benchmark_name
            )
            st.plotly_chart(tracking_chart, use_container_width=True)
            
            # Detailed metrics table
            st.markdown('<div class="subsection-header">📋 Detailed Tracking Metrics</div>', unsafe_allow_html=True)
            
            detailed_metrics = [
                {"Metric": "Annual Tracking Error", "Value": f"{metrics['tracking_error_annual']:.2f}%"},
                {"Metric": "Daily Tracking Error", "Value": f"{metrics['tracking_error_daily']:.2f}%"},
                {"Metric": "Annual Active Return", "Value": f"{metrics['active_return_annual']:.2f}%"},
                {"Metric": "Information Ratio", "Value": f"{metrics['information_ratio']:.2f}"},
                {"Metric": "Correlation", "Value": f"{metrics['correlation']:.3f}"},
                {"Metric": "R-squared", "Value": f"{metrics['r_squared']:.2f}%"},
                {"Metric": "Beta", "Value": f"{metrics['beta']:.3f}"},
                {"Metric": "Alpha", "Value": f"{metrics['alpha']:.2f}%"},
                {"Metric": "Up Capture Ratio", "Value": f"{metrics['up_capture']:.1f}%"},
                {"Metric": "Down Capture Ratio", "Value": f"{metrics['down_capture']:.1f}%"},
                {"Metric": "Capture Ratio", "Value": f"{metrics['capture_ratio']:.2f}"},
                {"Metric": "Observations", "Value": f"{metrics['observations']}"}
            ]
            
            detailed_df = pd.DataFrame(detailed_metrics)
            st.dataframe(detailed_df, use_container_width=True)

def create_enhanced_stress_test_tab():
    """Create enhanced stress testing tab with shock simulator"""
    st.markdown('<div class="section-header">⚠️ Enhanced Stress Testing & Shock Simulator</div>', unsafe_allow_html=True)
    
    if "portfolio_returns" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please run portfolio optimization first to generate portfolio returns.</div>', unsafe_allow_html=True)
        return
    
    portfolio_returns = st.session_state["portfolio_returns"]
    
    # Initialize stress test engine
    stress_engine = EnhancedStressTestEngine()
    shock_simulator = stress_engine.ShockSimulator(portfolio_returns)
    
    # Create tabs for different stress test features
    tab_historical, tab_shock, tab_custom = st.tabs([
        "📜 Historical Crises",
        "⚡ Shock Simulator",
        "🛠️ Custom Scenarios"
    ])
    
    with tab_historical:
        st.markdown('<div class="subsection-header">📜 Historical Financial Crises Analysis</div>', unsafe_allow_html=True)
        
        # Display historical crises timeline
        st.markdown("##### 📅 Historical Crises Timeline")
        
        # Create a DataFrame for display
        crises_list = []
        for name, details in stress_engine.HISTORICAL_CRISES.items():
            crises_list.append({
                "Crisis": name,
                "Start Date": details["start"],
                "End Date": details["end"],
                "Duration": f"{(datetime.strptime(details['end'], '%Y-%m-%d') - datetime.strptime(details['start'], '%Y-%m-%d')).days} days",
                "Severity": details["severity"],
                "Max Drawdown": f"{details['max_drawdown_global']:.1f}%",
                "Recovery": f"{details['recovery_days']} days"
            })
        
        crises_df = pd.DataFrame(crises_list)
        
        # Let user select crises to analyze
        selected_crises = st.multiselect(
            "Select Historical Crises to Analyze",
            options=list(stress_engine.HISTORICAL_CRISES.keys()),
            default=["COVID-19 Pandemic Crash (2020)", "2022 Inflation & Rate Hikes"],
            key="historical_crises_select"
        )
        
        if selected_crises and st.button("📈 Analyze Selected Crises", use_container_width=True):
            historical_results = []
            
            for crisis_name in selected_crises:
                crisis_details = stress_engine.HISTORICAL_CRISES[crisis_name]
                
                # Extract returns during crisis period
                mask = (
                    (portfolio_returns.index >= crisis_details["start"]) & 
                    (portfolio_returns.index <= crisis_details["end"])
                )
                crisis_returns = portfolio_returns.loc[mask]
                
                if len(crisis_returns) > 5:
                    total_return = (1 + crisis_returns).prod() - 1
                    max_dd = ((1 + crisis_returns).cumprod() / 
                             (1 + crisis_returns).cumprod().cummax() - 1).min()
                    vol = crisis_returns.std() * np.sqrt(252)
                    
                    historical_results.append({
                        "Crisis": crisis_name,
                        "Period": f"{crisis_details['start']} to {crisis_details['end']}",
                        "Days": len(crisis_returns),
                        "Portfolio Return": float(total_return * 100),
                        "Portfolio Max DD": float(max_dd * 100),
                        "Portfolio Volatility": float(vol * 100),
                        "Global Max DD": crisis_details["max_drawdown_global"],
                        "Severity": crisis_details["severity"],
                        "Description": crisis_details["description"]
                    })
            
            if historical_results:
                df_historical = pd.DataFrame(historical_results)
                
                # Display metrics
                st.markdown('<div class="subsection-header">📈 Historical Crisis Impact on Portfolio</div>', unsafe_allow_html=True)
                col_hist1, col_hist2, col_hist3 = st.columns(3)
                
                avg_return = df_historical["Portfolio Return"].mean()
                avg_drawdown = df_historical["Portfolio Max DD"].mean()
                worst_drawdown = df_historical["Portfolio Max DD"].min()
                
                with col_hist1:
                    st.metric("Average Portfolio Return", f"{avg_return:.2f}%")
                
                with col_hist2:
                    st.metric("Average Maximum Drawdown", f"{avg_drawdown:.2f}%")
                
                with col_hist3:
                    st.metric("Worst Historical Drawdown", f"{worst_drawdown:.2f}%")
                
                # Display detailed table
                st.dataframe(
                    df_historical.style.format({
                        "Portfolio Return": "{:.2f}%",
                        "Portfolio Max DD": "{:.2f}%",
                        "Portfolio Volatility": "{:.2f}%"
                    }),
                    use_container_width=True,
                    height=300
                )
            else:
                st.markdown('<div class="info-card">ℹ️ No overlapping data found for selected historical crises.</div>', unsafe_allow_html=True)
    
    with tab_shock:
        st.markdown('<div class="subsection-header">⚡ Interactive Shock Simulator</div>', unsafe_allow_html=True)
        
        # Shock parameters configuration
        st.markdown("##### ⚙️ Shock Parameters")
        
        col_shock1, col_shock2 = st.columns(2)
        
        with col_shock1:
            shock_type = st.selectbox(
                "Shock Type",
                options=["market_crash", "volatility_spike", "slow_bleed", "flash_crash"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="shock_type"
            )
            
            magnitude = st.slider(
                "Shock Magnitude (%)",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=0.5,
                key="shock_magnitude"
            ) / 100  # Convert to decimal
        
        with col_shock2:
            duration = st.slider(
                "Shock Duration (Days)",
                min_value=5,
                max_value=100,
                value=30,
                step=1,
                key="shock_duration"
            )
            
            volatility_multiplier = st.slider(
                "Volatility Multiplier",
                min_value=1.0,
                max_value=5.0,
                value=2.5,
                step=0.1,
                key="shock_vol_mult"
            )
        
        col_shock3, col_shock4 = st.columns(2)
        
        with col_shock3:
            recovery_speed = st.select_slider(
                "Recovery Speed",
                options=["slow", "medium", "fast"],
                value="medium",
                key="shock_recovery"
            )
        
        with col_shock4:
            sector_bias = st.selectbox(
                "Sector Bias",
                options=["broad", "tech", "financial", "defensive"],
                key="shock_sector"
            )
        
        # Run shock simulation
        if st.button("🚨 Run Shock Simulation", type="primary", use_container_width=True):
            shock_params = {
                "type": shock_type,
                "magnitude": magnitude,
                "duration": duration,
                "volatility_multiplier": volatility_multiplier,
                "recovery_speed": recovery_speed,
                "sector_bias": sector_bias
            }
            
            with st.spinner("⚡ Simulating shock scenario..."):
                shock_result = shock_simulator.apply_shock(shock_params)
            
            if shock_result:
                st.markdown('<div class="success-card">✅ Shock simulation completed!</div>', unsafe_allow_html=True)
                
                # Display results
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric(
                        "Cumulative Impact",
                        f"{shock_result['cumulative_impact']:.1f}%"
                    )
                
                with col_res2:
                    st.metric(
                        "Maximum Drawdown",
                        f"{shock_result['max_drawdown_shock']:.1f}%"
                    )
                
                with col_res3:
                    st.metric(
                        "Volatility During Shock",
                        f"{shock_result['volatility_during']*100:.1f}%"
                    )
                
                # Store shock result for comparison
                if "shock_results" not in st.session_state:
                    st.session_state.shock_results = []
                
                st.session_state.shock_results.append(shock_result)
                
                # Show shock path
                st.markdown('<div class="subsection-header">📊 Shock Path Visualization</div>', unsafe_allow_html=True)
                
                fig_shock = go.Figure()
                
                # Add shock returns
                fig_shock.add_trace(go.Scatter(
                    x=list(range(len(shock_result['shocked_returns']))),
                    y=np.cumprod(1 + shock_result['shocked_returns']) * 100,
                    mode='lines',
                    name='Shock Path',
                    line=dict(color='#d32f2f', width=3),
                    hovertemplate='Day: %{x}<br>Cumulative: %{y:.1f}%<extra></extra>'
                ))
                
                # Add baseline (no shock)
                fig_shock.add_hline(
                    y=100,
                    line_dash="dash",
                    line_color="#1a237e",
                    annotation_text="Baseline (100%)"
                )
                
                fig_shock.update_layout(
                    title=f"Shock Simulation: {shock_type.replace('_', ' ').title()}",
                    title_font_color="#1a237e",
                    xaxis_title="Days",
                    yaxis_title="Cumulative Performance (%)",
                    template="plotly_white",
                    height=400,
                    font_color="#424242"
                )
                
                st.plotly_chart(fig_shock, use_container_width=True)
        
        # Compare multiple shock scenarios
        if "shock_results" in st.session_state and len(st.session_state.shock_results) > 1:
            st.markdown('<div class="subsection-header">📈 Shock Scenario Comparison</div>', unsafe_allow_html=True)
            
            if st.button("📊 Compare All Shock Scenarios", use_container_width=True):
                comparison_chart = shock_simulator.create_shock_comparison_chart(st.session_state.shock_results)
                st.plotly_chart(comparison_chart, use_container_width=True)
            
            if st.button("🗑️ Clear Shock Scenarios", use_container_width=True):
                st.session_state.shock_results = []
                st.rerun()
    
    with tab_custom:
        st.markdown('<div class="subsection-header">🛠️ Custom Stress Scenario Builder</div>', unsafe_allow_html=True)
        
        # Custom scenario builder
        with st.expander("🔧 Build Custom Stress Scenario", expanded=True):
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            col_custom1, col_custom2 = st.columns(2)
            
            with col_custom1:
                scenario_name = st.text_input("Scenario Name", "My Custom Stress Test")
                
                scenario_type = st.selectbox(
                    "Scenario Type",
                    ["market_crash", "sector_collapse", "liquidity_crisis", "geopolitical_event"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="custom_scenario_type"
                )
                
                magnitude_custom = st.slider(
                    "Impact Magnitude", 
                    min_value=0.1, 
                    max_value=0.8, 
                    value=0.3, 
                    step=0.05,
                    help="Expected portfolio loss (0.3 = 30% loss)",
                    key="custom_magnitude"
                )
            
            with col_custom2:
                duration_custom = st.slider(
                    "Scenario Duration (Days)", 
                    min_value=5, 
                    max_value=250, 
                    value=30, 
                    step=5,
                    key="custom_duration"
                )
                
                volatility_impact = st.slider(
                    "Volatility Impact", 
                    min_value=1.0, 
                    max_value=5.0, 
                    value=2.5, 
                    step=0.5,
                    help="How many times normal volatility during stress",
                    key="custom_vol_impact"
                )
                
                include_correlation_breakdown = st.checkbox(
                    "Include Correlation Breakdown", 
                    value=True,
                    help="Assume correlations increase during stress",
                    key="custom_corr_breakdown"
                )
            
            # Advanced parameters
            with st.expander("⚙️ Advanced Parameters"):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    tail_risk_adjustment = st.slider(
                        "Tail Risk Adjustment",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        help="Adjustment for extreme losses (higher = fatter tails)",
                        key="custom_tail_risk"
                    )
                    
                    contagion_effect = st.checkbox(
                        "Include Contagion Effect",
                        value=True,
                        help="Losses spread from initial shock to other assets",
                        key="custom_contagion"
                    )
                
                with col_adv2:
                    market_cap_weighted = st.checkbox(
                        "Market Cap Weighted Impact",
                        value=True,
                        help="Larger companies experience proportionally larger impacts",
                        key="custom_market_cap"
                    )
                    
                    liquidity_impact = st.slider(
                        "Liquidity Impact", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.5, 
                    step=0.1,
                        help="0 = normal liquidity, 1 = severely impaired liquidity",
                        key="custom_liquidity"
                    )
            
            if st.button("🚀 Run Custom Scenario Simulation", use_container_width=True):
                # Create scenario
                custom_scenario = {
                    "name": scenario_name,
                    "type": scenario_type,
                    "magnitude": magnitude_custom,
                    "duration": duration_custom,
                    "volatility_multiplier": volatility_impact,
                    "correlation_breakdown": include_correlation_breakdown,
                    "tail_risk": tail_risk_adjustment,
                    "contagion_effect": contagion_effect,
                    "market_cap_weighted": market_cap_weighted,
                    "liquidity_impact": liquidity_impact
                }
                
                st.success(f"Custom scenario '{scenario_name}' created successfully!")
                st.json(custom_scenario)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================


# ==============================================================================
# 8B) INSTITUTIONAL PERFORMANCE / RISK / PORTFOLIO ENGINES (NEW)
# ==============================================================================

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series(dtype=float)

def _annualize_return(daily_returns: pd.Series, periods: int = 252) -> float:
    daily_returns = _to_series(daily_returns).dropna()
    if daily_returns.empty:
        return 0.0
    comp = (1.0 + daily_returns).prod()
    years = len(daily_returns) / periods
    if years <= 0:
        return 0.0
    return float(comp ** (1.0 / years) - 1.0)

def _annualize_vol(daily_returns: pd.Series, periods: int = 252) -> float:
    daily_returns = _to_series(daily_returns).dropna()
    if daily_returns.empty:
        return 0.0
    return float(daily_returns.std() * np.sqrt(periods))

def _max_drawdown(daily_returns: pd.Series) -> float:
    daily_returns = _to_series(daily_returns).dropna()
    if daily_returns.empty:
        return 0.0
    equity = (1.0 + daily_returns).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def _drawdown_series(daily_returns: pd.Series) -> pd.Series:
    daily_returns = _to_series(daily_returns).dropna()
    if daily_returns.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + daily_returns).cumprod()
    peak = equity.cummax()
    return (equity / peak) - 1.0

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in (w or {}).items() if v is not None}
    if not w:
        return {}
    s = sum(w.values())
    if abs(s) < 1e-12:
        return {k: 0.0 for k in w}
    return {k: float(v) / s for k, v in w.items()}

def _weights_to_df(w: Dict[str, float], ticker_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    w = _normalize_weights(w)
    rows = []
    for t, v in sorted(w.items(), key=lambda kv: kv[1], reverse=True):
        rows.append({
            "Ticker": t,
            "Name": (ticker_map or {}).get(t, t),
            "Weight": float(v),
            "Weight %": float(v) * 100.0,
        })
    return pd.DataFrame(rows)

def _portfolio_returns(returns_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    if returns_df is None or returns_df.empty:
        return pd.Series(dtype=float)
    w = _normalize_weights({k: v for k, v in (weights or {}).items() if k in returns_df.columns})
    if not w:
        cols = list(returns_df.columns)
        if not cols:
            return pd.Series(dtype=float)
        w = {c: 1.0 / len(cols) for c in cols}
    w_vec = pd.Series(w).reindex(returns_df.columns).fillna(0.0)
    pr = returns_df.dot(w_vec)
    pr.name = "Portfolio"
    return pr.dropna()

def _plot_weights_pie(weights: Dict[str, float], ticker_map: Optional[Dict[str, str]] = None, title: str = "Weights") -> go.Figure:
    dfw = _weights_to_df(weights, ticker_map)
    if dfw.empty:
        return go.Figure()
    labels = dfw["Name"].tolist()
    values = dfw["Weight %"].tolist()
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:.2f}%<extra></extra>",
    )])
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig

@st.cache_data(ttl=24*3600, show_spinner=False)
def _fetch_market_caps(tickers: Tuple[str, ...]) -> Dict[str, float]:
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            mc = info.get("market_cap", None) if isinstance(info, dict) else None
            if mc is None:
                mc = yf.Ticker(t).info.get("marketCap", None)
            if mc is not None:
                caps[t] = float(mc)
        except Exception:
            continue
    return caps

class PerformanceAnalyticsEngine:
    # Institutional performance metrics and visuals (portfolio + asset level).
    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = float(risk_free_rate)

    def compute_metrics(self, r: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        r = _to_series(r).dropna()
        b = _to_series(benchmark).dropna() if benchmark is not None else pd.Series(dtype=float)
        if r.empty:
            return {}

        ann_ret = _annualize_return(r)
        ann_vol = _annualize_vol(r)
        sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0.0

        downside = r[r < 0]
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
        sortino = (ann_ret - self.rf) / downside_vol if downside_vol > 0 else 0.0

        mdd = _max_drawdown(r)
        calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0

        skew = float(stats.skew(r, nan_policy="omit")) if len(r) > 3 else 0.0
        kurt = float(stats.kurtosis(r, fisher=True, nan_policy="omit")) if len(r) > 3 else 0.0

        jb_stat, jb_p = jarque_bera(r.values) if len(r) > 10 else (np.nan, np.nan)

        # Basic VaR/CVaR (95%) in % terms
        q = np.quantile(r, 0.05)
        var95 = float(-q)
        tail = r[r <= q]
        cvar95 = float(-tail.mean()) if len(tail) > 0 else float("nan")

        metrics = {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": mdd,
            "Calmar": calmar,
            "Skew": skew,
            "Kurtosis": kurt,
            "JB p-value": float(jb_p) if jb_p == jb_p else float("nan"),
            "VaR 95%": var95,
            "ES/CVaR 95%": cvar95,
        }

        # Benchmark-relative
        if benchmark is not None and not b.empty:
            aligned = pd.concat([r, b], axis=1).dropna()
            if not aligned.empty and aligned.shape[0] > 3:
                rr = aligned.iloc[:, 0]
                bb = aligned.iloc[:, 1]
                cov = np.cov(rr, bb)[0, 1]
                varb = np.var(bb)
                beta = cov / varb if varb > 0 else 0.0
                alpha = (rr.mean() * 252) - (self.rf + beta * ((bb.mean() * 252) - self.rf))
                te = (rr - bb).std() * np.sqrt(252)
                ir = (((rr.mean() - bb.mean()) * 252) / te) if te > 0 else 0.0
                metrics.update({
                    "Beta": float(beta),
                    "Alpha (ann.)": float(alpha),
                    "Tracking Error": float(te),
                    "Info Ratio": float(ir),
                })

        return metrics

    def metrics_table(self, returns_df: pd.DataFrame, portfolio_r: Optional[pd.Series] = None,
                      benchmark: Optional[pd.Series] = None, ticker_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        rows = []
        if portfolio_r is not None and not portfolio_r.empty:
            pm = self.compute_metrics(portfolio_r, benchmark=benchmark)
            rows.append({"Asset": "PORTFOLIO", **pm})
        for c in returns_df.columns:
            m = self.compute_metrics(returns_df[c], benchmark=benchmark)
            rows.append({"Asset": (ticker_map or {}).get(c, c), **m})
        df = pd.DataFrame(rows)
        preferred = [
            "Asset","Ann. Return","Ann. Vol","Sharpe","Sortino","Max Drawdown","Calmar",
            "VaR 95%","ES/CVaR 95%","Beta","Alpha (ann.)","Tracking Error","Info Ratio",
            "Skew","Kurtosis","JB p-value"
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        return df[cols]

    def performance_charts(self, portfolio_r: pd.Series, benchmark: Optional[pd.Series] = None,
                           title: str = "Performance Dashboard") -> go.Figure:
        r = _to_series(portfolio_r).dropna()
        if r.empty:
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cumulative Returns", "Drawdown (Underwater)",
                            "Rolling Volatility (60D)", "Rolling Sharpe (60D)"),
            specs=[[{"type":"scatter"},{"type":"scatter"}],
                   [{"type":"scatter"},{"type":"scatter"}]],
            vertical_spacing=0.15, horizontal_spacing=0.12
        )

        cum = (1 + r).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Portfolio"), row=1, col=1)

        if benchmark is not None and not _to_series(benchmark).dropna().empty:
            b = _to_series(benchmark).dropna()
            aligned = pd.concat([r, b], axis=1).dropna()
            if not aligned.empty:
                cb = (1 + aligned.iloc[:,1]).cumprod()
                fig.add_trace(go.Scatter(x=cb.index, y=cb.values, mode="lines", name="Benchmark", line=dict(dash="dash")), row=1, col=1)

        dd = _drawdown_series(r)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown", fill="tozeroy"), row=1, col=2)

        roll_vol = r.rolling(60).std() * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, mode="lines", name="Rolling Vol (%)"), row=2, col=1)

        roll_mean = r.rolling(60).mean() * 252
        roll_sh = (roll_mean - self.rf) / (r.rolling(60).std() * np.sqrt(252))
        fig.add_trace(go.Scatter(x=roll_sh.index, y=roll_sh.values, mode="lines", name="Rolling Sharpe"), row=2, col=2)

        fig.update_layout(
            template="plotly_white",
            height=820,
            title=title,
            title_font_color="#1a237e",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        )
        return fig

class RiskAnalyticsEngine:
    # VaR / ES in Historical, Parametric, Monte Carlo modes + Relative Risk
    @staticmethod
    def _losses(r: pd.Series) -> pd.Series:
        r = _to_series(r).dropna()
        return -r  # positive losses

    def historical_var_es(self, r: pd.Series, cl: float = 0.95) -> Tuple[float, float]:
        loss = self._losses(r)
        if loss.empty:
            return 0.0, 0.0
        var = float(np.quantile(loss, cl))
        tail = loss[loss >= var]
        es = float(tail.mean()) if len(tail) > 0 else var
        return var, es

    def parametric_var_es(self, r: pd.Series, cl: float = 0.95, dist: str = "normal", df: int = 6) -> Tuple[float, float]:
        r = _to_series(r).dropna()
        if r.empty:
            return 0.0, 0.0
        mu = float(r.mean())
        sigma = float(r.std())
        if sigma <= 0:
            return 0.0, 0.0

        if dist == "t":
            q = stats.t.ppf(1 - cl, df=df)
            var = -(mu + sigma * q)
            pdf = stats.t.pdf(q, df=df)
            cdf = stats.t.cdf(q, df=df)
            es_r = mu + sigma * (((df + q*q) / (df - 1)) * (pdf / cdf)) if cdf > 0 else (mu + sigma*q)
            es = -es_r
            return float(var), float(es)
        else:
            z = stats.norm.ppf(1 - cl)
            var = -(mu + sigma * z)
            es_r = mu - sigma * (stats.norm.pdf(z) / (1 - cl))
            es = -es_r
            return float(var), float(es)

    def monte_carlo_var_es(self, r_df: pd.DataFrame, weights: Dict[str, float], cl: float = 0.95,
                           n_sims: int = 20000, horizon_days: int = 1,
                           dist: str = "normal", df: int = 6, seed: int = 42) -> Tuple[float, float]:
        if r_df is None or r_df.empty:
            return 0.0, 0.0
        cols = list(r_df.columns)
        w = _normalize_weights({k: v for k, v in (weights or {}).items() if k in cols})
        if not w:
            w = {c: 1.0 / len(cols) for c in cols}
        w_vec = pd.Series(w).reindex(cols).fillna(0.0).values.reshape(-1, 1)

        mu = r_df.mean().values.reshape(-1, 1)
        cov = r_df.cov().values
        cov = cov + 1e-12 * np.eye(cov.shape[0])

        rng = np.random.default_rng(seed)

        if dist == "t":
            L = np.linalg.cholesky(cov)
            z = rng.standard_normal(size=(cov.shape[0], n_sims))
            g = rng.chisquare(df, size=n_sims)
            scale = np.sqrt(df / g)
            sims = (mu + (L @ z) * scale)
        else:
            sims = rng.multivariate_normal(mean=mu.flatten(), cov=cov, size=n_sims).T

        if horizon_days > 1:
            sims = sims * np.sqrt(horizon_days)

        port = (w_vec.T @ sims).flatten()
        loss = -port
        var = float(np.quantile(loss, cl))
        tail = loss[loss >= var]
        es = float(np.mean(tail)) if len(tail) > 0 else var
        return var, es

    def build_risk_table(self, r_port: pd.Series, r_df: pd.DataFrame, weights: Dict[str, float],
                         cls: List[float], mc_sims: int = 20000) -> pd.DataFrame:
        rows = []
        for cl in cls:
            hv, hes = self.historical_var_es(r_port, cl=cl)
            pv, pes = self.parametric_var_es(r_port, cl=cl, dist="normal")
            tv, tes = self.parametric_var_es(r_port, cl=cl, dist="t", df=6)
            mcv, mces = self.monte_carlo_var_es(r_df, weights, cl=cl, n_sims=mc_sims, dist="normal")
            mctv, mctes = self.monte_carlo_var_es(r_df, weights, cl=cl, n_sims=mc_sims, dist="t", df=6)
            rows.extend([
                {"Confidence": cl, "Method": "Historical", "VaR": hv, "ES": hes},
                {"Confidence": cl, "Method": "Parametric (Normal)", "VaR": pv, "ES": pes},
                {"Confidence": cl, "Method": "Parametric (t)", "VaR": tv, "ES": tes},
                {"Confidence": cl, "Method": "Monte Carlo (Normal)", "VaR": mcv, "ES": mces},
                {"Confidence": cl, "Method": "Monte Carlo (t)", "VaR": mctv, "ES": mctes},
            ])
        df = pd.DataFrame(rows)
        df["VaR %"] = df["VaR"] * 100
        df["ES %"] = df["ES"] * 100
        return df

class PortfolioOptimizationEngine:
    # Institutional portfolio optimization wrapper around PyPortfolioOpt
    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = float(risk_free_rate)

    def _mu_sigma(self, prices: pd.DataFrame, mu_method: str = "mean", cov_method: str = "ledoit_wolf") -> Tuple[pd.Series, pd.DataFrame]:
        if mu_method == "ema":
            mu = expected_returns.ema_historical_return(prices, span=180)
        elif mu_method == "capm":
            mu = expected_returns.capm_return(prices, market_prices=prices.mean(axis=1))
        else:
            mu = expected_returns.mean_historical_return(prices)

        if cov_method == "exp_cov":
            S = risk_models.exp_cov(prices)
        elif cov_method == "sample":
            S = risk_models.sample_cov(prices)
        else:
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        return mu, S

    def mean_variance(self, prices: pd.DataFrame, mu_method: str, cov_method: str,
                      objective: str, max_weight: float = 0.35,
                      target_return: Optional[float] = None, target_risk: Optional[float] = None) -> Dict:
        mu, S = self._mu_sigma(prices, mu_method, cov_method)
        ef = EfficientFrontier(mu, S, weight_bounds=(0.0, float(max_weight)))
        try:
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        except Exception:
            pass

        if objective == "max_sharpe":
            ef.max_sharpe(risk_free_rate=self.rf)
        elif objective == "min_volatility":
            ef.min_volatility()
        elif objective == "max_quadratic_utility":
            ef.max_quadratic_utility(risk_aversion=2)
        elif objective == "efficient_return" and target_return is not None:
            ef.efficient_return(target_return=float(target_return))
        elif objective == "efficient_risk" and target_risk is not None:
            ef.efficient_risk(target_volatility=float(target_risk))
        else:
            ef.max_sharpe(risk_free_rate=self.rf)

        w = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=self.rf, verbose=False)
        return {
            "weights": w,
            "expected_return": float(perf[0]),
            "volatility": float(perf[1]),
            "sharpe": float(perf[2]),
            "mu": mu,
            "cov": S,
        }

    def cla(self, prices: pd.DataFrame, mu_method: str, cov_method: str, max_weight: float = 0.35) -> Dict:
        mu, S = self._mu_sigma(prices, mu_method, cov_method)
        cla = CLA(mu, S, weight_bounds=(0.0, float(max_weight)))
        cla.min_volatility()
        w = cla.clean_weights()
        w_vec = pd.Series(w).reindex(prices.columns).fillna(0.0)
        r = prices.pct_change().dropna().dot(w_vec)
        return {"weights": w, "realized_ann_return": _annualize_return(r), "realized_ann_vol": _annualize_vol(r)}

    def hrp(self, returns: pd.DataFrame) -> Dict:
        hrp = HRPOpt(returns)
        w = hrp.optimize()
        return {"weights": w}

    def semivariance(self, prices: pd.DataFrame, benchmark: float = 0.0, max_weight: float = 0.35) -> Dict:
        mu = expected_returns.mean_historical_return(prices)
        returns = prices.pct_change().dropna()
        esv = EfficientSemivariance(mu, returns, benchmark=float(benchmark), weight_bounds=(0.0, float(max_weight)))
        esv.min_semivariance()
        w = esv.clean_weights()
        return {"weights": w}

    def cvar(self, prices: pd.DataFrame, beta: float = 0.95, max_weight: float = 0.35) -> Dict:
        mu = expected_returns.mean_historical_return(prices)
        returns = prices.pct_change().dropna()
        ecvar = EfficientCVaR(mu, returns, beta=float(beta), weight_bounds=(0.0, float(max_weight)))
        ecvar.min_cvar()
        w = ecvar.clean_weights()
        return {"weights": w}

    def cdar(self, prices: pd.DataFrame, beta: float = 0.95, max_weight: float = 0.35) -> Dict:
        mu = expected_returns.mean_historical_return(prices)
        returns = prices.pct_change().dropna()
        ecdar = EfficientCDaR(mu, returns, beta=float(beta), weight_bounds=(0.0, float(max_weight)))
        ecdar.min_cdar()
        w = ecdar.clean_weights()
        return {"weights": w}

    def black_litterman(self, prices: pd.DataFrame, views_df: pd.DataFrame,
                        market_weight_mode: str = "equal",
                        delta_override: Optional[float] = None) -> Dict:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        tickers = list(prices.columns)

        if market_weight_mode == "market_cap":
            caps = _fetch_market_caps(tuple(tickers))
            if caps:
                total = sum(caps.values())
                w_mkt = {t: (caps.get(t, 0.0) / total) for t in tickers}
            else:
                w_mkt = {t: 1.0 / len(tickers) for t in tickers}
        else:
            w_mkt = {t: 1.0 / len(tickers) for t in tickers}

        if delta_override is None:
            try:
                market_prices = prices.mean(axis=1)
                delta = float(market_implied_risk_aversion(market_prices))
                if not np.isfinite(delta) or delta <= 0:
                    delta = 2.5
            except Exception:
                delta = 2.5
        else:
            delta = float(delta_override)

        pi = market_implied_prior_returns(w_mkt, delta, S)

        P = None
        Q = None
        if views_df is not None and not views_df.empty:
            views_df = views_df.copy()
            views_df["Ticker"] = views_df["Ticker"].astype(str)
            views_df = views_df[views_df["Ticker"].isin(tickers)]
            if not views_df.empty:
                P = np.zeros((len(views_df), len(tickers)))
                Q_list = []
                for i, row in enumerate(views_df.itertuples(index=False)):
                    t = getattr(row, "Ticker")
                    vret = float(getattr(row, "ViewReturn"))
                    idx_t = tickers.index(t)
                    P[i, idx_t] = 1.0
                    Q_list.append(vret)
                Q = np.array(Q_list)

        bl = BlackLittermanModel(S, pi=pi, P=P, Q=Q)
        ret_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()

        ef = EfficientFrontier(ret_bl, cov_bl, weight_bounds=(0.0, 1.0))
        ef.max_sharpe(risk_free_rate=self.rf)
        w = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=self.rf, verbose=False)
        return {
            "weights": w,
            "prior_returns": pi,
            "posterior_returns": ret_bl,
            "posterior_cov": cov_bl,
            "expected_return": float(perf[0]),
            "volatility": float(perf[1]),
            "sharpe": float(perf[2]),
        }

# 9) MAIN APPLICATION WITH ALL TABS
# ==============================================================================

def main():
    st.markdown('<div class="main-header">⚡ QUANTUM | Advanced Risk Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Initialize data manager
    dm = EnhancedDataManager()
    
    # Session state initialization
    if "stress_test_results" not in st.session_state:
        st.session_state.stress_test_results = []
    if "shock_results" not in st.session_state:
        st.session_state.shock_results = []
    if "quick_scenario" not in st.session_state:
        st.session_state.quick_scenario = None
    if "custom_shocks" not in st.session_state:
        st.session_state.custom_shocks = []
    if "bl_views" not in st.session_state:
        st.session_state.bl_views = []
    if "bl_confidences" not in st.session_state:
        st.session_state.bl_confidences = []
    if "selected_assets_preset" not in st.session_state:
        st.session_state.selected_assets_preset = None
    
    with st.sidebar:
        st.markdown('<div class="section-header" style="font-size: 1.4rem; margin-top: 0;">🌍 Global Asset Selection</div>', unsafe_allow_html=True)
        
        # Quick portfolio presets
        st.markdown('<div class="subsection-header" style="font-size: 1.1rem;">Quick Portfolios</div>', unsafe_allow_html=True)
        col_preset1, col_preset2, col_preset3 = st.columns(3)
        
        with col_preset1:
            if st.button("Global 60/40", use_container_width=True):
                st.session_state.selected_assets_preset = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
        
        with col_preset2:
            if st.button("Tech Growth", use_container_width=True):
                st.session_state.selected_assets_preset = ["QQQ", "XLK", "AAPL", "MSFT", "NVDA"]
        
        with col_preset3:
            if st.button("Emerging Mkts", use_container_width=True):
                st.session_state.selected_assets_preset = ["EEM", "THYAO.IS", "BABA", "005930.KS"]
        
        st.divider()
        
        # Asset selection
        selected_assets = []
        default_assets = ["SPY", "TLT", "GLD", "AAPL", "THYAO.IS"]
        if st.session_state.selected_assets_preset:
            default_assets = st.session_state.selected_assets_preset
        
        for category, assets in dm.universe.items():
            with st.expander(
                f"📊 {category}",
                expanded=(category in ["US ETFs (Major & Active)", "Global Mega Caps"])
            ):
                selected = st.multiselect(
                    f"Select from {category}",
                    options=list(assets.keys()),
                    default=[k for k in assets.keys() if assets[k] in default_assets],
                    key=f"select_{category}"
                )
                for s in selected:
                    selected_assets.append(assets[s])
        
        # De-duplicate tickers
        if selected_assets:
            before = len(selected_assets)
            selected_assets = list(dict.fromkeys(selected_assets))
            after = len(selected_assets)
            if after < before:
                st.info(
                    f"ℹ️ Removed {before-after} duplicate ticker selection(s)."
                )
        
        st.divider()
        
        # Data settings
        st.markdown('<div class="subsection-header" style="font-size: 1.1rem;">Data Settings</div>', unsafe_allow_html=True)
        start_date = st.date_input("Start Date", value=datetime(2018, 1, 1))
        min_data_length = st.slider("Minimum Data Points", 100, 1000, 252,
                                   help="Assets with fewer data points will be removed")
        
        # Risk settings
        st.markdown('<div class="subsection-header" style="font-size: 1.1rem;">🎯 Risk Parameters</div>', unsafe_allow_html=True)
        
        risk_free_rate = st.number_input(
            "Risk Free Rate (%)", 
            value=4.5, 
            step=0.1
        ) / 100
        
        investment_amount = st.number_input(
            "Investment Amount ($)", 
            value=1000000, 
            step=100000
        )
        
        # Store in session state
        st.session_state["last_rf_rate"] = float(risk_free_rate)
        st.session_state["last_amount"] = float(investment_amount)
        
        # Show regional exposure
        if selected_assets:
            exposure = dm.get_regional_exposure(selected_assets)
            st.markdown('<div class="subsection-header" style="font-size: 1.1rem;">🌐 Regional Exposure</div>', unsafe_allow_html=True)
            for region, pct in exposure.items():
                st.progress(pct / 100, text=f"{region}: {pct:.1f}%")
        
        st.divider()
        st.caption("Tip: If Streamlit Cloud shows a redacted ModuleNotFoundError, add a requirements.txt with the packages used.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not selected_assets:
        st.markdown('<div class="warning-card">⚠️ Please select at least one asset from the sidebar.</div>', unsafe_allow_html=True)
        return
    
    # Check for optimization library (PyPortfolioOpt)
    OPTIMIZATION_AVAILABLE = bool(PYPFOPT_AVAILABLE)
    if not OPTIMIZATION_AVAILABLE:
        st.markdown(
            f'<div class="warning-card">⚠️ PyPortfolioOpt is not available. Some portfolio tabs will be disabled.<br>'
            f'<span style="font-size:0.9rem; opacity:0.9;">Install in requirements.txt: PyPortfolioOpt, cvxpy, ecos. '
            f'Import error: {PYPFOPT_IMPORT_ERROR}</span></div>',
            unsafe_allow_html=True
        )
# Fetch and align data
    with st.spinner("🔄 Fetching and aligning data..."):
        df_prices, benchmark_data, data_report = _fetch_and_align_data_cached(
            selected_tickers=tuple(selected_assets),
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            min_data_length=min_data_length
        )
    
    if df_prices is None or df_prices.empty:
        st.markdown('<div class="warning-card">❌ No valid data available after alignment. Please select different assets or adjust date range.</div>', unsafe_allow_html=True)
        return
    
    st.markdown(f'<div class="success-card">✅ Data ready for analysis: {len(df_prices)} data points, {len(df_prices.columns)} assets</div>', unsafe_allow_html=True)
    
    # Store data in session state for other tabs
    st.session_state.df_prices = df_prices
    st.session_state["ticker_map"] = {t: t for t in df_prices.columns}
    st.session_state["returns_df"] = df_prices.pct_change().dropna()
    if isinstance(benchmark_data, pd.Series) and not benchmark_data.empty:
        st.session_state.benchmark_returns = benchmark_data.pct_change().dropna()
    st.session_state.ticker_map = dm.get_ticker_name_map()
    
    # Create enhanced tabs with ALL NEW TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📈 Data Overview",
        "📊 Advanced Performance",
        "🎯 Black-Litterman",
        "📈 Advanced Frontier",
        "📊 Bollinger Bands",
        "📈 OHLC & Tracking",
        "⚠️ Enhanced Stress Testing",
        "🎯 Portfolio Optimization",
        "🎲 Advanced VaR/ES",
        "📊 Risk Analytics",
        "🔗 Correlation Analysis"
    ])
    
    # --- TAB 1: DATA OVERVIEW ---
    with tab1:
        st.markdown('<div class="section-header">📊 Data Overview & Visualization</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="subsection-header">📈 Normalized Price Performance</div>', unsafe_allow_html=True)
        normalized = (df_prices / df_prices.iloc[0]) * 100
        
        fig_prices = px.line(
            normalized,
            title="All Assets Rebased to 100",
            labels={"value": "Index Value", "variable": "Asset"}
        )
        fig_prices.update_layout(
            template="plotly_white",
            height=500,
            hovermode="x unified",
            title_font_color="#1a237e",
            font_color="#424242"
        )
        st.plotly_chart(fig_prices, use_container_width=True)
        
        st.markdown('<div class="subsection-header">📊 Summary Statistics</div>', unsafe_allow_html=True)
        returns = df_prices.pct_change().dropna()
        returns = returns.loc[:, ~returns.columns.duplicated()]
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            mean_return = returns.mean().mean() * 252
            st.metric("Mean Return (Ann.)", f"{mean_return:.2%}")
        with col_stat2:
            avg_vol = returns.std().mean() * np.sqrt(252)
            st.metric("Avg Volatility (Ann.)", f"{avg_vol:.2%}")
        with col_stat3:
            avg_corr = returns.corr().values.mean()
            st.metric("Avg Correlation", f"{avg_corr:.2f}")
    
    # --- TAB 2: ADVANCED PERFORMANCE METRICS ---
    with tab2:
        # This would be your existing Advanced Performance tab
        st.markdown('<div class="section-header">📊 Advanced Performance Metrics</div>', unsafe_allow_html=True)

        ticker_map = st.session_state.get("ticker_map", {})
        rf = float(st.session_state.get("last_rf_rate", 0.04))
        benchmark_r = st.session_state.get("benchmark_returns", pd.Series(dtype=float))

        returns_df = df_prices.pct_change().dropna()
        returns_df = returns_df.loc[:, ~returns_df.columns.duplicated()]

        if returns_df.empty:
            st.markdown('<div class="warning-card">⚠️ Not enough data to compute returns.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="subsection-header">📌 Portfolio Definition</div>', unsafe_allow_html=True)

            cols = list(returns_df.columns)
            eq_w = {c: 1.0 / len(cols) for c in cols} if cols else {}

            # Collect available strategy weights from session state
            strategy_weights = {
                "Equal Weight": eq_w,
            }
            for key, label in [
                ("mv_weights", "Mean-Variance (EF)"),
                ("cla_weights", "CLA (Min Vol)"),
                ("hrp_weights", "HRP"),
                ("cvar_weights", "Min CVaR"),
                ("cdar_weights", "Min CDaR"),
                ("bl_weights", "Black-Litterman"),
            ]:
                w = st.session_state.get(key, None)
                if isinstance(w, dict) and len(w) > 0:
                    strategy_weights[label] = w

            choice = st.selectbox("Portfolio / Strategy", list(strategy_weights.keys()) + ["Custom Weights"], index=0)

            if choice == "Custom Weights":
                base = _weights_to_df(eq_w, ticker_map)[["Ticker", "Weight"]].copy()
                st.caption("Enter portfolio weights (they will be automatically normalized).")
                edited = st.data_editor(base, num_rows="fixed", use_container_width=True, key="perf_custom_weights")
                custom = {row["Ticker"]: float(row["Weight"]) for _, row in edited.iterrows()}
                w_active = _normalize_weights(custom)
            else:
                w_active = _normalize_weights(strategy_weights[choice])

            st.session_state["active_weights"] = w_active

            r_port = _portfolio_returns(returns_df, w_active)

            # KPIs
            pe = PerformanceAnalyticsEngine(risk_free_rate=rf)
            m = pe.compute_metrics(r_port, benchmark=benchmark_r if not benchmark_r.empty else None)

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Ann. Return", f"{m.get('Ann. Return', 0.0):.2%}")
            k2.metric("Ann. Vol", f"{m.get('Ann. Vol', 0.0):.2%}")
            k3.metric("Sharpe", f"{m.get('Sharpe', 0.0):.2f}")
            k4.metric("Max DD", f"{m.get('Max Drawdown', 0.0):.2%}")
            k5.metric("VaR 95%", f"{m.get('VaR 95%', 0.0):.2%}")

            st.plotly_chart(
                pe.performance_charts(
                    r_port,
                    benchmark=benchmark_r if not benchmark_r.empty else None,
                    title=f"Portfolio Performance — {choice}"
                ),
                use_container_width=True
            )

            st.markdown('<div class="subsection-header">📋 Institutional Metrics Table</div>', unsafe_allow_html=True)
            df_metrics = pe.metrics_table(
                returns_df=returns_df,
                portfolio_r=r_port,
                benchmark=benchmark_r if not benchmark_r.empty else None,
                ticker_map=ticker_map
            )
            st.dataframe(df_metrics, use_container_width=True, height=460)

            st.markdown('<div class="subsection-header">🧭 Portfolio Weights</div>', unsafe_allow_html=True)
            colw1, colw2 = st.columns([1, 2])
            with colw1:
                st.dataframe(_weights_to_df(w_active, ticker_map), use_container_width=True, height=420)
            with colw2:
                st.plotly_chart(_plot_weights_pie(w_active, ticker_map, title="Weights (Donut)"), use_container_width=True)
    # --- TAB 3: BLACK-LITTERMAN ---
    with tab3:
        # This would be your existing Black-Litterman tab
        st.markdown('<div class="section-header">🎯 Black-Litterman Optimization</div>', unsafe_allow_html=True)

        ticker_map = st.session_state.get("ticker_map", {})
        rf = float(st.session_state.get("last_rf_rate", 0.04))

        if not OPTIMIZATION_AVAILABLE:
            st.markdown('<div class="warning-card">⚠️ Black-Litterman requires PyPortfolioOpt.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="subsection-header">🧠 Black-Litterman: Views → Posterior Returns → Optimized Weights</div>', unsafe_allow_html=True)
            st.caption("Tip: Add a few high-conviction views (annual return expectations) to tilt the portfolio institutionally.")

            returns_df = df_prices.pct_change().dropna()
            returns_df = returns_df.loc[:, ~returns_df.columns.duplicated()]
            if returns_df.empty or df_prices.shape[1] < 2:
                st.markdown('<div class="warning-card">⚠️ Not enough data for BL model.</div>', unsafe_allow_html=True)
            else:
                colA, colB, colC = st.columns(3)
                with colA:
                    market_mode = st.selectbox("Market weights", ["equal", "market_cap"], index=0)
                with colB:
                    delta = st.slider("Risk aversion (delta)", min_value=0.5, max_value=10.0, value=2.5, step=0.1)
                with colC:
                    max_w = st.slider("Max weight per asset", 0.05, 1.0, 0.35, 0.05)

                # Views editor
                tickers = list(df_prices.columns)
                default_views = pd.DataFrame({
                    "Ticker": tickers[:min(5, len(tickers))],
                    "ViewReturn": [0.10] * min(5, len(tickers)),  # annual return view
                })
                st.markdown('<div class="subsection-header">📝 Analyst Views</div>', unsafe_allow_html=True)
                st.caption("ViewReturn is annual (e.g., 0.12 means 12%). Leave empty to run BL with priors only.")
                views_df = st.data_editor(default_views, use_container_width=True, num_rows="dynamic", key="bl_views_editor")

                run_bl = st.button("Run Black-Litterman Optimization", use_container_width=True)
                if run_bl:
                    with st.spinner("Running BL model and optimizing..."):
                        poe = PortfolioOptimizationEngine(risk_free_rate=rf)
                        res = poe.black_litterman(df_prices, views_df=views_df, market_weight_mode=market_mode, delta_override=delta)
                        w = _normalize_weights(res["weights"])
                        st.session_state["bl_weights"] = w
                        st.session_state["active_weights"] = w

                        # Returns comparison
                        prior = res.get("prior_returns", pd.Series(dtype=float))
                        post = res.get("posterior_returns", pd.Series(dtype=float))

                        if hasattr(prior, "index") and hasattr(post, "index"):
                            df_ret = pd.DataFrame({
                                "Prior": prior.reindex(tickers),
                                "Posterior": post.reindex(tickers)
                            }).dropna()
                            df_ret = df_ret.sort_values("Posterior", ascending=False)

                            st.markdown('<div class="subsection-header">📈 Prior vs Posterior Returns</div>', unsafe_allow_html=True)
                            fig_ret = go.Figure()
                            fig_ret.add_trace(go.Bar(x=[ticker_map.get(t, t) for t in df_ret.index], y=df_ret["Prior"], name="Prior"))
                            fig_ret.add_trace(go.Bar(x=[ticker_map.get(t, t) for t in df_ret.index], y=df_ret["Posterior"], name="Posterior"))
                            fig_ret.update_layout(
                                barmode="group",
                                template="plotly_white",
                                height=520,
                                title="Black-Litterman: Prior vs Posterior Expected Returns (Annual)",
                                title_font_color="#1a237e",
                                xaxis_tickangle=-35
                            )
                            st.plotly_chart(fig_ret, use_container_width=True)

                        st.markdown('<div class="subsection-header">🎯 Optimized Weights</div>', unsafe_allow_html=True)
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.dataframe(_weights_to_df(w, ticker_map), use_container_width=True, height=420)
                        with c2:
                            st.plotly_chart(_plot_weights_pie(w, ticker_map, title="BL Weights (Donut)"), use_container_width=True)

                        st.markdown('<div class="subsection-header">📌 Expected Performance (Model)</div>', unsafe_allow_html=True)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Exp. Return", f"{res.get('expected_return', 0.0):.2%}")
                        m2.metric("Exp. Vol", f"{res.get('volatility', 0.0):.2%}")
                        m3.metric("Sharpe", f"{res.get('sharpe', 0.0):.2f}")
    # --- TAB 4: ADVANCED FRONTIER ---
    with tab4:
        # This would be your existing Advanced Frontier tab
        st.markdown('<div class="section-header">📈 Advanced Efficient Frontier</div>', unsafe_allow_html=True)

        ticker_map = st.session_state.get("ticker_map", {})
        rf = float(st.session_state.get("last_rf_rate", 0.04))

        returns_df = df_prices.pct_change().dropna()
        returns_df = returns_df.loc[:, ~returns_df.columns.duplicated()]

        if returns_df.empty or df_prices.shape[1] < 2:
            st.markdown('<div class="warning-card">⚠️ Not enough data for frontier analysis.</div>', unsafe_allow_html=True)
        elif not OPTIMIZATION_AVAILABLE:
            st.markdown('<div class="warning-card">⚠️ Efficient Frontier requires PyPortfolioOpt.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="subsection-header">📐 Efficient Frontier + Random Portfolio Cloud</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                n_sims = st.slider("Random portfolios", 200, 20000, 5000, 200)
            with col2:
                mu_method = st.selectbox("Expected returns", ["mean", "ema", "capm"], index=0)
            with col3:
                cov_method = st.selectbox("Covariance", ["ledoit_wolf", "sample", "exp_cov"], index=0)

            poe = PortfolioOptimizationEngine(risk_free_rate=rf)
            mu, S = poe._mu_sigma(df_prices, mu_method=mu_method, cov_method=cov_method)

            # Efficient Frontier points
            ef = EfficientFrontier(mu, S)
            rets = np.linspace(float(mu.min()), float(mu.max()), 25)
            frontier_x = []
            frontier_y = []
            for tr in rets:
                try:
                    ef_tmp = EfficientFrontier(mu, S)
                    ef_tmp.efficient_return(target_return=tr)
                    pr, pv, _ = ef_tmp.portfolio_performance(risk_free_rate=rf, verbose=False)
                    frontier_x.append(pv)
                    frontier_y.append(pr)
                except Exception:
                    continue

            # Random portfolios
            rng = np.random.default_rng(42)
            W = rng.random((n_sims, len(mu)))
            W = W / W.sum(axis=1, keepdims=True)
            port_rets = W @ mu.values
            port_vols = np.sqrt(np.einsum('ij,jk,ik->i', W, S.values, W))

            sharpe = (port_rets - rf) / np.where(port_vols == 0, np.nan, port_vols)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=port_vols, y=port_rets,
                mode="markers",
                marker=dict(size=5, color=sharpe, colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")),
                name="Random Portfolios",
                hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>"
            ))
            if frontier_x and frontier_y:
                fig.add_trace(go.Scatter(
                    x=frontier_x, y=frontier_y,
                    mode="lines",
                    name="Efficient Frontier",
                    line=dict(width=4)
                ))

            # highlight last optimized weights if exists
            w_last = st.session_state.get("mv_weights", None)
            if isinstance(w_last, dict) and len(w_last) > 0:
                wv = pd.Series(_normalize_weights(w_last)).reindex(df_prices.columns).fillna(0.0).values
                p_ret = float(wv @ mu.values)
                p_vol = float(np.sqrt(wv.T @ S.values @ wv))
                fig.add_trace(go.Scatter(
                    x=[p_vol], y=[p_ret],
                    mode="markers",
                    marker=dict(size=14, symbol="star"),
                    name="Last EF Portfolio",
                    hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
                ))

            fig.update_layout(
                template="plotly_white",
                height=720,
                title="Efficient Frontier (Return vs Volatility)",
                title_font_color="#1a237e",
                xaxis_title="Volatility (σ)",
                yaxis_title="Expected Return (μ)",
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="subsection-header">🧊 3D Frontier View (Return–Vol–Sharpe)</div>', unsafe_allow_html=True)
            fig3d = go.Figure(data=[go.Scatter3d(
                x=port_vols,
                y=port_rets,
                z=sharpe,
                mode="markers",
                marker=dict(size=3, color=sharpe, colorscale="Viridis", showscale=True),
                hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>Sharpe: %{z:.2f}<extra></extra>"
            )])
            fig3d.update_layout(
                template="plotly_white",
                height=680,
                title="3D Portfolio Cloud",
                title_font_color="#1a237e",
                scene=dict(
                    xaxis_title="Vol",
                    yaxis_title="Return",
                    zaxis_title="Sharpe"
                )
            )
            st.plotly_chart(fig3d, use_container_width=True)
    # --- NEW TAB 5: BOLLINGER BANDS ---
    with tab5:
        create_bollinger_bands_tab()
    
    # --- NEW TAB 6: OHLC & TRACKING ERROR ---
    with tab6:
        create_ohlc_tracking_error_tab()
    
    # --- NEW TAB 7: ENHANCED STRESS TESTING ---
    with tab7:
        create_enhanced_stress_test_tab()
    
    # --- TAB 8: PORTFOLIO OPTIMIZATION ---
    with tab8:
        if not OPTIMIZATION_AVAILABLE:
            st.markdown('<div class="warning-card">⚠️ Portfolio optimization requires PyPortfolioOpt. Add it to requirements.txt: PyPortfolioOpt</div>', unsafe_allow_html=True)
        else:
            # This would be your existing Portfolio Optimization tab
            st.markdown('<div class="section-header">🎯 Portfolio Construction & Optimization</div>', unsafe_allow_html=True)

        ticker_map = st.session_state.get("ticker_map", {})
        rf = float(st.session_state.get("last_rf_rate", 0.04))
        investment_amount = float(st.session_state.get("last_amount", 1000000.0))

        if not OPTIMIZATION_AVAILABLE:
            st.markdown(
                '<div class="warning-card">⚠️ Portfolio Optimization requires PyPortfolioOpt (+ cvxpy, ecos). '
                'Please add them to requirements.txt.</div>',
                unsafe_allow_html=True
            )
        else:
            returns_df = df_prices.pct_change().dropna()
            returns_df = returns_df.loc[:, ~returns_df.columns.duplicated()]
            if returns_df.empty or df_prices.shape[1] < 2:
                st.markdown('<div class="warning-card">⚠️ Not enough data for optimization.</div>', unsafe_allow_html=True)
            else:
                poe = PortfolioOptimizationEngine(risk_free_rate=rf)

                st.markdown('<div class="subsection-header">🧩 Institutional Optimizers (PyPortfolioOpt)</div>', unsafe_allow_html=True)
                opt_tab1, opt_tab2, opt_tab3, opt_tab4, opt_tab5 = st.tabs([
                    "⚙️ Mean-Variance (EF)",
                    "🧱 Tail & Downside",
                    "🌿 HRP / CLA",
                    "🧪 Strategy Comparison",
                    "💵 Discrete Allocation"
                ])

                # -----------------------------
                # TAB 1: EF Mean-Variance
                # -----------------------------
                with opt_tab1:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        mu_method = st.selectbox("Expected returns", ["mean", "ema", "capm"], index=0, key="mv_mu_method")
                    with c2:
                        cov_method = st.selectbox("Covariance", ["ledoit_wolf", "sample", "exp_cov"], index=0, key="mv_cov_method")
                    with c3:
                        objective = st.selectbox(
                            "Objective",
                            ["max_sharpe", "min_volatility", "max_quadratic_utility", "efficient_return", "efficient_risk"],
                            index=0,
                            key="mv_objective"
                        )
                    with c4:
                        max_w = st.slider("Max weight", 0.05, 1.0, 0.35, 0.05, key="mv_maxw")

                    colT1, colT2 = st.columns(2)
                    with colT1:
                        target_return = st.slider("Target Return (ann.)", 0.00, 0.60, 0.15, 0.01, key="mv_target_return")
                    with colT2:
                        target_risk = st.slider("Target Volatility (ann.)", 0.01, 0.80, 0.20, 0.01, key="mv_target_risk")

                    run = st.button("Run Mean-Variance Optimization", use_container_width=True, key="mv_run")
                    if run:
                        with st.spinner("Optimizing Mean-Variance portfolio..."):
                            res = poe.mean_variance(
                                df_prices,
                                mu_method=mu_method,
                                cov_method=cov_method,
                                objective=objective,
                                max_weight=max_w,
                                target_return=target_return if objective == "efficient_return" else None,
                                target_risk=target_risk if objective == "efficient_risk" else None
                            )
                            w = _normalize_weights(res["weights"])
                            st.session_state["mv_weights"] = w
                            st.session_state["active_weights"] = w

                            m1, m2, m3 = st.columns(3)
                            m1.metric("Exp. Return", f"{res.get('expected_return', 0.0):.2%}")
                            m2.metric("Exp. Vol", f"{res.get('volatility', 0.0):.2%}")
                            m3.metric("Sharpe", f"{res.get('sharpe', 0.0):.2f}")

                            dw1, dw2 = st.columns([1, 2])
                            with dw1:
                                st.dataframe(_weights_to_df(w, ticker_map), use_container_width=True, height=450)
                            with dw2:
                                st.plotly_chart(_plot_weights_pie(w, ticker_map, title="EF Weights (Donut)"), use_container_width=True)

                # -----------------------------
                # TAB 2: Tail & Downside (SemiVar / CVaR / CDaR)
                # -----------------------------
                with opt_tab2:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        beta_tail = st.selectbox("Tail confidence (beta)", [0.90, 0.95, 0.975, 0.99], index=1, key="tail_beta")
                    with c2:
                        max_w2 = st.slider("Max weight", 0.05, 1.0, 0.35, 0.05, key="tail_maxw")
                    with c3:
                        semivar_benchmark = st.slider("Semivariance benchmark", -0.02, 0.02, 0.00, 0.001, key="semivar_bench")

                    run_sv, run_cvar, run_cdar = st.columns(3)
                    with run_sv:
                        do_sv = st.button("Min Semivariance", use_container_width=True, key="run_semivar")
                    with run_cvar:
                        do_cvar = st.button("Min CVaR", use_container_width=True, key="run_cvar")
                    with run_cdar:
                        do_cdar = st.button("Min CDaR", use_container_width=True, key="run_cdar")

                    if do_sv:
                        with st.spinner("Optimizing Semivariance portfolio..."):
                            res = poe.semivariance(df_prices, benchmark=semivar_benchmark, max_weight=max_w2)
                            w = _normalize_weights(res["weights"])
                            st.session_state["sv_weights"] = w
                            st.session_state["active_weights"] = w
                            st.success("Semivariance optimization completed.")
                            st.plotly_chart(_plot_weights_pie(w, ticker_map, title="Semivariance Weights (Donut)"), use_container_width=True)
                            st.dataframe(_weights_to_df(w, ticker_map), use_container_width=True)

                    if do_cvar:
                        with st.spinner("Optimizing CVaR portfolio..."):
                            res = poe.cvar(df_prices, beta=float(beta_tail), max_weight=max_w2)
                            w = _normalize_weights(res["weights"])
                            st.session_state["cvar_weights"] = w
                            st.session_state["active_weights"] = w
                            st.success("CVaR optimization completed.")
                            st.plotly_chart(_plot_weights_pie(w, ticker_map, title="Min CVaR Weights (Donut)"), use_container_width=True)
                            st.dataframe(_weights_to_df(w, ticker_map), use_container_width=True)

                    if do_cdar:
                        with st.spinner("Optimizing CDaR portfolio..."):
                            res = poe.cdar(df_prices, beta=float(beta_tail), max_weight=max_w2)
                            w = _normalize_weights(res["weights"])
                            st.session_state["cdar_weights"] = w
                            st.session_state["active_weights"] = w
                            st.success("CDaR optimization completed.")
                            st.plotly_chart(_plot_weights_pie(w, ticker_map, title="Min CDaR Weights (Donut)"), use_container_width=True)
                            st.dataframe(_weights_to_df(w, ticker_map), use_container_width=True)

                # -----------------------------
                # TAB 3: HRP / CLA
                # -----------------------------
                with opt_tab3:
                    c1, c2 = st.columns(2)
                    with c1:
                        do_hrp = st.button("Run HRP (Hierarchical Risk Parity)", use_container_width=True, key="run_hrp")
                    with c2:
                        do_cla = st.button("Run CLA (Min Vol)", use_container_width=True, key="run_cla")

                    if do_hrp:
                        with st.spinner("Running HRP optimization..."):
                            res = poe.hrp(returns_df)
                            w = _normalize_weights(res["weights"])
                            st.session_state["hrp_weights"] = w
                            st.session_state["active_weights"] = w
                            st.success("HRP optimization completed.")
                            cA, cB = st.columns([1, 2])
                            with cA:
                                st.dataframe(_weights_to_df(w, ticker_map), use_container_width=True, height=420)
                            with cB:
                                st.plotly_chart(_plot_weights_pie(w, ticker_map, title="HRP Weights (Donut)"), use_container_width=True)

                    if do_cla:
                        with st.spinner("Running CLA optimization..."):
                            res = poe.cla(df_prices, mu_method="mean", cov_method="ledoit_wolf", max_weight=0.35)
                            w = _normalize_weights(res["weights"])
                            st.session_state["cla_weights"] = w
                            st.session_state["active_weights"] = w
                            st.success("CLA optimization completed.")
                            cA, cB = st.columns([1, 2])
                            with cA:
                                st.dataframe(_weights_to_df(w, ticker_map), use_container_width=True, height=420)
                            with cB:
                                st.plotly_chart(_plot_weights_pie(w, ticker_map, title="CLA Weights (Donut)"), use_container_width=True)

                # -----------------------------
                # TAB 4: Strategy comparison (backtest)
                # -----------------------------
                with opt_tab4:
                    st.markdown('<div class="subsection-header">📊 Strategy Comparison (Backtest)</div>', unsafe_allow_html=True)
                    strategies = {
                        "Equal Weight": st.session_state.get("active_weights", None) or {c: 1.0/len(returns_df.columns) for c in returns_df.columns},
                        "EF (Max Sharpe)": st.session_state.get("mv_weights", None),
                        "CLA": st.session_state.get("cla_weights", None),
                        "HRP": st.session_state.get("hrp_weights", None),
                        "Min CVaR": st.session_state.get("cvar_weights", None),
                        "Min CDaR": st.session_state.get("cdar_weights", None),
                        "Black-Litterman": st.session_state.get("bl_weights", None),
                    }
                    strategies = {k: v for k, v in strategies.items() if isinstance(v, dict) and len(v) > 0}

                    if len(strategies) < 1:
                        st.markdown('<div class="warning-card">⚠️ Run at least one optimizer to compare strategies.</div>', unsafe_allow_html=True)
                    else:
                        fig = go.Figure()
                        perf_engine = PerformanceAnalyticsEngine(risk_free_rate=rf)

                        rows = []
                        for name, w in strategies.items():
                            r_p = _portfolio_returns(returns_df, w)
                            eq = (1 + r_p).cumprod()
                            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name))
                            met = perf_engine.compute_metrics(r_p, benchmark=None)
                            rows.append({"Strategy": name, **met})

                        fig.update_layout(
                            template="plotly_white",
                            height=560,
                            title="Cumulative Performance — Strategy Comparison",
                            title_font_color="#1a237e",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        df_strat = pd.DataFrame(rows)
                        st.dataframe(df_strat, use_container_width=True, height=420)

                # -----------------------------
                # TAB 5: Discrete allocation
                # -----------------------------
                with opt_tab5:
                    st.markdown('<div class="subsection-header">🪙 Discrete Allocation</div>', unsafe_allow_html=True)
                    active = st.session_state.get("active_weights", None)
                    if not isinstance(active, dict) or len(active) == 0:
                        st.markdown('<div class="warning-card">⚠️ No active weights found. Run an optimizer first.</div>', unsafe_allow_html=True)
                    else:
                        amt = st.number_input("Investment amount", min_value=1000.0, value=float(investment_amount), step=1000.0)
                        try:
                            latest_prices = get_latest_prices(df_prices)
                            da = DiscreteAllocation(active, latest_prices, total_portfolio_value=float(amt))
                            allocation, leftover = da.lp_portfolio()
                            df_alloc = pd.DataFrame({"Shares": allocation}).sort_values("Shares", ascending=False)
                            st.markdown('<div class="info-card">Leftover cash: <b>${:,.2f}</b></div>'.format(leftover), unsafe_allow_html=True)
                            st.dataframe(df_alloc, use_container_width=True)
                        except Exception as e:
                            st.markdown(f'<div class="warning-card">⚠️ Discrete allocation error: {e}</div>', unsafe_allow_html=True)
    # --- TAB 9: ADVANCED VAR/ES ---
    with tab9:
        # This would be your existing VaR/ES tab
        st.markdown('<div class="section-header">🎲 Advanced VaR/ES Analysis</div>', unsafe_allow_html=True)

        ticker_map = st.session_state.get("ticker_map", {})
        rf = float(st.session_state.get("last_rf_rate", 0.04))
        benchmark_r = st.session_state.get("benchmark_returns", pd.Series(dtype=float))

        returns_df = df_prices.pct_change().dropna()
        returns_df = returns_df.loc[:, ~returns_df.columns.duplicated()]

        if returns_df.empty:
            st.markdown('<div class="warning-card">⚠️ Not enough return data for VaR/ES.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="subsection-header">🧨 VaR / CVaR (ES) — Historical, Parametric, Monte Carlo</div>', unsafe_allow_html=True)

            # Weight source
            weight_sources = {
                "Equal Weight": {c: 1.0/len(returns_df.columns) for c in returns_df.columns},
                "Active (Last Selected)": st.session_state.get("active_weights", {}),
                "EF (Mean-Variance)": st.session_state.get("mv_weights", {}),
                "HRP": st.session_state.get("hrp_weights", {}),
                "Min CVaR": st.session_state.get("cvar_weights", {}),
                "Min CDaR": st.session_state.get("cdar_weights", {}),
                "Black-Litterman": st.session_state.get("bl_weights", {}),
            }
            weight_sources = {k: v for k, v in weight_sources.items() if isinstance(v, dict) and len(v) > 0}
            src = st.selectbox("Weights source", list(weight_sources.keys()), index=0)

            w = _normalize_weights(weight_sources[src])
            st.session_state["active_weights"] = w

            r_port = _portfolio_returns(returns_df, w)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cls = st.multiselect("Confidence levels", [0.90, 0.95, 0.975, 0.99], default=[0.95, 0.99])
            with c2:
                mc_sims = st.slider("MC simulations", 2000, 100000, 20000, 2000)
            with c3:
                horizon = st.slider("Horizon (days)", 1, 20, 1, 1)
            with c4:
                dist = st.selectbox("MC distribution", ["normal", "t"], index=0)

            ra = RiskAnalyticsEngine()
            if st.button("Compute VaR / ES", use_container_width=True):
                with st.spinner("Computing VaR / ES across methods..."):
                    df_risk = ra.build_risk_table(r_port, returns_df, w, cls=cls, mc_sims=int(mc_sims))
                    st.session_state["risk_table"] = df_risk

            df_risk = st.session_state.get("risk_table", None)
            if isinstance(df_risk, pd.DataFrame) and not df_risk.empty:
                st.markdown('<div class="subsection-header">📋 Risk Table</div>', unsafe_allow_html=True)
                st.dataframe(df_risk, use_container_width=True, height=420)

                st.markdown('<div class="subsection-header">📊 Comparative Charts</div>', unsafe_allow_html=True)
                # VaR comparison
                fig_var = px.bar(df_risk, x="Method", y="VaR %", color="Confidence", barmode="group",
                                 title="VaR (%) by Method & Confidence")
                fig_var.update_layout(template="plotly_white", height=520, title_font_color="#1a237e", xaxis_tickangle=-25)
                st.plotly_chart(fig_var, use_container_width=True)

                # ES comparison
                fig_es = px.bar(df_risk, x="Method", y="ES %", color="Confidence", barmode="group",
                                title="ES / CVaR (%) by Method & Confidence")
                fig_es.update_layout(template="plotly_white", height=520, title_font_color="#1a237e", xaxis_tickangle=-25)
                st.plotly_chart(fig_es, use_container_width=True)

                st.markdown('<div class="subsection-header">📈 Return Distribution (Portfolio)</div>', unsafe_allow_html=True)
                # Histogram with VaR lines at the highest confidence selected
                cl_max = max(cls) if cls else 0.95
                var_h, es_h = ra.historical_var_es(r_port, cl=cl_max)
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=r_port.values * 100, nbinsx=60, name="Daily Returns (%)"))
                fig_hist.add_vline(x=-var_h * 100, line_dash="dash", annotation_text=f"VaR {int(cl_max*100)}%")
                fig_hist.add_vline(x=-es_h * 100, line_dash="dot", annotation_text=f"ES {int(cl_max*100)}%")
                fig_hist.update_layout(template="plotly_white", height=520, title="Portfolio Return Distribution",
                                       title_font_color="#1a237e", xaxis_title="Daily Return (%)", yaxis_title="Count")
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown('<div class="subsection-header">⏱️ Rolling Historical VaR</div>', unsafe_allow_html=True)
                win = st.slider("Rolling window", 20, 252, 60, 5)
                roll = r_port.rolling(win)
                roll_var = roll.apply(lambda x: np.quantile(-x, cl_max), raw=False) * 100
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(x=roll_var.index, y=roll_var.values, mode="lines", name=f"VaR {int(cl_max*100)}%"))
                fig_roll.update_layout(template="plotly_white", height=420, title=f"Rolling Historical VaR ({win}D)",
                                       title_font_color="#1a237e", yaxis_title="VaR (%)")
                st.plotly_chart(fig_roll, use_container_width=True)

            st.markdown('<div class="subsection-header">🧭 Current Weights (Donut)</div>', unsafe_allow_html=True)
            st.plotly_chart(_plot_weights_pie(w, ticker_map, title=f"Weights — {src}"), use_container_width=True)
    # --- TAB 10: RISK ANALYTICS ---
    with tab10:
        # This would be your existing Risk Analytics tab
        st.markdown('<div class="section-header">🛡️ Risk Analytics</div>', unsafe_allow_html=True)

        ticker_map = st.session_state.get("ticker_map", {})
        rf = float(st.session_state.get("last_rf_rate", 0.04))
        benchmark_r = st.session_state.get("benchmark_returns", pd.Series(dtype=float))

        returns_df = df_prices.pct_change().dropna()
        returns_df = returns_df.loc[:, ~returns_df.columns.duplicated()]
        if returns_df.empty:
            st.markdown('<div class="warning-card">⚠️ Not enough data for risk analytics.</div>', unsafe_allow_html=True)
        else:
            weights = st.session_state.get("active_weights", {c: 1.0/len(returns_df.columns) for c in returns_df.columns})
            weights = _normalize_weights(weights)
            st.session_state["active_weights"] = weights

            r_port = _portfolio_returns(returns_df, weights)

            sub1, sub2 = st.tabs(["🧱 Risk Dashboard", "⚖️ Relative Risk (vs Benchmark)"])

            # -----------------------
            # Subtab 1: Risk Dashboard
            # -----------------------
            with sub1:
                st.markdown('<div class="subsection-header">📌 Portfolio Risk Snapshot</div>', unsafe_allow_html=True)
                pe = PerformanceAnalyticsEngine(risk_free_rate=rf)
                m = pe.compute_metrics(r_port, benchmark=benchmark_r if not benchmark_r.empty else None)

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Ann. Vol", f"{m.get('Ann. Vol', 0.0):.2%}")
                k2.metric("Sharpe", f"{m.get('Sharpe', 0.0):.2f}")
                k3.metric("Skew", f"{m.get('Skew', 0.0):.2f}")
                k4.metric("Kurtosis", f"{m.get('Kurtosis', 0.0):.2f}")
                k5.metric("Max DD", f"{m.get('Max Drawdown', 0.0):.2%}")

                st.plotly_chart(pe.performance_charts(r_port, benchmark=benchmark_r if not benchmark_r.empty else None,
                                                     title="Risk Dashboard — Portfolio & Benchmark"), use_container_width=True)

                st.markdown('<div class="subsection-header">🧮 Volatility Risk Contribution</div>', unsafe_allow_html=True)
                cov = returns_df.cov() * 252
                w_vec = pd.Series(weights).reindex(returns_df.columns).fillna(0.0)
                port_var = float(w_vec.T @ cov.values @ w_vec.values)
                if port_var <= 0:
                    st.markdown('<div class="warning-card">⚠️ Could not compute risk contribution (variance is zero).</div>', unsafe_allow_html=True)
                else:
                    mrc = cov.values @ w_vec.values
                    rc = (w_vec.values * mrc) / port_var
                    df_rc = pd.DataFrame({
                        "Ticker": returns_df.columns,
                        "Name": [ticker_map.get(t, t) for t in returns_df.columns],
                        "Weight %": w_vec.values * 100,
                        "Risk Contribution %": rc * 100
                    }).sort_values("Risk Contribution %", ascending=False)

                    cA, cB = st.columns([1, 2])
                    with cA:
                        st.dataframe(df_rc, use_container_width=True, height=420)
                    with cB:
                        fig_rc = go.Figure(data=[go.Pie(
                            labels=df_rc["Name"],
                            values=df_rc["Risk Contribution %"],
                            hole=0.55,
                            textinfo="label+percent"
                        )])
                        fig_rc.update_layout(template="plotly_white", height=420, title="Risk Contribution (Vol)",
                                            title_font_color="#1a237e",
                                            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
                        st.plotly_chart(fig_rc, use_container_width=True)

                st.markdown('<div class="subsection-header">🧨 Asset-Level Tail Risk (Historical)</div>', unsafe_allow_html=True)
                ra = RiskAnalyticsEngine()
                cl = st.selectbox("Confidence", [0.90, 0.95, 0.975, 0.99], index=1, key="risk_dash_cl")
                rows = []
                for c in returns_df.columns:
                    v, e = ra.historical_var_es(returns_df[c], cl=cl)
                    rows.append({"Asset": ticker_map.get(c, c), "VaR %": v*100, "ES %": e*100})
                df_asset_tail = pd.DataFrame(rows).sort_values("ES %", ascending=False)
                st.dataframe(df_asset_tail, use_container_width=True, height=420)

                st.markdown('<div class="subsection-header">🧭 Portfolio Weights</div>', unsafe_allow_html=True)
                st.plotly_chart(_plot_weights_pie(weights, ticker_map, title="Active Weights (Donut)"), use_container_width=True)

            # -----------------------
            # Subtab 2: Relative Risk
            # -----------------------
            with sub2:
                st.markdown('<div class="subsection-header">⚖️ Relative VaR / Relative ES (Portfolio − Benchmark)</div>', unsafe_allow_html=True)

                if benchmark_r is None or benchmark_r.empty:
                    st.markdown('<div class="warning-card">⚠️ Benchmark is missing. Choose a benchmark in the sidebar or ensure benchmark data is available.</div>', unsafe_allow_html=True)
                else:
                    ra = RiskAnalyticsEngine()
                    cls = st.multiselect("Confidence levels", [0.90, 0.95, 0.975, 0.99], default=[0.95, 0.99], key="rel_cls")
                    mc_sims = st.slider("MC simulations", 2000, 100000, 20000, 2000, key="rel_mc")
                    methods = ["Historical", "Parametric (Normal)", "Parametric (t)", "Monte Carlo (Normal)", "Monte Carlo (t)"]

                    # Build both tables
                    df_p = ra.build_risk_table(r_port, returns_df, weights, cls=cls, mc_sims=int(mc_sims))
                    df_b = ra.build_risk_table(benchmark_r, pd.DataFrame({"B": benchmark_r}).dropna(), {"B": 1.0}, cls=cls, mc_sims=int(mc_sims))

                    # Align to same schema and compute diffs (Portfolio - Benchmark)
                    df_merge = df_p.merge(df_b, on=["Confidence", "Method"], suffixes=("_P", "_B"))
                    df_merge["Rel VaR %"] = df_merge["VaR %_P"] - df_merge["VaR %_B"]
                    df_merge["Rel ES %"] = df_merge["ES %_P"] - df_merge["ES %_B"]

                    st.markdown('<div class="subsection-header">📋 Relative Risk Table</div>', unsafe_allow_html=True)
                    show_cols = ["Confidence","Method","VaR %_P","VaR %_B","Rel VaR %","ES %_P","ES %_B","Rel ES %"]
                    st.dataframe(df_merge[show_cols], use_container_width=True, height=460)

                    st.markdown('<div class="subsection-header">📊 Comparative Charts</div>', unsafe_allow_html=True)
                    # Absolute VaR chart
                    fig_abs_var = px.bar(df_merge, x="Method", y="VaR %_P", color="Confidence", barmode="group",
                                         title="Portfolio VaR (%) by Method")
                    fig_abs_var.update_layout(template="plotly_white", height=520, title_font_color="#1a237e", xaxis_tickangle=-25)
                    st.plotly_chart(fig_abs_var, use_container_width=True)

                    fig_abs_var_b = px.bar(df_merge, x="Method", y="VaR %_B", color="Confidence", barmode="group",
                                           title="Benchmark VaR (%) by Method")
                    fig_abs_var_b.update_layout(template="plotly_white", height=520, title_font_color="#1a237e", xaxis_tickangle=-25)
                    st.plotly_chart(fig_abs_var_b, use_container_width=True)

                    # Relative VaR chart
                    fig_rel_var = px.bar(df_merge, x="Method", y="Rel VaR %", color="Confidence", barmode="group",
                                         title="Relative VaR (%) = Portfolio − Benchmark")
                    fig_rel_var.update_layout(template="plotly_white", height=520, title_font_color="#1a237e", xaxis_tickangle=-25)
                    st.plotly_chart(fig_rel_var, use_container_width=True)

                    # Relative ES chart
                    fig_rel_es = px.bar(df_merge, x="Method", y="Rel ES %", color="Confidence", barmode="group",
                                        title="Relative ES (%) = Portfolio − Benchmark")
                    fig_rel_es.update_layout(template="plotly_white", height=520, title_font_color="#1a237e", xaxis_tickangle=-25)
                    st.plotly_chart(fig_rel_es, use_container_width=True)

                    st.markdown('<div class="subsection-header">⏱️ Rolling Relative Historical VaR</div>', unsafe_allow_html=True)
                    cl_max = max(cls) if cls else 0.95
                    win = st.slider("Rolling window", 20, 252, 60, 5, key="rel_roll_win")
                    roll_p = r_port.rolling(win).apply(lambda x: np.quantile(-x, cl_max), raw=False) * 100
                    roll_b = benchmark_r.rolling(win).apply(lambda x: np.quantile(-x, cl_max), raw=False) * 100
                    rel = (roll_p - roll_b)
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(x=rel.index, y=rel.values, mode="lines", name="Rel VaR (%)"))
                    fig_roll.update_layout(template="plotly_white", height=420,
                                           title=f"Rolling Relative VaR ({win}D, {int(cl_max*100)}%)",
                                           title_font_color="#1a237e", yaxis_title="Rel VaR (%)")
                    st.plotly_chart(fig_roll, use_container_width=True)
    # --- TAB 11: CORRELATION ANALYSIS ---
    with tab11:
        # This would be your existing Correlation Analysis tab
        st.markdown('<div class="section-header">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
        
        returns = df_prices.pct_change().dropna()
        returns = returns.loc[:, ~returns.columns.duplicated()]
        
        if returns.empty or returns.shape[1] < 2:
            st.markdown('<div class="warning-card">⚠️ Please select at least two assets with sufficient overlapping data.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="subsection-header">📊 Correlation Matrix</div>', unsafe_allow_html=True)
            corr_matrix = returns.corr()
            
            ticker_map = dm.get_ticker_name_map()
            labels = [ticker_map.get(t, t) for t in corr_matrix.columns]
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=labels,
                y=labels,
                zmin=-1,
                zmax=1,
                colorscale="RdBu_r",
                hoverongaps=False,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.3f}<extra></extra>"
            ))
            fig_corr.update_layout(
                template="plotly_white",
                height=700,
                title="Asset Correlation Matrix",
                title_font_color="#1a237e",
                xaxis_title="Assets",
                yaxis_title="Assets",
                xaxis=dict(tickangle=-45),
                font_color="#424242"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# 10) MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
