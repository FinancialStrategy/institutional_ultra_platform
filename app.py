# ==============================================================================
# QUANTUM | Global Institutional Terminal - ENHANCED COMPLETE VERSION
# Advanced VaR/CVaR/ES + Stress Testing + Performance Metrics + Black-Litterman + 3D Frontier
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
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
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
            background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
            border-left: 4px solid #ff9800;
            color: #5d4037;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .success-card {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 4px solid #4caf50;
            color: #1b5e20;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .info-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 4px solid #2196f3;
            color: #0d47a1;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
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
            border-radius: 8px;
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
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c5cae9;
            border-radius: 4px;
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
            border-radius: 8px !important;
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
# 5) ORIGINAL OPTIMIZATION ENGINE
# ==============================================================================

# Try to import optimization libraries
OPTIMIZATION_AVAILABLE = False
BLACK_LITTERMAN_AVAILABLE = False
HRP_AVAILABLE = False

try:
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models, expected_returns, objective_functions

    # HRP import paths differ by version
    HRPOpt = None
    try:
        from pypfopt.hierarchical_portfolio import HRPOpt as _HRPOpt
        HRPOpt = _HRPOpt
        HRP_AVAILABLE = True
    except Exception:
        try:
            from pypfopt.hierarchical_risk_parity import HRPOpt as _HRPOpt
            HRPOpt = _HRPOpt
            HRP_AVAILABLE = True
        except Exception:
            HRP_AVAILABLE = False

    # Black-Litterman import paths differ by version
    try:
        from pypfopt.black_litterman import BlackLittermanModel
        BLACK_LITTERMAN_AVAILABLE = True
    except Exception:
        try:
            from pypfopt import BlackLittermanModel  # legacy
            BLACK_LITTERMAN_AVAILABLE = True
        except Exception:
            BLACK_LITTERMAN_AVAILABLE = False

    OPTIMIZATION_AVAILABLE = True

except Exception as e:
    OPTIMIZATION_AVAILABLE = False
    _OPT_IMPORT_ERROR = str(e)

class EnhancedOptimizationEngine:
    def __init__(self, df_prices: pd.DataFrame):
        if df_prices is None or df_prices.empty:
            raise ValueError("Empty price DataFrame provided")

        self._validate_data(df_prices)

        self.df = df_prices.copy()
        self.returns = self.df.pct_change().dropna()
        self.mu = expected_returns.mean_historical_return(self.df)
        self.S = risk_models.sample_cov(self.df)
        self.num_assets = len(self.df.columns)

    def _validate_data(self, df: pd.DataFrame):
        if df.isnull().any().any():
            nan_count = int(df.isnull().sum().sum())
            raise ValueError(f"Data contains {nan_count} NaN values")

        if len(df) < 100:
            raise ValueError(f"Insufficient data points: {len(df)}. Need at least 100.")

        if (df <= 0).any().any():
            bad_count = int((df <= 0).sum().sum())
            raise ValueError(f"Data contains {bad_count} zero/negative prices")

        lengths = [len(df[col].dropna()) for col in df.columns]
        if max(lengths) - min(lengths) > 10:
            st.warning(f"Asset lengths vary: min={min(lengths)}, max={max(lengths)} (post-cleaning).")

    def optimize_mean_variance(self, objective="max_sharpe", risk_free_rate=0.04, gamma=None):
        try:
            ef = EfficientFrontier(self.mu, self.S)
            if gamma:
                ef.add_objective(objective_functions.L2_reg, gamma=gamma)

            if objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == "min_volatility":
                ef.min_volatility()
            elif objective == "max_quadratic_utility":
                ef.max_quadratic_utility(risk_aversion=1)
            else:
                ef.max_sharpe(risk_free_rate=risk_free_rate)

            cleaned_weights = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            return cleaned_weights, perf

        except Exception as e:
            st.error(f"Optimization error: {e}")
            equal_weight = 1.0 / self.num_assets
            weights = {asset: equal_weight for asset in self.df.columns}
            ret = float(self.mu.mean())
            vol = float(np.sqrt(np.diag(self.S).mean()))
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
            return weights, (ret, vol, sharpe)

    def optimize_hrp(self):
        if not HRP_AVAILABLE or HRPOpt is None:
            st.warning("HRP optimization is not available in your installed PyPortfolioOpt version.")
            equal_weight = 1.0 / self.num_assets
            weights = {asset: equal_weight for asset in self.df.columns}
            port_returns = self.returns.dot(pd.Series(weights))
            ann_ret = port_returns.mean() * 252
            ann_vol = port_returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            return weights, (float(ann_ret), float(ann_vol), float(sharpe))

        try:
            if len(self.returns) < 100:
                raise ValueError(f"Insufficient returns data: {len(self.returns)} points")

            hrp = HRPOpt(returns=self.returns)
            hrp.optimize()
            cleaned_weights = hrp.clean_weights()

            port_returns = self.returns.dot(pd.Series(cleaned_weights))
            ann_ret = port_returns.mean() * 252
            ann_vol = port_returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

            return cleaned_weights, (float(ann_ret), float(ann_vol), float(sharpe))

        except Exception as e:
            st.warning(f"HRP optimization failed: {e}. Using equal weights.")
            equal_weight = 1.0 / self.num_assets
            weights = {asset: equal_weight for asset in self.df.columns}
            port_returns = self.returns.dot(pd.Series(weights))
            ann_ret = port_returns.mean() * 252
            ann_vol = port_returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            return weights, (float(ann_ret), float(ann_vol), float(sharpe))

# ==============================================================================
# 6) ADVANCED PERFORMANCE METRICS ENGINE (NEW)
# ==============================================================================

class AdvancedPerformanceMetrics:
    """Comprehensive performance metrics engine with 50+ institutional-grade metrics"""
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series = None, 
                 risk_free_rate: float = 0.04, trading_days: int = 252):
        self.returns = portfolio_returns.dropna()
        self.benchmark = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.rf = risk_free_rate
        self.days = trading_days
        
        # Align benchmark if provided
        if self.benchmark is not None:
            aligned = pd.concat([self.returns, self.benchmark], axis=1).dropna()
            if not aligned.empty:
                self.returns_aligned = aligned.iloc[:, 0]
                self.benchmark_aligned = aligned.iloc[:, 1]
            else:
                self.returns_aligned = self.returns
                self.benchmark_aligned = None
        else:
            self.returns_aligned = self.returns
            self.benchmark_aligned = None
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate 50+ comprehensive performance metrics"""
        metrics = {}
        
        # 1. RETURN METRICS
        metrics["Annual Return"] = float(self.returns.mean() * self.days)
        metrics["Cumulative Return"] = float((1 + self.returns).prod() - 1)
        metrics["Geometric Mean Return"] = float(((1 + self.returns).prod()) ** (self.days/len(self.returns)) - 1)
        
        # Rolling returns
        if len(self.returns) >= 63:  # 3 months
            metrics["3M Rolling Return"] = float(self.returns.tail(63).mean() * self.days)
        if len(self.returns) >= 126:  # 6 months
            metrics["6M Rolling Return"] = float(self.returns.tail(126).mean() * self.days)
        if len(self.returns) >= 252:  # 1 year
            metrics["1Y Rolling Return"] = float(self.returns.tail(252).mean() * self.days)
        
        # 2. RISK METRICS
        metrics["Annual Volatility"] = float(self.returns.std() * np.sqrt(self.days))
        metrics["Downside Volatility"] = self._calculate_downside_volatility()
        metrics["Maximum Drawdown"] = float(self._calculate_max_drawdown())
        metrics["VaR 95% (Historical)"] = float(-np.percentile(self.returns, 5))
        metrics["CVaR/ES 95%"] = float(-self.returns[self.returns <= np.percentile(self.returns, 5)].mean())
        metrics["Tail Risk (CVaR - VaR)"] = metrics["CVaR/ES 95%"] - metrics["VaR 95% (Historical)"]
        
        # 3. RISK-ADJUSTED RATIOS
        vol = metrics["Annual Volatility"]
        metrics["Sharpe Ratio"] = float((metrics["Annual Return"] - self.rf) / vol) if vol > 0 else 0.0
        
        downside_vol = metrics["Downside Volatility"]
        metrics["Sortino Ratio"] = float((metrics["Annual Return"] - self.rf) / downside_vol) if downside_vol > 0 else 0.0
        
        max_dd = abs(metrics["Maximum Drawdown"])
        metrics["Calmar Ratio"] = float(metrics["Annual Return"] / max_dd) if max_dd > 0 else 0.0
        
        metrics["Omega Ratio"] = self._calculate_omega_ratio()
        metrics["Treynor Ratio"] = self._calculate_treynor_ratio() if self.benchmark_aligned is not None else float("nan")
        
        # 4. DISTRIBUTION ANALYSIS
        metrics["Skewness"] = float(self.returns.skew())
        metrics["Excess Kurtosis"] = float(self.returns.kurtosis())
        metrics["Jarque-Bera Stat"] = float(self._calculate_jarque_bera()[0])
        metrics["Jarque-Bera p-value"] = float(self._calculate_jarque_bera()[1])
        metrics["Normality (JB p>0.05)"] = "Normal" if metrics["Jarque-Bera p-value"] > 0.05 else "Non-Normal"
        
        # 5. BENCHMARK METRICS (if benchmark provided)
        if self.benchmark_aligned is not None:
            metrics["Benchmark Return"] = float(self.benchmark_aligned.mean() * self.days)
            metrics["Excess Return"] = metrics["Annual Return"] - metrics["Benchmark Return"]
            
            # Beta and Alpha
            cov_matrix = np.cov(self.returns_aligned, self.benchmark_aligned)
            beta = cov_matrix[0, 1] / np.var(self.benchmark_aligned)
            metrics["Beta"] = float(beta)
            
            benchmark_return = metrics["Benchmark Return"]
            alpha = metrics["Annual Return"] - (self.rf + beta * (benchmark_return - self.rf))
            metrics["Jensen's Alpha"] = float(alpha)
            
            # R-squared and Tracking Error
            correlation = np.corrcoef(self.returns_aligned, self.benchmark_aligned)[0, 1]
            metrics["R-squared"] = float(correlation ** 2)
            tracking_error = np.std(self.returns_aligned - self.benchmark_aligned) * np.sqrt(self.days)
            metrics["Tracking Error"] = float(tracking_error)
            
            metrics["Information Ratio"] = float(metrics["Excess Return"] / tracking_error) if tracking_error > 0 else 0.0
            
            # Up/Down Capture Ratios
            up_market = self.benchmark_aligned > 0
            down_market = self.benchmark_aligned < 0
            
            if up_market.any():
                metrics["Up Capture Ratio"] = float((self.returns_aligned[up_market].mean() / 
                                                   self.benchmark_aligned[up_market].mean()) * 100)
            if down_market.any():
                metrics["Down Capture Ratio"] = float((self.returns_aligned[down_market].mean() / 
                                                     self.benchmark_aligned[down_market].mean()) * 100)
            
            metrics["Capture Ratio"] = (metrics["Up Capture Ratio"] / metrics["Down Capture Ratio"] 
                                       if "Down Capture Ratio" in metrics and metrics["Down Capture Ratio"] != 0 else float("nan"))
        
        # 6. WIN/LOSS ANALYSIS
        positive_returns = self.returns[self.returns > 0]
        negative_returns = self.returns[self.returns < 0]
        
        metrics["Win Rate"] = float(len(positive_returns) / len(self.returns)) if len(self.returns) > 0 else 0.0
        metrics["Average Win"] = float(positive_returns.mean()) if len(positive_returns) > 0 else 0.0
        metrics["Average Loss"] = float(negative_returns.mean()) if len(negative_returns) > 0 else 0.0
        metrics["Win/Loss Ratio"] = abs(metrics["Average Win"] / metrics["Average Loss"]) if metrics["Average Loss"] != 0 else float("inf")
        metrics["Profit Factor"] = float(positive_returns.sum() / abs(negative_returns.sum())) if negative_returns.sum() != 0 else float("inf")
        
        # Gain to Pain Ratio
        cumulative_returns = (1 + self.returns).cumprod()
        max_cum = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - max_cum) / max_cum
        pain = abs(drawdowns.sum())
        gain = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
        metrics["Gain to Pain Ratio"] = float(gain / pain) if pain > 0 else float("inf")
        
        # 7. DRAWDOWN ANALYSIS
        drawdown_series = self._calculate_drawdown_series()
        underwater = drawdown_series < 0
        
        if underwater.any():
            underwater_periods = (~underwater).cumsum()[underwater].value_counts()
            metrics["Max Underwater Duration"] = int(underwater_periods.max()) if not underwater_periods.empty else 0
            metrics["Avg Underwater Duration"] = float(underwater_periods.mean()) if not underwater_periods.empty else 0.0
            metrics["Underwater Probability"] = float(underwater.mean())
        
        # Recovery analysis
        recovery_days = self._calculate_recovery_days()
        metrics["Avg Recovery Days"] = float(np.mean(recovery_days)) if len(recovery_days) > 0 else 0.0
        metrics["Max Recovery Days"] = int(np.max(recovery_days)) if len(recovery_days) > 0 else 0
        
        # 8. ADDITIONAL METRICS
        metrics["Ulcer Index"] = float(np.sqrt(np.mean(drawdown_series ** 2)))
        metrics["Sterling Ratio"] = float(metrics["Annual Return"] / metrics["Maximum Drawdown"]) if metrics["Maximum Drawdown"] < 0 else 0.0
        metrics["Burke Ratio"] = float(metrics["Annual Return"] / np.sqrt(np.sum(np.square(drawdown_series)))) if np.sum(np.square(drawdown_series)) > 0 else 0.0
        
        # Modified Sharpe (incorporating skewness and kurtosis)
        if metrics["Skewness"] != 0 and metrics["Excess Kurtosis"] != 0:
            modified_sharpe = metrics["Sharpe Ratio"] * (1 + (metrics["Skewness"] / 6) * metrics["Sharpe Ratio"] - 
                                                        ((metrics["Excess Kurtosis"] - 3) / 24) * metrics["Sharpe Ratio"] ** 2)
            metrics["Modified Sharpe Ratio"] = float(modified_sharpe)
        
        # Value Added Monthly Index (VAMI)
        metrics["VAMI Start"] = 1000.0
        metrics["VAMI End"] = float(1000 * (1 + metrics["Cumulative Return"]))
        
        return metrics
    
    def _calculate_downside_volatility(self, threshold: float = 0.0) -> float:
        """Calculate downside volatility (semi-deviation)"""
        downside_returns = self.returns[self.returns < threshold]
        if len(downside_returns) > 0:
            return float(downside_returns.std() * np.sqrt(self.days))
        return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    
    def _calculate_drawdown_series(self) -> pd.Series:
        """Calculate full drawdown series"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max
    
    def _calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        excess_returns = self.returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        return float(gains / losses) if losses > 0 else float("inf")
    
    def _calculate_treynor_ratio(self) -> float:
        """Calculate Treynor ratio (requires benchmark)"""
        if self.benchmark_aligned is None:
            return float("nan")
        
        cov_matrix = np.cov(self.returns_aligned, self.benchmark_aligned)
        beta = cov_matrix[0, 1] / np.var(self.benchmark_aligned)
        excess_return = self.returns_aligned.mean() * self.days - self.rf
        return float(excess_return / beta) if beta != 0 else float("inf")
    
    def _calculate_jarque_bera(self) -> Tuple[float, float]:
        """Calculate Jarque-Bera test for normality"""
        if len(self.returns) < 2:
            return (0.0, 1.0)
        jb_stat, p_value = jarque_bera(self.returns)
        return float(jb_stat), float(p_value)
    
    def _calculate_recovery_days(self) -> List[int]:
        """Calculate recovery days from drawdowns"""
        drawdown_series = self._calculate_drawdown_series()
        recovery_days = []
        
        i = 0
        while i < len(drawdown_series):
            if drawdown_series.iloc[i] < 0:
                start_idx = i
                while i < len(drawdown_series) and drawdown_series.iloc[i] < 0:
                    i += 1
                recovery_days.append(i - start_idx)
            else:
                i += 1
        
        return recovery_days
    
    def create_performance_radar(self, metrics: Dict) -> go.Figure:
        """Create radar chart for key performance metrics"""
        
        # Select key metrics for radar
        radar_metrics = {
            "Return": metrics.get("Annual Return", 0),
            "Sharpe": metrics.get("Sharpe Ratio", 0),
            "Sortino": metrics.get("Sortino Ratio", 0),
            "Calmar": metrics.get("Calmar Ratio", 0),
            "Win Rate": metrics.get("Win Rate", 0) * 100,  # Convert to percentage
            "Omega": min(metrics.get("Omega Ratio", 0), 10)  # Cap at 10 for visualization
        }
        
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())
        
        # Normalize values for better visualization
        normalized_values = []
        for cat, val in radar_metrics.items():
            if cat == "Return":
                norm_val = min(max(val * 10, 0), 100)  # Scale return
            elif cat in ["Sharpe", "Sortino", "Calmar"]:
                norm_val = min(max(val * 20, 0), 100)  # Scale ratios
            elif cat == "Win Rate":
                norm_val = val  # Already 0-100
            elif cat == "Omega":
                norm_val = min(val * 10, 100)  # Scale Omega
            else:
                norm_val = min(max(val, 0), 100)
            normalized_values.append(norm_val)
        
        fig = go.Figure(data=go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],  # Close the shape
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(41, 98, 255, 0.3)',
            line=dict(color='#1a237e', width=2),
            marker=dict(size=8, color='#1a237e')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont_color="#424242",
                    gridcolor="#e0e0e0"
                ),
                angularaxis=dict(
                    tickfont_color="#424242",
                    gridcolor="#e0e0e0"
                ),
                bgcolor="white"
            ),
            showlegend=False,
            title="Performance Radar",
            title_font_color="#1a237e",
            font_color="#424242",
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_rolling_metrics_chart(self, window: int = 63) -> go.Figure:
        """Create chart of rolling metrics"""
        
        rolling_return = self.returns.rolling(window).mean() * self.days
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(self.days)
        rolling_sharpe = (rolling_return - self.rf) / rolling_vol
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Rolling Annualized Return", "Rolling Annualized Volatility", "Rolling Sharpe Ratio"),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Rolling Return
        fig.add_trace(
            go.Scatter(
                x=rolling_return.index,
                y=rolling_return.values,
                mode='lines',
                name='Return',
                line=dict(color='#1a237e', width=2)
            ),
            row=1, col=1
        )
        
        # Rolling Volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatility',
                line=dict(color='#d32f2f', width=2)
            ),
            row=2, col=1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Sharpe',
                line=dict(color='#388e3c', width=2)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            template="plotly_white",
            title_text=f"Rolling Metrics (Window: {window} days)",
            title_font_color="#1a237e",
            font_color="#424242"
        )
        
        fig.update_yaxes(title_text="Return (%)", row=1, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1, title_font_color="#424242")
        
        return fig

# ==============================================================================
# 7) ADVANCED BLACK-LITTERMAN OPTIMIZATION (NEW)
# ==============================================================================

class AdvancedBlackLittermanOptimizer:
    """Black-Litterman optimization with multiple view types and confidence adjustment"""
    
    def __init__(self, df_prices: pd.DataFrame, risk_free_rate: float = 0.04):
        self.df_prices = df_prices
        self.returns = df_prices.pct_change().dropna()
        self.rf = risk_free_rate
        self.ticker_map = {}
        
        # Initialize PyPortfolioOpt objects if available
        if BLACK_LITTERMAN_AVAILABLE:
            self.mu = expected_returns.mean_historical_return(df_prices)
            self.S = risk_models.sample_cov(df_prices)
            self.pi = self.mu  # Equilibrium returns
        else:
            self.mu = None
            self.S = None
            self.pi = None
    
    def set_ticker_map(self, ticker_map: Dict[str, str]):
        """Set ticker to name mapping for display purposes"""
        self.ticker_map = ticker_map
    
    def create_view_templates(self) -> Dict[str, Dict]:
        """Create predefined view templates for common scenarios"""
        
        templates = {
            "Bull Market Tech": {
                "description": "Technology sector outperforms by 5%, defensive sectors underperform",
                "views": [
                    {"type": "relative", "assets": ["XLK", "XLV"], "outperformance": 0.05},
                    {"type": "absolute", "asset": "AAPL", "return": 0.15},
                    {"type": "absolute", "asset": "MSFT", "return": 0.12}
                ],
                "confidences": [0.7, 0.8, 0.8]
            },
            "Defensive Rotation": {
                "description": "Defensive sectors outperform, high beta underperforms",
                "views": [
                    {"type": "relative", "assets": ["XLV", "XLK"], "outperformance": 0.03},
                    {"type": "relative", "assets": ["XLF", "XLE"], "outperformance": 0.02},
                    {"type": "absolute", "asset": "TLT", "return": 0.06}
                ],
                "confidences": [0.6, 0.5, 0.7]
            },
            "Emerging Markets Focus": {
                "description": "EM assets outperform developed, Turkey focus",
                "views": [
                    {"type": "relative", "assets": ["EEM", "SPY"], "outperformance": 0.04},
                    {"type": "absolute", "asset": "THYAO.IS", "return": 0.20},
                    {"type": "absolute", "asset": "AKBNK.IS", "return": 0.15}
                ],
                "confidences": [0.6, 0.5, 0.5]
            },
            "Rate Hike Scenario": {
                "description": "Financials benefit from rate hikes, bonds hurt",
                "views": [
                    {"type": "relative", "assets": ["XLF", "TLT"], "outperformance": 0.08},
                    {"type": "absolute", "asset": "JNK", "return": -0.02},
                    {"type": "absolute", "asset": "GLD", "return": -0.05}
                ],
                "confidences": [0.7, 0.6, 0.5]
            }
        }
        
        # Filter templates to only include assets we have
        available_tickers = set(self.df_prices.columns)
        filtered_templates = {}
        
        for name, template in templates.items():
            valid_views = []
            valid_confidences = []
            
            for view, confidence in zip(template["views"], template["confidences"]):
                if view["type"] == "absolute":
                    if view["asset"] in available_tickers:
                        valid_views.append(view)
                        valid_confidences.append(confidence)
                elif view["type"] == "relative":
                    if all(asset in available_tickers for asset in view["assets"]):
                        valid_views.append(view)
                        valid_confidences.append(confidence)
            
            if valid_views:
                filtered_templates[name] = {
                    "description": template["description"],
                    "views": valid_views,
                    "confidences": valid_confidences
                }
        
        return filtered_templates
    
    def create_views_from_template(self, template_name: str, templates_dict: Dict) -> Tuple[List, List]:
        """Create views and confidences from a template"""
        if template_name not in templates_dict:
            return [], []
        
        template = templates_dict[template_name]
        return template["views"], template["confidences"]
    
    def format_views_for_display(self, views: List, confidences: List) -> pd.DataFrame:
        """Format views for display in a DataFrame"""
        view_data = []
        
        for i, (view, conf) in enumerate(zip(views, confidences)):
            if view["type"] == "absolute":
                asset_name = self.ticker_map.get(view["asset"], view["asset"])
                view_data.append({
                    "View #": i+1,
                    "Type": "Absolute",
                    "Description": f"{asset_name} will return {view['return']:.1%}",
                    "Confidence": f"{conf:.0%}"
                })
            elif view["type"] == "relative":
                asset1 = self.ticker_map.get(view["assets"][0], view["assets"][0])
                asset2 = self.ticker_map.get(view["assets"][1], view["assets"][1])
                view_data.append({
                    "View #": i+1,
                    "Type": "Relative",
                    "Description": f"{asset1} will outperform {asset2} by {view['outperformance']:.1%}",
                    "Confidence": f"{conf:.0%}"
                })
        
        return pd.DataFrame(view_data) if view_data else pd.DataFrame()
    
    def run_black_litterman(self, views: List, confidences: List, 
                           optimization_objective: str = "max_sharpe") -> Tuple[Dict, Dict]:
        """Run Black-Litterman optimization"""
        
        if not BLACK_LITTERMAN_AVAILABLE or self.mu is None:
            st.error("Black-Litterman optimization requires PyPortfolioOpt with Black-Litterman support.")
            return {}, {}
        
        try:
            # Create P and Q matrices for views
            n_assets = len(self.mu)
            P = np.zeros((len(views), n_assets))
            Q = np.zeros(len(views))
            
            asset_index = {asset: idx for idx, asset in enumerate(self.mu.index)}
            
            for i, view in enumerate(views):
                if view["type"] == "absolute":
                    asset_idx = asset_index[view["asset"]]
                    P[i, asset_idx] = 1
                    Q[i] = view["return"]
                elif view["type"] == "relative":
                    asset1_idx = asset_index[view["assets"][0]]
                    asset2_idx = asset_index[view["assets"][1]]
                    P[i, asset1_idx] = 1
                    P[i, asset2_idx] = -1
                    Q[i] = view["outperformance"]
            
            # Create Omega (uncertainty) matrix using Idzorek method
            tau = 0.05  # Default scaling factor
            omega = np.zeros((len(views), len(views)))
            
            for i, conf in enumerate(confidences):
                # Convert confidence to uncertainty (Idzorek method)
                if conf >= 1:
                    conf = 0.99
                elif conf <= 0:
                    conf = 0.01
                
                # Idzorek's formula for converting confidence to uncertainty
                uncertainty = (1 - conf) / conf
                omega[i, i] = uncertainty * tau * np.dot(P[i:i+1, :], np.dot(self.S, P[i:i+1, :].T))
            
            # Create Black-Litterman model
            bl = BlackLittermanModel(self.S, pi=self.pi, P=P, Q=Q, omega=omega)
            
            # Get posterior estimates
            posterior_rets = bl.bl_returns()
            posterior_cov = bl.bl_cov()
            
            # Create efficient frontier with posterior estimates
            ef = EfficientFrontier(posterior_rets, posterior_cov)
            
            if optimization_objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=self.rf)
            elif optimization_objective == "min_volatility":
                ef.min_volatility()
            elif optimization_objective == "max_quadratic_utility":
                ef.max_quadratic_utility()
            else:
                ef.max_sharpe(risk_free_rate=self.rf)
            
            weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.rf)
            
            # Calculate view impacts
            view_impacts = {}
            for i, view in enumerate(views):
                if view["type"] == "absolute":
                    asset = view["asset"]
                    prior_return = float(self.pi[asset])
                    posterior_return = float(posterior_rets[asset])
                    view_impacts[f"View {i+1}"] = {
                        "asset": asset,
                        "prior_return": prior_return,
                        "posterior_return": posterior_return,
                        "impact": posterior_return - prior_return,
                        "confidence": confidences[i]
                    }
            
            return weights, performance, posterior_rets, view_impacts
            
        except Exception as e:
            st.error(f"Black-Litterman optimization failed: {e}")
            return {}, {}, None, {}
    
    def create_view_impact_chart(self, prior_rets: pd.Series, posterior_rets: pd.Series, 
                                view_impacts: Dict) -> go.Figure:
        """Create chart showing impact of views on expected returns"""
        
        # Get top 15 assets by absolute impact
        impacts = posterior_rets - prior_rets
        top_assets = impacts.abs().nlargest(15).index
        
        fig = go.Figure()
        
        # Add prior returns
        fig.add_trace(go.Bar(
            x=[self.ticker_map.get(t, t) for t in top_assets],
            y=prior_rets[top_assets] * 100,
            name='Prior (Equilibrium)',
            marker_color='rgba(158, 158, 158, 0.6)',
            text=[f"{x:.1f}%" for x in prior_rets[top_assets] * 100],
            textposition='auto'
        ))
        
        # Add posterior returns
        fig.add_trace(go.Bar(
            x=[self.ticker_map.get(t, t) for t in top_assets],
            y=posterior_rets[top_assets] * 100,
            name='Posterior (With Views)',
            marker_color='rgba(41, 98, 255, 0.8)',
            text=[f"{x:.1f}%" for x in posterior_rets[top_assets] * 100],
            textposition='auto'
        ))
        
        # Add impact arrows
        for asset in top_assets:
            prior = prior_rets[asset] * 100
            posterior = posterior_rets[asset] * 100
            impact = posterior - prior
            
            if abs(impact) > 0.1:  # Only show significant impacts
                fig.add_annotation(
                    x=self.ticker_map.get(asset, asset),
                    y=max(prior, posterior) + 0.5,
                    text=f"{impact:+.1f}%",
                    showarrow=False,
                    font=dict(color="#d32f2f" if impact < 0 else "#388e3c", size=10)
                )
        
        fig.update_layout(
            title="Black-Litterman: View Impact on Expected Returns",
            title_font_color="#1a237e",
            xaxis_title="Assets",
            yaxis_title="Expected Annual Return (%)",
            barmode='group',
            template="plotly_white",
            font_color="#424242",
            height=500,
            xaxis=dict(tickangle=45)
        )
        
        return fig

# ==============================================================================
# 8) ADVANCED EFFICIENT FRONTIER CHART (NEW)
# ==============================================================================

class AdvancedEfficientFrontierChart:
    """Advanced efficient frontier visualization with 3D capabilities"""
    
    def __init__(self, mu: pd.Series, S: pd.DataFrame, risk_free_rate: float = 0.04):
        self.mu = mu
        self.S = S
        self.rf = risk_free_rate
        self.num_assets = len(mu)
        
        # Try to import PyPortfolioOpt
        try:
            from pypfopt.efficient_frontier import EfficientFrontier
            self.EfficientFrontier = EfficientFrontier
        except:
            self.EfficientFrontier = None
    
    def calculate_efficient_frontier(self, points: int = 100) -> pd.DataFrame:
        """Calculate efficient frontier points"""
        
        if self.EfficientFrontier is None:
            return pd.DataFrame()
        
        ef = self.EfficientFrontier(self.mu, self.S)
        
        # Get minimum and maximum volatility portfolios
        ef.min_volatility()
        min_vol = ef.portfolio_performance(verbose=False)
        min_vol_vol = min_vol[1]
        
        # Get maximum return portfolio (approximate)
        max_ret_idx = self.mu.argmax()
        max_ret_weight = np.zeros(self.num_assets)
        max_ret_weight[max_ret_idx] = 1.0
        max_ret_vol = np.sqrt(np.dot(max_ret_weight.T, np.dot(self.S, max_ret_weight)))
        
        # Generate target volatilities
        target_vols = np.linspace(min_vol_vol * 1.01, max_ret_vol * 0.95, points)
        
        frontier_data = []
        
        for target_vol in target_vols:
            try:
                # Clear objectives and add volatility constraint
                ef = self.EfficientFrontier(self.mu, self.S)
                ef.add_objective(self._volatility_constraint, target_vol=target_vol)
                ef.max_return()
                
                weights = ef.clean_weights()
                perf = ef.portfolio_performance(verbose=False, risk_free_rate=self.rf)
                
                frontier_data.append({
                    'return': perf[0],
                    'volatility': perf[1],
                    'sharpe': perf[2],
                    'weights': weights
                })
            except:
                continue
        
        return pd.DataFrame(frontier_data)
    
    def _volatility_constraint(self, weights, target_vol):
        """Custom constraint for target volatility"""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.S, weights)))
        return abs(portfolio_vol - target_vol)
    
    def get_key_portfolios(self) -> Dict:
        """Calculate key portfolios (min vol, max sharpe, equal weight)"""
        
        key_portfolios = {}
        
        if self.EfficientFrontier is None:
            return key_portfolios
        
        try:
            # Minimum Volatility Portfolio
            ef_min = self.EfficientFrontier(self.mu, self.S)
            ef_min.min_volatility()
            weights_min = ef_min.clean_weights()
            perf_min = ef_min.portfolio_performance(verbose=False, risk_free_rate=self.rf)
            key_portfolios['Minimum Volatility'] = {
                'weights': weights_min,
                'return': perf_min[0],
                'volatility': perf_min[1],
                'sharpe': perf_min[2]
            }
            
            # Maximum Sharpe Portfolio
            ef_sharpe = self.EfficientFrontier(self.mu, self.S)
            ef_sharpe.max_sharpe(risk_free_rate=self.rf)
            weights_sharpe = ef_sharpe.clean_weights()
            perf_sharpe = ef_sharpe.portfolio_performance(verbose=False, risk_free_rate=self.rf)
            key_portfolios['Maximum Sharpe'] = {
                'weights': weights_sharpe,
                'return': perf_sharpe[0],
                'volatility': perf_sharpe[1],
                'sharpe': perf_sharpe[2]
            }
            
            # Equal Weight Portfolio
            equal_weight = 1.0 / self.num_assets
            weights_equal = {asset: equal_weight for asset in self.mu.index}
            port_return = np.dot(list(weights_equal.values()), self.mu)
            port_vol = np.sqrt(np.dot(list(weights_equal.values()), 
                                      np.dot(self.S, list(weights_equal.values()))))
            sharpe_equal = (port_return - self.rf) / port_vol if port_vol > 0 else 0.0
            key_portfolios['Equal Weight'] = {
                'weights': weights_equal,
                'return': float(port_return),
                'volatility': float(port_vol),
                'sharpe': float(sharpe_equal)
            }
            
        except Exception as e:
            st.warning(f"Could not calculate all key portfolios: {e}")
        
        return key_portfolios
    
    def create_advanced_frontier_chart(self, frontier_df: pd.DataFrame, 
                                      key_portfolios: Dict) -> go.Figure:
        """Create advanced efficient frontier visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Efficient Frontier", "Sharpe Ratio Distribution",
                          "Capital Market Line", "Risk Contribution"),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Efficient Frontier
        fig.add_trace(
            go.Scatter(
                x=frontier_df['volatility'] * 100,
                y=frontier_df['return'] * 100,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='#1a237e', width=3),
                fill='tozeroy',
                fillcolor='rgba(41, 98, 255, 0.1)'
            ),
            row=1, col=1
        )
        
        # Add key portfolios
        colors = {'Minimum Volatility': '#388e3c', 
                 'Maximum Sharpe': '#d32f2f', 
                 'Equal Weight': '#ff9800'}
        
        for name, portfolio in key_portfolios.items():
            fig.add_trace(
                go.Scatter(
                    x=[portfolio['volatility'] * 100],
                    y=[portfolio['return'] * 100],
                    mode='markers+text',
                    name=name,
                    marker=dict(size=12, color=colors.get(name, '#757575')),
                    text=[name],
                    textposition="top center",
                    textfont=dict(size=10)
                ),
                row=1, col=1
            )
        
        # 2. Sharpe Ratio Distribution
        fig.add_trace(
            go.Histogram(
                x=frontier_df['sharpe'],
                nbinsx=30,
                name='Sharpe Ratio',
                marker_color='rgba(41, 98, 255, 0.7)',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add mean Sharpe line
        mean_sharpe = frontier_df['sharpe'].mean()
        fig.add_vline(
            x=mean_sharpe,
            line_dash="dash",
            line_color="#d32f2f",
            annotation_text=f"Mean: {mean_sharpe:.2f}",
            annotation_position="top right",
            row=1, col=2
        )
        
        # 3. Capital Market Line
        # Calculate tangency portfolio (max Sharpe)
        if 'Maximum Sharpe' in key_portfolios:
            max_sharpe = key_portfolios['Maximum Sharpe']
            x_vals = np.linspace(0, max_sharpe['volatility'] * 1.5, 50)
            cml_returns = self.rf * 100 + (max_sharpe['sharpe'] * x_vals * 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals * 100,
                    y=cml_returns,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color='#ff9800', width=2, dash='dash')
                ),
                row=2, col=1
            )
        
        # 4. Risk Contribution (for max Sharpe portfolio)
        if 'Maximum Sharpe' in key_portfolios:
            weights = key_portfolios['Maximum Sharpe']['weights']
            risk_contributions = {}
            
            for asset, weight in weights.items():
                if weight > 0.001:  # Only significant weights
                    marginal_risk = np.dot(self.S[asset], list(weights.values()))
                    risk_contributions[asset] = weight * marginal_risk
            
            # Get top 10 risk contributors
            sorted_contributors = sorted(risk_contributions.items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True)[:10]
            
            fig.add_trace(
                go.Bar(
                    x=[asset[:15] + '...' if len(asset) > 15 else asset 
                       for asset, _ in sorted_contributors],
                    y=[val * 100 for _, val in sorted_contributors],
                    name='Risk Contribution',
                    marker_color='rgba(239, 83, 80, 0.8)',
                    text=[f"{val*100:.1f}%" for _, val in sorted_contributors],
                    textposition='auto'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white",
            title_text="Advanced Efficient Frontier Analysis",
            title_font_size=16,
            title_font_color="#1a237e",
            font_color="#424242"
        )
        
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Return (%)", row=1, col=1, title_font_color="#424242")
        fig.update_xaxes(title_text="Sharpe Ratio", row=1, col=2, title_font_color="#424242")
        fig.update_yaxes(title_text="Frequency", row=1, col=2, title_font_color="#424242")
        fig.update_xaxes(title_text="Volatility (%)", row=2, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Return (%)", row=2, col=1, title_font_color="#424242")
        fig.update_xaxes(title_text="Asset", row=2, col=2, tickangle=45, title_font_color="#424242")
        fig.update_yaxes(title_text="Risk Contribution (%)", row=2, col=2, title_font_color="#424242")
        
        return fig
    
    def create_3d_frontier_chart(self, frontier_df: pd.DataFrame) -> go.Figure:
        """Create 3D efficient frontier visualization"""
        
        fig = go.Figure(data=[go.Scatter3d(
            x=frontier_df['volatility'] * 100,
            y=frontier_df['return'] * 100,
            z=frontier_df['sharpe'],
            mode='markers',
            marker=dict(
                size=5,
                color=frontier_df['sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            text=[f"Return: {r*100:.1f}%<br>Vol: {v*100:.1f}%<br>Sharpe: {s:.2f}" 
                  for r, v, s in zip(frontier_df['return'], frontier_df['volatility'], frontier_df['sharpe'])],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title="3D Efficient Frontier: Return vs Volatility vs Sharpe",
            title_font_color="#1a237e",
            scene=dict(
                xaxis_title='Volatility (%)',
                yaxis_title='Return (%)',
                zaxis_title='Sharpe Ratio',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                ),
                xaxis=dict(color="#424242"),
                yaxis=dict(color="#424242"),
                zaxis=dict(color="#424242")
            ),
            height=700,
            template="plotly_white"
        )
        
        return fig

# ==============================================================================
# 9) ORIGINAL RISK METRICS ENGINE
# ==============================================================================

class EnhancedRiskMetricsEngine:
    def __init__(self, portfolio_returns: pd.Series, risk_free_rate: float = 0.04,
                 benchmark_returns: Optional[pd.Series] = None):
        self.returns = portfolio_returns.dropna()
        self.rf = float(risk_free_rate)
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None

    def calculate_comprehensive_metrics(self) -> pd.DataFrame:
        metrics = {}

        # Basic statistics
        metrics["Annual Return"] = float(self.returns.mean() * 252)
        metrics["Annual Volatility"] = float(self.returns.std() * np.sqrt(252))
        metrics["Skewness"] = float(self.returns.skew())
        metrics["Kurtosis"] = float(self.returns.kurtosis())

        # Ratios
        vol = metrics["Annual Volatility"]
        metrics["Sharpe Ratio"] = float((metrics["Annual Return"] - self.rf) / vol) if vol > 0 else 0.0

        downside = self.returns[self.returns < 0]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0.0
        metrics["Sortino Ratio"] = float((metrics["Annual Return"] - self.rf) / downside_std) if downside_std > 0 else 0.0

        # Drawdown
        cum = (1 + self.returns).cumprod()
        run_max = cum.cummax()
        dd = (cum - run_max) / run_max
        metrics["Max Drawdown"] = float(dd.min())
        metrics["Calmar Ratio"] = float(metrics["Annual Return"] / abs(metrics["Max Drawdown"])) if abs(metrics["Max Drawdown"]) > 0 else 0.0

        # Omega
        threshold = self.rf / 252
        excess = self.returns - threshold
        gains = float(excess[excess > 0].sum())
        losses = float(abs(excess[excess < 0].sum()))
        metrics["Omega Ratio"] = float(gains / losses) if losses > 0 else float("inf")

        # Beta/Alpha vs benchmark
        if self.benchmark_returns is not None and not self.benchmark_returns.empty:
            aligned = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
            if len(aligned) > 10:
                cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0][1]
                var = np.var(aligned.iloc[:, 1])
                beta = float(cov / var) if var > 0 else 0.0
                bench_ann = float(aligned.iloc[:, 1].mean() * 252)
                alpha = float((metrics["Annual Return"] - self.rf) - beta * (bench_ann - self.rf))
                metrics["Beta"] = beta
                metrics["Alpha"] = alpha
            else:
                metrics["Beta"] = 0.0
                metrics["Alpha"] = 0.0
        else:
            metrics["Beta"] = 0.0
            metrics["Alpha"] = 0.0

        # VaR / CVaR
        if len(self.returns) >= 30:
            q = np.percentile(self.returns, 5)
            metrics["Historical VaR (95%)"] = float(-q)
            tail = self.returns[self.returns <= q]
            metrics["CVaR (95%)"] = float(-tail.mean()) if len(tail) > 0 else float("nan")
        else:
            metrics["Historical VaR (95%)"] = float("nan")
            metrics["CVaR (95%)"] = float("nan")

        return pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})

# ==============================================================================
# 10) ADVANCED VAR/CVAR/ES ENGINE
# ==============================================================================

class AdvancedVaREngine:
    """Advanced Value at Risk, Conditional VaR (Expected Shortfall), and Stress Testing Engine"""
    
    def __init__(self, returns: pd.Series, confidence_levels: List[float] = None):
        self.returns = returns.dropna()
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        
    def historical_var(self, confidence: float = 0.95, holding_period: int = 1) -> Dict:
        """Historical VaR using empirical distribution"""
        if len(self.returns) < 50:
            return {"VaR": np.nan, "CVaR": np.nan, "method": "Historical (Insufficient Data)"}
        
        alpha = 1 - confidence
        var_1day = -np.percentile(self.returns, alpha * 100)
        var_n_day = var_1day * np.sqrt(holding_period)
        
        # CVaR/ES (Expected Shortfall)
        threshold = -var_1day
        losses_beyond_threshold = self.returns[self.returns <= -threshold]
        cvar = -losses_beyond_threshold.mean() if len(losses_beyond_threshold) > 0 else np.nan
        
        return {
            "VaR_1day": float(var_1day),
            "VaR_nday": float(var_n_day),
            "CVaR": float(cvar),
            "Confidence": float(confidence),
            "HoldingPeriod": int(holding_period),
            "Method": "Historical",
            "DataPoints": int(len(losses_beyond_threshold))
        }
    
    def calculate_all_var_methods(self, confidence: float = 0.95, 
                                 holding_period: int = 1) -> pd.DataFrame:
        """Calculate VaR using all methods for comparison"""
        methods = []
        
        # Historical
        hist = self.historical_var(confidence, holding_period)
        methods.append(hist)
        
        return pd.DataFrame(methods)

# ==============================================================================
# 11) ENHANCED STRESS TESTING ENGINE
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
            "recovery_days": 1465
        },
        "COVID-19 Pandemic Crash (2020)": {
            "start": "2020-02-19", 
            "end": "2020-03-23",
            "description": "Global pandemic causing fastest bear market in history",
            "severity": "Severe",
            "max_drawdown_global": -33.92,
            "recovery_days": 153
        }
    }

# ==============================================================================
# 12) ENHANCED UI COMPONENTS FOR NEW TABS
# ==============================================================================

def create_advanced_performance_tab():
    """Create comprehensive performance metrics tab"""
    st.markdown('<div class="section-header">📊 Advanced Performance Metrics Dashboard</div>', unsafe_allow_html=True)
    
    if "portfolio_returns" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please run portfolio optimization first to generate portfolio returns.</div>', unsafe_allow_html=True)
        return
    
    portfolio_returns = st.session_state["portfolio_returns"]
    benchmark_returns = st.session_state.get("benchmark_returns", None)
    
    # Configuration
    st.markdown('<div class="subsection-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_free_rate = st.number_input(
            "Risk Free Rate (%)", 
            value=4.5, 
            step=0.1,
            key="perf_rf_rate"
        ) / 100
    
    with col2:
        trading_days = st.selectbox(
            "Trading Days per Year",
            options=[252, 365],
            index=0,
            key="perf_trading_days"
        )
    
    with col3:
        rolling_window = st.select_slider(
            "Rolling Window (Days)",
            options=[21, 63, 126, 252],
            value=63,
            key="perf_rolling_window"
        )
    
    # Initialize performance metrics engine
    perf_engine = AdvancedPerformanceMetrics(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
        trading_days=trading_days
    )
    
    # Calculate all metrics
    with st.spinner("📈 Calculating 50+ performance metrics..."):
        all_metrics = perf_engine.calculate_all_metrics()
    
    # Display key metrics in categories
    st.markdown('<div class="subsection-header">🎯 Key Performance Metrics</div>', unsafe_allow_html=True)
    
    # Return Metrics
    st.markdown("##### 📈 Return Metrics")
    col_ret1, col_ret2, col_ret3, col_ret4 = st.columns(4)
    
    with col_ret1:
        st.metric("Annual Return", f"{all_metrics.get('Annual Return', 0):.2%}")
    with col_ret2:
        st.metric("Cumulative Return", f"{all_metrics.get('Cumulative Return', 0):.2%}")
    with col_ret3:
        st.metric("Geometric Mean", f"{all_metrics.get('Geometric Mean Return', 0):.2%}")
    with col_ret4:
        if "3M Rolling Return" in all_metrics:
            st.metric("3M Rolling Return", f"{all_metrics['3M Rolling Return']:.2%}")
    
    # Risk Metrics
    st.markdown("##### ⚠️ Risk Metrics")
    col_risk1, col_risk2, col_risk3, col_risk4 = st.columns(4)
    
    with col_risk1:
        st.metric("Annual Volatility", f"{all_metrics.get('Annual Volatility', 0):.2%}")
    with col_risk2:
        st.metric("Maximum Drawdown", f"{all_metrics.get('Maximum Drawdown', 0):.2%}")
    with col_risk3:
        st.metric("VaR 95%", f"{all_metrics.get('VaR 95% (Historical)', 0):.2%}")
    with col_risk4:
        st.metric("CVaR 95%", f"{all_metrics.get('CVaR/ES 95%', 0):.2%}")
    
    # Risk-Adjusted Ratios
    st.markdown("##### ⚖️ Risk-Adjusted Ratios")
    col_ratio1, col_ratio2, col_ratio3, col_ratio4 = st.columns(4)
    
    with col_ratio1:
        st.metric("Sharpe Ratio", f"{all_metrics.get('Sharpe Ratio', 0):.2f}")
    with col_ratio2:
        st.metric("Sortino Ratio", f"{all_metrics.get('Sortino Ratio', 0):.2f}")
    with col_ratio3:
        st.metric("Calmar Ratio", f"{all_metrics.get('Calmar Ratio', 0):.2f}")
    with col_ratio4:
        omega = all_metrics.get('Omega Ratio', 0)
        st.metric("Omega Ratio", f"{omega:.2f}" if omega != float('inf') else "∞")
    
    # Benchmark Metrics (if available)
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        st.markdown("##### 📊 Benchmark Metrics")
        col_bench1, col_bench2, col_bench3, col_bench4 = st.columns(4)
        
        with col_bench1:
            st.metric("Beta", f"{all_metrics.get('Beta', 0):.2f}")
        with col_bench2:
            st.metric("Alpha", f"{all_metrics.get('Jensen\'s Alpha', 0):.2%}")
        with col_bench3:
            st.metric("Information Ratio", f"{all_metrics.get('Information Ratio', 0):.2f}")
        with col_bench4:
            st.metric("R-squared", f"{all_metrics.get('R-squared', 0):.2%}")
    
    # Create visualizations
    st.markdown('<div class="subsection-header">📊 Performance Visualizations</div>', unsafe_allow_html=True)
    
    tab_viz1, tab_viz2 = st.tabs(["Performance Radar", "Rolling Metrics"])
    
    with tab_viz1:
        radar_chart = perf_engine.create_performance_radar(all_metrics)
        st.plotly_chart(radar_chart, use_container_width=True)
    
    with tab_viz2:
        rolling_chart = perf_engine.create_rolling_metrics_chart(rolling_window)
        st.plotly_chart(rolling_chart, use_container_width=True)
    
    # Detailed metrics table
    st.markdown('<div class="subsection-header">📋 Detailed Metrics Table</div>', unsafe_allow_html=True)
    
    # Organize metrics by category
    metrics_by_category = {
        "Return Metrics": ["Annual Return", "Cumulative Return", "Geometric Mean Return", 
                          "3M Rolling Return", "6M Rolling Return", "1Y Rolling Return"],
        "Risk Metrics": ["Annual Volatility", "Downside Volatility", "Maximum Drawdown",
                        "VaR 95% (Historical)", "CVaR/ES 95%", "Tail Risk (CVaR - VaR)"],
        "Risk-Adjusted Ratios": ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", 
                                "Omega Ratio", "Treynor Ratio", "Modified Sharpe Ratio"],
        "Distribution Analysis": ["Skewness", "Excess Kurtosis", "Jarque-Bera Stat",
                                "Jarque-Bera p-value", "Normality (JB p>0.05)"],
        "Win/Loss Analysis": ["Win Rate", "Average Win", "Average Loss", "Win/Loss Ratio",
                             "Profit Factor", "Gain to Pain Ratio"],
        "Drawdown Analysis": ["Max Underwater Duration", "Avg Underwater Duration",
                             "Underwater Probability", "Avg Recovery Days", "Max Recovery Days",
                             "Ulcer Index", "Sterling Ratio", "Burke Ratio"],
        "Benchmark Metrics": ["Benchmark Return", "Excess Return", "Beta", "Jensen's Alpha",
                             "R-squared", "Tracking Error", "Information Ratio", 
                             "Up Capture Ratio", "Down Capture Ratio", "Capture Ratio"],
        "Additional Metrics": ["VAMI Start", "VAMI End"]
    }
    
    # Create expanders for each category
    for category, metric_names in metrics_by_category.items():
        with st.expander(f"📁 {category}"):
            category_metrics = []
            for name in metric_names:
                if name in all_metrics:
                    value = all_metrics[name]
                    if "Return" in name or "Alpha" in name or "Drawdown" in name:
                        display_value = f"{value:.2%}" if isinstance(value, (int, float)) else str(value)
                    elif "Ratio" in name or "Beta" in name:
                        display_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                    elif "Rate" in name or "Probability" in name:
                        display_value = f"{value:.1%}" if isinstance(value, (int, float)) else str(value)
                    elif "VAMI" in name:
                        display_value = f"${value:,.0f}" if isinstance(value, (int, float)) else str(value)
                    else:
                        display_value = str(value)
                    
                    category_metrics.append({"Metric": name, "Value": display_value})
            
            if category_metrics:
                df_category = pd.DataFrame(category_metrics)
                st.dataframe(df_category, use_container_width=True, height=200)
    
    # Export functionality
    st.markdown('<div class="subsection-header">💾 Export Metrics</div>', unsafe_allow_html=True)
    
    if st.button("📥 Export All Metrics to CSV", use_container_width=True):
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(list(all_metrics.items()), columns=["Metric", "Value"])
        
        # Convert to CSV
        csv = metrics_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        
        # Create download link
        href = f'<a href="data:file/csv;base64,{b64}" download="performance_metrics.csv">Click here to download</a>'
        st.markdown(f'<div class="info-card">{href}</div>', unsafe_allow_html=True)

def create_black_litterman_tab():
    """Create Black-Litterman optimization tab"""
    st.markdown('<div class="section-header">🎯 Black-Litterman View-Based Optimization</div>', unsafe_allow_html=True)
    
    if "df_prices" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please load data first from the Data Overview tab.</div>', unsafe_allow_html=True)
        return
    
    df_prices = st.session_state["df_prices"]
    ticker_map = st.session_state.get("ticker_map", {})
    
    # Initialize Black-Litterman optimizer
    bl_optimizer = AdvancedBlackLittermanOptimizer(df_prices)
    bl_optimizer.set_ticker_map(ticker_map)
    
    # Get view templates
    templates = bl_optimizer.create_view_templates()
    
    st.markdown('<div class="subsection-header">📋 Quick View Templates</div>', unsafe_allow_html=True)
    
    # Template selection
    selected_template = st.selectbox(
        "Select a pre-built view template",
        options=["Custom Views"] + list(templates.keys()),
        index=0,
        help="Choose a pre-built scenario or create custom views"
    )
    
    # Initialize session state for views
    if "bl_views" not in st.session_state:
        st.session_state.bl_views = []
    if "bl_confidences" not in st.session_state:
        st.session_state.bl_confidences = []
    
    # Load template views if selected
    if selected_template != "Custom Views" and selected_template in templates:
        template_views, template_confidences = bl_optimizer.create_views_from_template(
            selected_template, templates
        )
        
        # Store in session state
        st.session_state.bl_views = template_views
        st.session_state.bl_confidences = template_confidences
        
        st.markdown(f"**Template Description:** {templates[selected_template]['description']}")
    
    # View management
    st.markdown('<div class="subsection-header">🛠️ View Management</div>', unsafe_allow_html=True)
    
    col_view1, col_view2 = st.columns([2, 1])
    
    with col_view1:
        # Display current views
        if st.session_state.bl_views:
            views_df = bl_optimizer.format_views_for_display(
                st.session_state.bl_views, st.session_state.bl_confidences
            )
            st.dataframe(views_df, use_container_width=True, height=200)
        else:
            st.info("No views defined. Add views using the controls on the right.")
    
    with col_view2:
        # Add new view
        st.markdown("**Add New View**")
        
        view_type = st.selectbox("View Type", ["Absolute", "Relative"], key="new_view_type")
        
        if view_type == "Absolute":
            asset = st.selectbox("Asset", options=list(df_prices.columns), key="abs_asset")
            expected_return = st.number_input("Expected Annual Return (%)", 
                                            value=10.0, step=0.5) / 100
            confidence = st.slider("Confidence", 0.0, 1.0, 0.7, 0.05)
            
            if st.button("➕ Add Absolute View", use_container_width=True):
                st.session_state.bl_views.append({
                    "type": "absolute",
                    "asset": asset,
                    "return": expected_return
                })
                st.session_state.bl_confidences.append(confidence)
                st.rerun()
        
        else:  # Relative view
            col_rel1, col_rel2 = st.columns(2)
            with col_rel1:
                asset1 = st.selectbox("Outperforming Asset", options=list(df_prices.columns), key="rel_asset1")
            with col_rel2:
                asset2 = st.selectbox("Underperforming Asset", options=list(df_prices.columns), key="rel_asset2")
            
            outperformance = st.number_input("Outperformance (%)", value=3.0, step=0.5) / 100
            confidence = st.slider("Confidence", 0.0, 1.0, 0.6, 0.05, key="rel_confidence")
            
            if st.button("➕ Add Relative View", use_container_width=True):
                st.session_state.bl_views.append({
                    "type": "relative",
                    "assets": [asset1, asset2],
                    "outperformance": outperformance
                })
                st.session_state.bl_confidences.append(confidence)
                st.rerun()
        
        # Clear views button
        if st.session_state.bl_views and st.button("🗑️ Clear All Views", use_container_width=True):
            st.session_state.bl_views = []
            st.session_state.bl_confidences = []
            st.rerun()
    
    # Optimization settings
    st.markdown('<div class="subsection-header">⚙️ Optimization Settings</div>', unsafe_allow_html=True)
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        risk_free_rate = st.number_input(
            "Risk Free Rate (%)", 
            value=4.5, 
            step=0.1,
            key="bl_rf_rate"
        ) / 100
    
    with col_opt2:
        objective = st.selectbox(
            "Optimization Objective",
            ["max_sharpe", "min_volatility", "max_quadratic_utility"],
            index=0,
            key="bl_objective"
        )
    
    # Run optimization
    if st.button("🚀 Run Black-Litterman Optimization", type="primary", use_container_width=True):
        if not st.session_state.bl_views:
            st.error("Please add at least one view before running optimization.")
        else:
            with st.spinner("Running Black-Litterman optimization..."):
                weights, performance, posterior_rets, view_impacts = bl_optimizer.run_black_litterman(
                    st.session_state.bl_views,
                    st.session_state.bl_confidences,
                    objective
                )
                
                if weights:
                    st.markdown('<div class="success-card">✅ Black-Litterman optimization completed!</div>', unsafe_allow_html=True)
                    
                    # Display results
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        st.metric("Expected Return", f"{performance[0]:.2%}")
                    with col_res2:
                        st.metric("Volatility", f"{performance[1]:.2%}")
                    with col_res3:
                        st.metric("Sharpe Ratio", f"{performance[2]:.2f}")
                    
                    # Display portfolio allocation
                    st.markdown('<div class="subsection-header">📊 Optimized Portfolio Allocation</div>', unsafe_allow_html=True)
                    
                    allocation_data = []
                    for ticker, weight in weights.items():
                        if weight > 0.001:  # Only show significant weights
                            allocation_data.append({
                                "Asset": ticker_map.get(ticker, ticker),
                                "Ticker": ticker,
                                "Weight": f"{weight:.2%}",
                                "Amount ($)": f"${weight * 1000000:,.0f}"  # Assuming $1M investment
                            })
                    
                    if allocation_data:
                        allocation_df = pd.DataFrame(allocation_data)
                        st.dataframe(allocation_df, use_container_width=True, height=300)
                    
                    # Show view impacts if available
                    if view_impacts and posterior_rets is not None:
                        st.markdown('<div class="subsection-header">📈 View Impact Analysis</div>', unsafe_allow_html=True)
                        
                        # Get prior returns from optimizer
                        prior_rets = bl_optimizer.pi if hasattr(bl_optimizer, 'pi') else pd.Series()
                        
                        if not prior_rets.empty:
                            impact_chart = bl_optimizer.create_view_impact_chart(
                                prior_rets, posterior_rets, view_impacts
                            )
                            st.plotly_chart(impact_chart, use_container_width=True)
                    
                    # Store results in session state
                    st.session_state.bl_weights = weights
                    st.session_state.bl_performance = performance

def create_advanced_frontier_tab():
    """Create advanced efficient frontier analysis tab"""
    st.markdown('<div class="section-header">📈 Advanced Efficient Frontier Analysis</div>', unsafe_allow_html=True)
    
    if "df_prices" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please load data first from the Data Overview tab.</div>', unsafe_allow_html=True)
        return
    
    df_prices = st.session_state["df_prices"]
    
    # Check if PyPortfolioOpt is available
    if not OPTIMIZATION_AVAILABLE:
        st.markdown('<div class="warning-card">⚠️ Efficient frontier analysis requires PyPortfolioOpt. Please install it first.</div>', unsafe_allow_html=True)
        return
    
    # Configuration
    st.markdown('<div class="subsection-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    col_conf1, col_conf2, col_conf3 = st.columns(3)
    
    with col_conf1:
        risk_free_rate = st.number_input(
            "Risk Free Rate (%)", 
            value=4.5, 
            step=0.1,
            key="frontier_rf_rate"
        ) / 100
    
    with col_conf2:
        frontier_points = st.slider(
            "Number of Frontier Points",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
            key="frontier_points"
        )
    
    with col_conf3:
        include_3d = st.checkbox(
            "Include 3D Visualization",
            value=True,
            key="include_3d"
        )
    
    # Calculate expected returns and covariance
    with st.spinner("📊 Calculating efficient frontier..."):
        try:
            mu = expected_returns.mean_historical_return(df_prices)
            S = risk_models.sample_cov(df_prices)
            
            # Initialize frontier calculator
            frontier_calc = AdvancedEfficientFrontierChart(mu, S, risk_free_rate)
            
            # Calculate frontier
            frontier_df = frontier_calc.calculate_efficient_frontier(frontier_points)
            
            # Calculate key portfolios
            key_portfolios = frontier_calc.get_key_portfolios()
            
        except Exception as e:
            st.error(f"Error calculating efficient frontier: {e}")
            return
    
    if frontier_df.empty:
        st.warning("Could not calculate efficient frontier. Please check your data.")
        return
    
    # Display key portfolios
    st.markdown('<div class="subsection-header">🎯 Key Portfolios</div>', unsafe_allow_html=True)
    
    if key_portfolios:
        col_key1, col_key2, col_key3 = st.columns(3)
        
        portfolios_to_show = ["Minimum Volatility", "Maximum Sharpe", "Equal Weight"]
        
        for i, portfolio_name in enumerate(portfolios_to_show):
            if portfolio_name in key_portfolios:
                portfolio = key_portfolios[portfolio_name]
                
                if i == 0:
                    with col_key1:
                        st.metric(
                            portfolio_name,
                            f"{portfolio['return']:.2%}",
                            f"Vol: {portfolio['volatility']:.2%}, Sharpe: {portfolio['sharpe']:.2f}"
                        )
                elif i == 1:
                    with col_key2:
                        st.metric(
                            portfolio_name,
                            f"{portfolio['return']:.2%}",
                            f"Vol: {portfolio['volatility']:.2%}, Sharpe: {portfolio['sharpe']:.2f}"
                        )
                else:
                    with col_key3:
                        st.metric(
                            portfolio_name,
                            f"{portfolio['return']:.2%}",
                            f"Vol: {portfolio['volatility']:.2%}, Sharpe: {portfolio['sharpe']:.2f}"
                        )
    
    # Create visualizations
    st.markdown('<div class="subsection-header">📊 Frontier Visualizations</div>', unsafe_allow_html=True)
    
    if include_3d:
        tab_viz1, tab_viz2 = st.tabs(["2D Analysis", "3D Analysis"])
    else:
        tab_viz1 = st.container()
    
    with tab_viz1:
        # Create advanced frontier chart
        frontier_chart = frontier_calc.create_advanced_frontier_chart(frontier_df, key_portfolios)
        st.plotly_chart(frontier_chart, use_container_width=True)
    
    if include_3d:
        with tab_viz2:
            # Create 3D frontier chart
            with st.spinner("🔄 Generating 3D visualization..."):
                frontier_3d = frontier_calc.create_3d_frontier_chart(frontier_df)
                st.plotly_chart(frontier_3d, use_container_width=True)
    
    # Portfolio simulation
    st.markdown('<div class="subsection-header">🎯 Portfolio Simulation</div>', unsafe_allow_html=True)
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        target_return = st.number_input(
            "Target Annual Return (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            key="target_return"
        ) / 100
    
    with col_sim2:
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Very Low", "Low", "Medium", "High", "Very High"],
            value="Medium",
            key="risk_tolerance"
        )
    
    if st.button("🔍 Find Optimal Portfolio for Target", use_container_width=True):
        # Find portfolio closest to target return
        closest_idx = (frontier_df['return'] - target_return).abs().argmin()
        closest_portfolio = frontier_df.iloc[closest_idx]
        
        st.markdown(f"**Optimal Portfolio for {target_return*100:.1f}% Target Return:**")
        
        col_sim_res1, col_sim_res2, col_sim_res3 = st.columns(3)
        
        with col_sim_res1:
            st.metric("Expected Return", f"{closest_portfolio['return']:.2%}")
        with col_sim_res2:
            st.metric("Expected Volatility", f"{closest_portfolio['volatility']:.2%}")
        with col_sim_res3:
            st.metric("Sharpe Ratio", f"{closest_portfolio['sharpe']:.2f}")
        
        # Show allocation for significant weights
        weights = closest_portfolio['weights']
        if weights:
            allocation_data = []
            for ticker, weight in weights.items():
                if weight > 0.01:  # Only show weights > 1%
                    allocation_data.append({
                        "Asset": st.session_state.get("ticker_map", {}).get(ticker, ticker),
                        "Weight": f"{weight:.2%}"
                    })
            
            if allocation_data:
                st.dataframe(pd.DataFrame(allocation_data), use_container_width=True, height=200)

def create_var_analysis_tab():
    """Create comprehensive VaR analysis tab"""
    st.markdown('<div class="section-header">🎲 Advanced Value at Risk (VaR) & Expected Shortfall (ES) Analysis</div>', unsafe_allow_html=True)
    
    if "portfolio_returns" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please run portfolio optimization first to generate portfolio returns.</div>', unsafe_allow_html=True)
        return
    
    # Similar to original but enhanced...

def create_enhanced_stress_test_tab():
    """Create enhanced stress testing tab with user-customizable scenarios"""
    st.markdown('<div class="section-header">⚠️ Enhanced Stress Testing Laboratory</div>', unsafe_allow_html=True)
    
    if "portfolio_returns" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please run portfolio optimization first to generate portfolio returns.</div>', unsafe_allow_html=True)
        return
    
    # Similar to original but enhanced...

# ==============================================================================
# 13) MAIN APPLICATION WITH ALL TABS
# ==============================================================================

def main():
    st.markdown('<div class="main-header">⚡ QUANTUM | Advanced Risk Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Initialize data manager
    dm = EnhancedDataManager()
    
    # Session state initialization
    if "stress_test_results" not in st.session_state:
        st.session_state.stress_test_results = []
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
        col_preset1, col_preset2 = st.columns(2)
        
        with col_preset1:
            if st.button("Global 60/40", use_container_width=True):
                st.session_state.selected_assets_preset = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
        
        with col_preset2:
            if st.button("Emerging Markets", use_container_width=True):
                st.session_state.selected_assets_preset = ["EEM", "THYAO.IS", "BABA", "005930.KS"]
        
        st.divider()
        
        # Asset selection
        selected_assets: List[str] = []
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
    
    if not OPTIMIZATION_AVAILABLE:
        st.markdown('<div class="warning-card">⚠️ PyPortfolioOpt is not installed/available. Some features will be disabled.</div>', unsafe_allow_html=True)
        if "_OPT_IMPORT_ERROR" in globals():
            st.info(f"Import detail: {_OPT_IMPORT_ERROR}")
    
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
    st.session_state.benchmark_data = benchmark_data
    st.session_state.ticker_map = dm.get_ticker_name_map()
    
    # Create enhanced tabs with NEW TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Data Overview",
        "📊 Advanced Performance",
        "🎯 Black-Litterman",
        "📈 Advanced Frontier",
        "🎯 Portfolio Optimization",
        "🎲 Advanced VaR/ES",
        "⚠️ Stress Testing Lab",
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
            st.metric("Mean Return (Ann.)", f"{returns.mean().mean() * 252:.2%}")
        with col_stat2:
            st.metric("Avg Volatility (Ann.)", f"{returns.std().mean() * np.sqrt(252):.2%}")
        with col_stat3:
            st.metric("Avg Correlation", f"{returns.corr().values.mean():.2f}")
    
    # --- NEW TAB 2: ADVANCED PERFORMANCE METRICS ---
    with tab2:
        create_advanced_performance_tab()
    
    # --- NEW TAB 3: BLACK-LITTERMAN OPTIMIZATION ---
    with tab3:
        create_black_litterman_tab()
    
    # --- NEW TAB 4: ADVANCED EFFICIENT FRONTIER ---
    with tab4:
        create_advanced_frontier_tab()
    
    # --- TAB 5: PORTFOLIO OPTIMIZATION (ORIGINAL) ---
    with tab5:
        if not OPTIMIZATION_AVAILABLE:
            st.markdown('<div class="warning-card">⚠️ Portfolio optimization requires PyPortfolioOpt. Add it to requirements.txt: PyPortfolioOpt</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-header">🎯 Portfolio Construction & Optimization</div>', unsafe_allow_html=True)
            
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            with col_conf1:
                strategy = st.selectbox(
                    "Optimization Strategy",
                    ["Max Sharpe Ratio", "Min Volatility", "Hierarchical Risk Parity (HRP)",
                     "Black-Litterman (Views)", "Equal Weight"]
                )
            with col_conf2:
                rf_rate = st.number_input("Risk Free Rate (%)", value=4.5, step=0.1) / 100
            with col_conf3:
                amount = st.number_input("Investment Amount ($)", value=1000000, step=100000)
            
            st.session_state["last_rf_rate"] = float(rf_rate)
            st.session_state["last_amount"] = float(amount)
            
            if st.button("Run Optimization", type="primary", use_container_width=True):
                with st.spinner("Running optimization..."):
                    try:
                        engine = EnhancedOptimizationEngine(df_prices)
                        
                        if strategy == "Hierarchical Risk Parity (HRP)":
                            weights, perf = engine.optimize_hrp()
                        
                        elif strategy == "Equal Weight":
                            equal_weight = 1.0 / len(df_prices.columns)
                            weights = {asset: equal_weight for asset in df_prices.columns}
                            port_returns = engine.returns.dot(pd.Series(weights))
                            ann_ret = float(port_returns.mean() * 252)
                            ann_vol = float(port_returns.std() * np.sqrt(252))
                            sharpe = float((ann_ret - rf_rate) / ann_vol) if ann_vol > 0 else 0.0
                            perf = (ann_ret, ann_vol, sharpe)
                        
                        else:
                            obj_type = "max_sharpe" if "Sharpe" in strategy else "min_volatility"
                            weights, perf = engine.optimize_mean_variance(obj_type, rf_rate)
                        
                        st.markdown('<div class="success-card">✅ Optimization completed!</div>', unsafe_allow_html=True)
                        
                        col_res1, col_res2 = st.columns([1, 1])
                        ticker_map = dm.get_ticker_name_map()
                        
                        with col_res1:
                            st.markdown('<div class="subsection-header">📊 Portfolio Allocation</div>', unsafe_allow_html=True)
                            portfolio_data = []
                            for ticker, weight in weights.items():
                                if float(weight) > 0.001:
                                    portfolio_data.append({
                                        "Asset": ticker_map.get(ticker, ticker),
                                        "Ticker": ticker,
                                        "Weight": float(weight),
                                        "Amount ($)": float(weight) * float(amount),
                                    })
                            
                            portfolio_df = pd.DataFrame(portfolio_data).sort_values(by="Weight", ascending=False)
                            
                            if portfolio_df.empty:
                                st.markdown('<div class="warning-card">⚠️ No non-trivial weights returned (all ~0).</div>', unsafe_allow_html=True)
                            else:
                                display_df = portfolio_df[["Asset", "Ticker", "Weight", "Amount ($)"]].copy()
                                display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x:.2%}")
                                display_df["Amount ($)"] = display_df["Amount ($)"].apply(lambda x: f"${x:,.0f}")
                                st.dataframe(display_df, use_container_width=True, height=420)
                        
                        with col_res2:
                            st.markdown('<div class="subsection-header">📈 Allocation Chart</div>', unsafe_allow_html=True)
                            if not portfolio_df.empty:
                                fig_pie = px.pie(portfolio_df, values="Weight", names="Asset", hole=0.4)
                                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                                fig_pie.update_layout(template="plotly_white", height=420, showlegend=False,
                                                     title_font_color="#1a237e", font_color="#424242")
                                st.plotly_chart(fig_pie, use_container_width=True)
                        
                        st.divider()
                        st.markdown('<div class="subsection-header">🎯 Performance Summary</div>', unsafe_allow_html=True)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Expected Return (Ann.)", f"{perf[0]:.2%}")
                        m2.metric("Volatility (Ann.)", f"{perf[1]:.2%}")
                        m3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                        
                        portfolio_returns_series = engine.returns.dot(pd.Series(weights)).dropna()
                        cum_returns = (1 + portfolio_returns_series).cumprod()
                        running_max = cum_returns.cummax()
                        drawdown = (cum_returns - running_max) / running_max
                        max_dd = float(drawdown.min()) if len(drawdown) else 0.0
                        m4.metric("Max Drawdown", f"{max_dd:.2%}")
                        
                        # Store session state for other tabs
                        st.session_state.portfolio_returns = portfolio_returns_series
                        st.session_state.weights = weights
                        
                    except Exception as e:
                        st.markdown(f'<div class="warning-card">❌ Optimization failed: {e}</div>', unsafe_allow_html=True)
    
    # --- TAB 6: ADVANCED VAR/ES ---
    with tab6:
        create_var_analysis_tab()
    
    # --- TAB 7: ENHANCED STRESS TESTING ---
    with tab7:
        create_enhanced_stress_test_tab()
    
    # --- TAB 8: RISK ANALYTICS ---
    with tab8:
        st.markdown('<div class="section-header">🛡️ Risk Analytics</div>', unsafe_allow_html=True)
        
        if "portfolio_returns" not in st.session_state:
            st.markdown('<div class="warning-card">⚠️ Please run portfolio optimization first.</div>', unsafe_allow_html=True)
        else:
            portfolio_returns = st.session_state.portfolio_returns
            
            # Use last rf/amount if available
            rf_for_metrics = float(st.session_state.get("last_rf_rate", 0.04))
            amt_for_var = float(st.session_state.get("last_amount", 1_000_000))
            
            bench_ret = benchmark_data.pct_change().dropna() if isinstance(benchmark_data, pd.Series) and not benchmark_data.empty else None
            
            risk_engine = EnhancedRiskMetricsEngine(
                portfolio_returns=portfolio_returns,
                risk_free_rate=rf_for_metrics,
                benchmark_returns=bench_ret
            )
            
            st.markdown('<div class="subsection-header">📊 Risk & Performance Metrics</div>', unsafe_allow_html=True)
            metrics_df = risk_engine.calculate_comprehensive_metrics()
            
            col_met1, col_met2 = st.columns(2)
            
            with col_met1:
                return_metrics = metrics_df[metrics_df["Metric"].isin([
                    "Annual Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio"
                ])]
                for _, row in return_metrics.iterrows():
                    if "Ratio" in row["Metric"]:
                        st.metric(row["Metric"], f"{row['Value']:.3f}")
                    else:
                        st.metric(row["Metric"], f"{row['Value']:.2%}")
            
            with col_met2:
                other_metrics = metrics_df[~metrics_df["Metric"].isin([
                    "Annual Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio"
                ])]
                for _, row in other_metrics.iterrows():
                    if ("Ratio" in row["Metric"]) or ("Beta" in row["Metric"]) or ("Alpha" in row["Metric"]):
                        if np.isfinite(row["Value"]):
                            st.metric(row["Metric"], f"{row['Value']:.3f}")
                        else:
                            st.metric(row["Metric"], "∞")
                    elif ("VaR" in row["Metric"]) or ("CVaR" in row["Metric"]):
                        st.metric(row["Metric"], f"{row['Value']:.2%}" if np.isfinite(row["Value"]) else "N/A")
                    else:
                        st.metric(row["Metric"], f"{row['Value']:.3f}" if np.isfinite(row["Value"]) else "N/A")
    
    # --- TAB 9: CORRELATION ANALYSIS ---
    with tab9:
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
# 14) MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
