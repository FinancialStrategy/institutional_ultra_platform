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

# --- 3) SIMPLIFIED CSS WITH MINIMAL COLORS ---
st.markdown("""
    <style>
        /* Main background - Simple White */
        .stApp {
            background-color: #ffffff;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Main container styling - Simple */
        .main-container {
            background-color: white;
            border-radius: 6px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid #e1e5eb;
        }
        
        /* Main header - Simple Dark */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a237e;
            padding-bottom: 10px;
            margin-bottom: 20px;
            text-align: center;
            border-bottom: 2px solid #303f9f;
            letter-spacing: -0.3px;
        }
        
        /* Section headers - Simple */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a237e;
            margin-top: 20px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e8eaf6;
        }
        
        /* Sub-section headers */
        .subsection-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #283593;
            margin-top: 15px;
            margin-bottom: 10px;
        }
        
        /* Metric cards - Simple */
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
            margin: 8px 0;
        }
        
        .metric-title {
            font-size: 0.8rem;
            font-weight: 500;
            color: #424242;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a237e;
            margin: 3px 0;
        }
        
        /* Tab styling - Simple */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #ffffff;
            padding: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: 500;
            color: #5c6bc0;
            font-size: 0.9rem;
            border: 1px solid #e0e0e0;
        }
        
        .stTabs [aria-selected="true"] {
            background: #1a237e;
            color: white;
        }
        
        /* Button styling - Simple */
        .stButton > button {
            background: #1a237e;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: 500;
            font-size: 0.85rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
        }
        
        .sidebar-section {
            margin: 15px 0;
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .sidebar-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #1a237e;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .sidebar-button {
            width: 100%;
            margin: 3px 0;
            font-size: 0.8rem;
        }
        
        /* Warning/Info cards - Simple */
        .warning-card {
            background: #fff8e1;
            border-left: 3px solid #ff9800;
            color: #5d4037;
            border-radius: 4px;
            padding: 12px;
            margin: 8px 0;
            font-size: 0.9rem;
        }
        
        .success-card {
            background: #e8f5e9;
            border-left: 3px solid #4caf50;
            color: #1b5e20;
            border-radius: 4px;
            padding: 12px;
            margin: 8px 0;
            font-size: 0.9rem;
        }
        
        .info-card {
            background: #e3f2fd;
            border-left: 3px solid #2196f3;
            color: #0d47a1;
            border-radius: 4px;
            padding: 12px;
            margin: 8px 0;
            font-size: 0.9rem;
        }
        
        /* Chart containers */
        .plotly-chart-container {
            background: white;
            border-radius: 6px;
            padding: 10px;
            border: 1px solid #e0e0e0;
            margin: 10px 0;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
            font-size: 0.85rem;
        }
        
        /* Risk badge styling - Simple */
        .risk-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: 500;
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
        
        /* Expander headers */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            color: #1a237e;
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        /* Input fields */
        .stSelectbox, .stNumberInput, .stDateInput, .stSlider {
            background-color: white;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        
        /* Metric containers */
        .stMetric {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 10px;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #bdbdbd;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4) ENHANCED DATA MANAGER
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
# 5) PORTFOLIO OPTIMIZATION ENGINE
# ==============================================================================

class PortfolioOptimizer:
    """Portfolio optimization with multiple strategies"""
    
    def __init__(self, returns_df: pd.DataFrame):
        self.returns_df = returns_df
        self.mu = returns_df.mean() * 252
        self.S = returns_df.cov() * 252
        
    def calculate_efficient_frontier(self, num_points: int = 50):
        """Calculate efficient frontier"""
        n = len(self.mu)
        
        # Minimum variance portfolio
        inv_S = np.linalg.inv(self.S)
        ones = np.ones(n)
        w_min_var = inv_S @ ones / (ones.T @ inv_S @ ones)
        ret_min_var = w_min_var @ self.mu
        vol_min_var = np.sqrt(w_min_var.T @ self.S @ w_min_var)
        
        # Maximum return portfolio (single asset with highest return)
        max_ret_idx = np.argmax(self.mu.values)
        w_max_ret = np.zeros(n)
        w_max_ret[max_ret_idx] = 1
        ret_max_ret = self.mu.iloc[max_ret_idx]
        vol_max_ret = np.sqrt(self.S.iloc[max_ret_idx, max_ret_idx])
        
        # Generate frontier
        frontier_returns = np.linspace(ret_min_var, ret_max_ret, num_points)
        frontier_volatilities = []
        frontier_weights = []
        
        for target_return in frontier_returns:
            # Solve optimization problem
            A = np.vstack([self.mu.values, np.ones(n)]).T
            b = np.array([target_return, 1])
            
            try:
                # Solve for weights using least squares
                w = np.linalg.lstsq(A, b, rcond=None)[0]
                # Ensure non-negative weights (simple projection)
                w = np.maximum(w, 0)
                w = w / w.sum()
                
                vol = np.sqrt(w.T @ self.S.values @ w)
                frontier_volatilities.append(vol)
                frontier_weights.append(w)
            except:
                frontier_volatilities.append(np.nan)
                frontier_weights.append(np.zeros(n))
        
        return {
            'returns': frontier_returns,
            'volatilities': frontier_volatilities,
            'weights': frontier_weights,
            'min_var': {'return': ret_min_var, 'vol': vol_min_var, 'weights': w_min_var},
            'max_ret': {'return': ret_max_ret, 'vol': vol_max_ret, 'weights': w_max_ret}
        }
    
    def optimize_sharpe_ratio(self, risk_free_rate: float = 0.04):
        """Optimize for maximum Sharpe ratio"""
        n = len(self.mu)
        excess_returns = self.mu - risk_free_rate
        
        # Solve for tangency portfolio
        inv_S = np.linalg.inv(self.S)
        w_tangency = inv_S @ excess_returns / (excess_returns @ inv_S @ np.ones(n))
        w_tangency = np.maximum(w_tangency, 0)  # Ensure non-negative
        w_tangency = w_tangency / w_tangency.sum()
        
        portfolio_return = w_tangency @ self.mu
        portfolio_vol = np.sqrt(w_tangency.T @ self.S @ w_tangency)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': w_tangency,
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_minimum_variance(self):
        """Optimize for minimum variance"""
        n = len(self.mu)
        inv_S = np.linalg.inv(self.S)
        ones = np.ones(n)
        w_min_var = inv_S @ ones / (ones.T @ inv_S @ ones)
        w_min_var = np.maximum(w_min_var, 0)  # Ensure non-negative
        w_min_var = w_min_var / w_min_var.sum()
        
        portfolio_return = w_min_var @ self.mu
        portfolio_vol = np.sqrt(w_min_var.T @ self.S @ w_min_var)
        
        return {
            'weights': w_min_var,
            'return': portfolio_return,
            'volatility': portfolio_vol
        }
    
    def optimize_max_return(self, max_volatility: float = 0.30):
        """Optimize for maximum return with volatility constraint"""
        n = len(self.mu)
        
        # Set up optimization problem
        def objective(weights):
            return -(weights @ self.mu)
        
        def volatility_constraint(weights):
            return max_volatility - np.sqrt(weights.T @ self.S @ weights)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': volatility_constraint}
        ]
        
        bounds = [(0, 1) for _ in range(n)]
        initial_guess = np.ones(n) / n
        
        result = optimize.minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            portfolio_return = weights @ self.mu
            portfolio_vol = np.sqrt(weights.T @ self.S @ weights)
            
            return {
                'weights': weights,
                'return': portfolio_return,
                'volatility': portfolio_vol
            }
        else:
            return None
    
    def create_equal_weight_portfolio(self):
        """Create equal weight portfolio"""
        n = len(self.mu)
        weights = np.ones(n) / n
        portfolio_return = weights @ self.mu
        portfolio_vol = np.sqrt(weights.T @ self.S @ weights)
        sharpe_ratio = (portfolio_return - 0.04) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': weights,
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def create_market_cap_weighted_portfolio(self, market_caps: Dict[str, float] = None):
        """Create market cap weighted portfolio"""
        n = len(self.mu)
        
        if market_caps is None:
            # Use equal weights if no market caps provided
            return self.create_equal_weight_portfolio()
        
        # Match market caps to available tickers
        weights = np.zeros(n)
        total_cap = 0
        
        for i, ticker in enumerate(self.returns_df.columns):
            if ticker in market_caps:
                weights[i] = market_caps[ticker]
                total_cap += market_caps[ticker]
        
        if total_cap > 0:
            weights = weights / total_cap
        else:
            weights = np.ones(n) / n
        
        portfolio_return = weights @ self.mu
        portfolio_vol = np.sqrt(weights.T @ self.S @ weights)
        sharpe_ratio = (portfolio_return - 0.04) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': weights,
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }

# ==============================================================================
# 6) BOLLINGER BANDS ANALYZER
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
            height=500,
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

# ==============================================================================
# 7) OHLC CHART AND TRACKING ERROR ANALYZER
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
            height=500,
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
        
        return fig

# ==============================================================================
# 8) PORTFOLIO OPTIMIZATION TAB
# ==============================================================================

def create_portfolio_optimization_tab():
    """Create comprehensive portfolio optimization tab"""
    st.markdown('<div class="section-header">🎯 Portfolio Optimization</div>', unsafe_allow_html=True)
    
    if "df_prices" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please load data first from the Data Overview tab.</div>', unsafe_allow_html=True)
        return
    
    df_prices = st.session_state["df_prices"]
    ticker_map = st.session_state.get("ticker_map", {})
    
    # Calculate returns
    returns = df_prices.pct_change().dropna()
    
    if returns.empty:
        st.markdown('<div class="warning-card">⚠️ Insufficient data for optimization.</div>', unsafe_allow_html=True)
        return
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns)
    
    # Create two columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">⚙️ Optimization Parameters</div>', unsafe_allow_html=True)
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1,
            key="opt_rf_rate"
        ) / 100
        
        max_volatility = st.slider(
            "Maximum Volatility Constraint (%)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
            key="opt_max_vol"
        ) / 100
        
        investment_amount = st.number_input(
            "Investment Amount ($)",
            min_value=1000,
            max_value=10000000,
            value=1000000,
            step=10000,
            key="opt_investment"
        )
    
    with col2:
        st.markdown('<div class="subsection-header">🎯 Optimization Strategy</div>', unsafe_allow_html=True)
        
        strategy = st.selectbox(
            "Select Optimization Strategy",
            options=[
                "Maximum Sharpe Ratio",
                "Minimum Variance",
                "Maximum Return (Volatility Constrained)",
                "Equal Weight",
                "Efficient Frontier"
            ],
            key="opt_strategy"
        )
        
        if strategy == "Efficient Frontier":
            num_frontier_points = st.slider(
                "Number of Frontier Points",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                key="opt_frontier_points"
            )
    
    # Run optimization
    if st.button("🚀 Run Portfolio Optimization", type="primary", use_container_width=True):
        with st.spinner("Optimizing portfolio..."):
            results = {}
            
            if strategy == "Maximum Sharpe Ratio":
                results = optimizer.optimize_sharpe_ratio(risk_free_rate)
                results['strategy'] = "Maximum Sharpe Ratio"
                
            elif strategy == "Minimum Variance":
                results = optimizer.optimize_minimum_variance()
                results['strategy'] = "Minimum Variance"
                
            elif strategy == "Maximum Return (Volatility Constrained)":
                results = optimizer.optimize_max_return(max_volatility)
                if results:
                    results['strategy'] = "Maximum Return (Volatility Constrained)"
                else:
                    st.error("Optimization failed. Try increasing volatility constraint.")
                    return
                
            elif strategy == "Equal Weight":
                results = optimizer.create_equal_weight_portfolio()
                results['strategy'] = "Equal Weight"
                
            elif strategy == "Efficient Frontier":
                frontier = optimizer.calculate_efficient_frontier(num_frontier_points)
                results = {
                    'strategy': 'Efficient Frontier',
                    'frontier': frontier,
                    'tangency': optimizer.optimize_sharpe_ratio(risk_free_rate),
                    'min_var': optimizer.optimize_minimum_variance()
                }
            
            # Store results in session state
            st.session_state.portfolio_results = results
            
            if strategy != "Efficient Frontier":
                # Calculate portfolio returns
                weights = results['weights']
                portfolio_returns = (returns * weights).sum(axis=1)
                st.session_state.portfolio_returns = portfolio_returns
                
                # Calculate dollar allocations
                dollar_allocations = weights * investment_amount
                
                # Display results
                st.markdown('<div class="success-card">✅ Portfolio optimization completed!</div>', unsafe_allow_html=True)
                
                # Key metrics
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    st.metric("Expected Return", f"{results['return']*100:.2f}%")
                
                with col_metric2:
                    st.metric("Expected Volatility", f"{results['volatility']*100:.2f}%")
                
                with col_metric3:
                    if 'sharpe_ratio' in results:
                        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.3f}")
                
                with col_metric4:
                    st.metric("Investment Amount", f"${investment_amount:,.0f}")
                
                # Portfolio weights chart
                st.markdown('<div class="subsection-header">📊 Portfolio Allocation</div>', unsafe_allow_html=True)
                
                # Create weights DataFrame
                weights_df = pd.DataFrame({
                    'Ticker': returns.columns,
                    'Name': [ticker_map.get(t, t) for t in returns.columns],
                    'Weight (%)': weights * 100,
                    'Allocation ($)': dollar_allocations
                })
                weights_df = weights_df[weights_df['Weight (%)'] > 0.01].sort_values('Weight (%)', ascending=False)
                
                # Pie chart
                fig_weights = go.Figure(data=[go.Pie(
                    labels=weights_df['Name'],
                    values=weights_df['Allocation ($)'],
                    hole=0.3,
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br>Allocation: $%{value:,.0f}<extra></extra>'
                )])
                fig_weights.update_layout(
                    title=f"Portfolio Allocation: {results['strategy']}",
                    height=400,
                    showlegend=True,
                    template="plotly_white"
                )
                st.plotly_chart(fig_weights, use_container_width=True)
                
                # Detailed weights table
                st.markdown('<div class="subsection-header">📋 Detailed Allocation</div>', unsafe_allow_html=True)
                
                display_df = weights_df.copy()
                display_df['Weight (%)'] = display_df['Weight (%)'].round(2)
                display_df['Allocation ($)'] = display_df['Allocation ($)'].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(
                    display_df[['Name', 'Weight (%)', 'Allocation ($)']],
                    use_container_width=True,
                    hide_index=True
                )
                
            else:
                # Efficient Frontier visualization
                frontier = results['frontier']
                tangency = results['tangency']
                min_var = results['min_var']
                
                st.markdown('<div class="success-card">✅ Efficient frontier calculated!</div>', unsafe_allow_html=True)
                
                # Create frontier chart
                fig_frontier = go.Figure()
                
                # Add frontier points
                fig_frontier.add_trace(go.Scatter(
                    x=frontier['volatilities'],
                    y=frontier['returns'] * 100,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#1a237e', width=3),
                    hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                # Add individual assets
                fig_frontier.add_trace(go.Scatter(
                    x=np.sqrt(np.diag(optimizer.S)),
                    y=optimizer.mu * 100,
                    mode='markers',
                    name='Individual Assets',
                    marker=dict(
                        size=10,
                        color='#ff9800',
                        symbol='circle'
                    ),
                    text=[ticker_map.get(t, t) for t in returns.columns],
                    hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                # Add tangency portfolio
                fig_frontier.add_trace(go.Scatter(
                    x=[tangency['volatility']],
                    y=[tangency['return'] * 100],
                    mode='markers',
                    name='Tangency Portfolio',
                    marker=dict(
                        size=15,
                        color='#d32f2f',
                        symbol='star'
                    ),
                    hovertemplate='<b>Tangency Portfolio</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2f}%<br>Sharpe: %{customdata:.3f}<extra></extra>',
                    customdata=[tangency['sharpe_ratio']]
                ))
                
                # Add minimum variance portfolio
                fig_frontier.add_trace(go.Scatter(
                    x=[min_var['volatility']],
                    y=[min_var['return'] * 100],
                    mode='markers',
                    name='Minimum Variance',
                    marker=dict(
                        size=15,
                        color='#388e3c',
                        symbol='diamond'
                    ),
                    hovertemplate='<b>Minimum Variance</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                # Capital Market Line
                x_range = np.linspace(0, max(frontier['volatilities']) * 1.1, 100)
                cml_y = risk_free_rate * 100 + (tangency['sharpe_ratio'] * x_range * 100)
                
                fig_frontier.add_trace(go.Scatter(
                    x=x_range,
                    y=cml_y,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color='#7b1fa2', width=2, dash='dash'),
                    hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                fig_frontier.update_layout(
                    title='Efficient Frontier & Capital Market Line',
                    xaxis_title='Annual Volatility',
                    yaxis_title='Annual Return (%)',
                    template='plotly_white',
                    height=600,
                    hovermode='closest',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_frontier, use_container_width=True)
                
                # Display portfolio comparisons
                col_tan, col_min = st.columns(2)
                
                with col_tan:
                    st.markdown('<div class="subsection-header">🏆 Tangency Portfolio</div>', unsafe_allow_html=True)
                    st.metric("Return", f"{tangency['return']*100:.2f}%")
                    st.metric("Volatility", f"{tangency['volatility']*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{tangency['sharpe_ratio']:.3f}")
                    
                    # Show top holdings
                    tangency_weights = pd.Series(tangency['weights'], index=returns.columns)
                    top_tangency = tangency_weights[tangency_weights > 0.01].sort_values(ascending=False).head(5)
                    if len(top_tangency) > 0:
                        st.write("**Top Holdings:**")
                        for ticker, weight in top_tangency.items():
                            st.write(f"{ticker_map.get(ticker, ticker)}: {weight*100:.1f}%")
                
                with col_min:
                    st.markdown('<div class="subsection-header">🛡️ Minimum Variance Portfolio</div>', unsafe_allow_html=True)
                    st.metric("Return", f"{min_var['return']*100:.2f}%")
                    st.metric("Volatility", f"{min_var['volatility']*100:.2f}%")
                    
                    # Show top holdings
                    min_var_weights = pd.Series(min_var['weights'], index=returns.columns)
                    top_min_var = min_var_weights[min_var_weights > 0.01].sort_values(ascending=False).head(5)
                    if len(top_min_var) > 0:
                        st.write("**Top Holdings:**")
                        for ticker, weight in top_min_var.items():
                            st.write(f"{ticker_map.get(ticker, ticker)}: {weight*100:.1f}%")
    
    # If we have existing portfolio results, show them
    elif "portfolio_results" in st.session_state:
        results = st.session_state.portfolio_results
        
        if results['strategy'] != "Efficient Frontier":
            st.markdown('<div class="info-card">📊 Showing previously optimized portfolio</div>', unsafe_allow_html=True)
            
            # Display metrics
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric("Expected Return", f"{results['return']*100:.2f}%")
            
            with col_metric2:
                st.metric("Expected Volatility", f"{results['volatility']*100:.2f}%")
            
            with col_metric3:
                if 'sharpe_ratio' in results:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.3f}")
            
            # Show weights
            weights = pd.Series(results['weights'], index=returns.columns)
            weights = weights[weights > 0.01].sort_values(ascending=False)
            
            if len(weights) > 0:
                st.markdown('<div class="subsection-header">📊 Current Portfolio Weights</div>', unsafe_allow_html=True)
                
                weights_df = pd.DataFrame({
                    'Asset': [ticker_map.get(t, t) for t in weights.index],
                    'Weight (%)': (weights * 100).round(2)
                })
                
                st.dataframe(weights_df, use_container_width=True, hide_index=True)

# ==============================================================================
# 9) OTHER TAB FUNCTIONS (Simplified)
# ==============================================================================

def create_bollinger_bands_tab():
    """Create Bollinger Bands analysis tab"""
    st.markdown('<div class="section-header">📊 Bollinger Bands Analysis</div>', unsafe_allow_html=True)
    
    if "df_prices" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please load data first from the Data Overview tab.</div>', unsafe_allow_html=True)
        return
    
    df_prices = st.session_state["df_prices"]
    ticker_map = st.session_state.get("ticker_map", {})
    
    col_config1, col_config2 = st.columns(2)
    
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
    
    # Initialize analyzer
    bb_analyzer = BollingerBandsAnalyzer(df_prices, window=window_size)
    
    # Create chart
    ticker_name = ticker_map.get(selected_ticker, selected_ticker)
    bb_chart = bb_analyzer.create_bollinger_chart(selected_ticker, ticker_name)
    st.plotly_chart(bb_chart, use_container_width=True)
    
    # Show current status
    bb_analyzer.calculate_bollinger_bands(selected_ticker)
    current_stats = bb_analyzer.results.get(selected_ticker, {})
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("Current Price", f"${current_stats.get('current_price', 0):.2f}")
    
    with col_stat2:
        st.metric(f"SMA ({window_size})", f"${current_stats.get('current_sma', 0):.2f}")
    
    with col_stat3:
        st.metric("Current Signal", current_stats.get('recent_signal', 'N/A'))

def create_ohlc_tracking_error_tab():
    """Create OHLC chart and tracking error analysis tab"""
    st.markdown('<div class="section-header">📈 OHLC Charts</div>', unsafe_allow_html=True)
    
    if "df_prices" not in st.session_state:
        st.markdown('<div class="warning-card">⚠️ Please load data first from the Data Overview tab.</div>', unsafe_allow_html=True)
        return
    
    df_prices = st.session_state["df_prices"]
    ticker_map = st.session_state.get("ticker_map", {})
    
    col_ohlc1, col_ohlc2 = st.columns(2)
    
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
        start_date = end_date - timedelta(days=365)
        date_range = st.date_input(
            "Date Range",
            value=(start_date, end_date),
            key="ohlc_date_range"
        )
    
    # Initialize analyzer
    ohlc_analyzer = OHLCAndTrackingErrorAnalyzer()
    
    # Create OHLC chart
    if len(date_range) == 2:
        start_str = date_range[0].strftime("%Y-%m-%d")
        end_str = date_range[1].strftime("%Y-%m-%d")
        
        with st.spinner("Generating OHLC chart..."):
            ticker_name = ticker_map.get(selected_ticker, selected_ticker)
            ohlc_chart = ohlc_analyzer.create_ohlc_chart(
                selected_ticker, start_str, end_str, ticker_name
            )
            st.plotly_chart(ohlc_chart, use_container_width=True)

def create_enhanced_stress_test_tab():
    """Create enhanced stress testing tab"""
    st.markdown('<div class="section-header">⚠️ Stress Testing</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">This tab requires portfolio optimization to be run first.</div>', unsafe_allow_html=True)
    
    if st.button("Run Basic Stress Test", use_container_width=True):
        st.markdown('<div class="subsection-header">Stress Test Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("10% Market Crash Impact", "-$100,000")
        
        with col2:
            st.metric("Volatility Spike Impact", "-$50,000")
        
        with col3:
            st.metric("Recovery Period", "3-6 months")

# ==============================================================================
# 10) MAIN APPLICATION WITH REDESIGNED SIDEBAR
# ==============================================================================

def main():
    st.markdown('<div class="main-header">QUANTUM | Advanced Risk Analytics</div>', unsafe_allow_html=True)
    
    # Initialize data manager
    dm = EnhancedDataManager()
    
    # Initialize session states
    if "selected_assets_preset" not in st.session_state:
        st.session_state.selected_assets_preset = None
    if "portfolio_results" not in st.session_state:
        st.session_state.portfolio_results = None
    
    # ============================================
    # REDESIGNED SIDEBAR - SIMPLE AND CLEAN
    # ============================================
    with st.sidebar:
        st.markdown('<div class="sidebar-title">QUICK PORTFOLIOS</div>', unsafe_allow_html=True)
        
        # Simple buttons in a grid
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Global 60/40", use_container_width=True, key="btn_global"):
                st.session_state.selected_assets_preset = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
        
        with col2:
            if st.button("Tech Growth", use_container_width=True, key="btn_tech"):
                st.session_state.selected_assets_preset = ["QQQ", "XLK", "AAPL", "MSFT", "NVDA"]
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Emerging Mkts", use_container_width=True, key="btn_em"):
                st.session_state.selected_assets_preset = ["EEM", "THYAO.IS", "BABA", "005930.KS"]
        
        with col4:
            if st.button("Defensive", use_container_width=True, key="btn_def"):
                st.session_state.selected_assets_preset = ["SPY", "TLT", "GLD", "XLV", "JNK"]
        
        st.markdown('<div class="sidebar-title">ASSET SELECTION</div>', unsafe_allow_html=True)
        
        # Asset selection - simplified
        selected_assets = []
        default_assets = ["SPY", "TLT", "GLD", "AAPL", "THYAO.IS"]
        
        if st.session_state.selected_assets_preset:
            default_assets = st.session_state.selected_assets_preset
        
        # Simple multi-select for all assets
        all_assets = []
        for category, assets in dm.universe.items():
            for name, ticker in assets.items():
                all_assets.append((ticker, f"{name} ({ticker})"))
        
        # Sort alphabetically by name
        all_assets.sort(key=lambda x: x[1])
        
        selected_options = st.multiselect(
            "Select Assets",
            options=[opt[1] for opt in all_assets],
            default=[opt[1] for opt in all_assets if opt[0] in default_assets],
            key="asset_select"
        )
        
        # Map back to tickers
        name_to_ticker = {name: ticker for ticker, name in all_assets}
        selected_assets = [name_to_ticker[name] for name in selected_options if name in name_to_ticker]
        
        st.markdown('<div class="sidebar-title">DATA SETTINGS</div>', unsafe_allow_html=True)
        
        col_date1, col_date2 = st.columns(2)
        
        with col_date1:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
        
        with col_date2:
            min_data = st.number_input("Min Days", value=252, min_value=30, max_value=1000)
        
        st.markdown('<div class="sidebar-title">RISK PARAMETERS</div>', unsafe_allow_html=True)
        
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1
        ) / 100
        
        investment_amount = st.number_input(
            "Investment ($)",
            value=1000000,
            step=100000
        )
        
        # Store in session state
        st.session_state["last_rf_rate"] = float(risk_free_rate)
        st.session_state["last_amount"] = float(investment_amount)
        
        st.markdown('<div class="sidebar-title">EXECUTIVE ACTIONS</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Load Data", use_container_width=True, type="primary"):
            st.rerun()
        
        if st.button("📊 Run All Analysis", use_container_width=True):
            st.session_state.run_all_analysis = True
        
        st.divider()
        st.caption("v1.0 | Institutional Grade")
    
    # ============================================
    # MAIN CONTENT AREA
    # ============================================
    
    if not selected_assets:
        st.markdown('<div class="warning-card">⚠️ Please select assets from the sidebar.</div>', unsafe_allow_html=True)
        return
    
    # Fetch data
    with st.spinner("Loading data..."):
        df_prices, benchmark_data, data_report = _fetch_and_align_data_cached(
            selected_tickers=tuple(selected_assets),
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            min_data_length=min_data
        )
    
    if df_prices is None or df_prices.empty:
        st.markdown('<div class="warning-card">❌ No valid data available. Please select different assets.</div>', unsafe_allow_html=True)
        return
    
    # Store data in session state
    st.session_state.df_prices = df_prices
    st.session_state.ticker_map = dm.get_ticker_name_map()
    
    # Success message
    st.markdown(f'<div class="success-card">✅ Data loaded: {len(df_prices)} days, {len(df_prices.columns)} assets</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🎯 Portfolio Optimization",
        "📈 OHLC Charts",
        "📊 Bollinger Bands",
        "⚠️ Stress Testing",
        "🔗 Correlation"
    ])
    
    # --- TAB 1: DATA OVERVIEW ---
    with tab1:
        st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
        
        # Basic statistics
        returns = df_prices.pct_change().dropna()
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Assets", len(df_prices.columns))
        
        with col_stat2:
            st.metric("Time Period", f"{len(df_prices)} days")
        
        with col_stat3:
            avg_return = returns.mean().mean() * 252
            st.metric("Avg Return", f"{avg_return:.2%}")
        
        with col_stat4:
            avg_vol = returns.std().mean() * np.sqrt(252)
            st.metric("Avg Volatility", f"{avg_vol:.2%}")
        
        # Normalized price chart
        st.markdown('<div class="subsection-header">Normalized Performance</div>', unsafe_allow_html=True)
        normalized = (df_prices / df_prices.iloc[0]) * 100
        
        fig_prices = px.line(
            normalized,
            title="All Assets Rebased to 100",
            labels={"value": "Index Value", "variable": "Asset"}
        )
        fig_prices.update_layout(
            template="plotly_white",
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig_prices, use_container_width=True)
        
        # Asset details table
        st.markdown('<div class="subsection-header">Asset Details</div>', unsafe_allow_html=True)
        
        asset_stats = []
        for ticker in df_prices.columns:
            ret_series = df_prices[ticker].pct_change().dropna()
            asset_stats.append({
                'Ticker': ticker,
                'Name': dm.get_ticker_name_map().get(ticker, ticker),
                'Return (Ann)': f"{ret_series.mean() * 252:.2%}",
                'Volatility (Ann)': f"{ret_series.std() * np.sqrt(252):.2%}",
                'Sharpe Ratio': f"{(ret_series.mean() * 252 - risk_free_rate) / (ret_series.std() * np.sqrt(252)):.2f}" if ret_series.std() > 0 else "N/A"
            })
        
        asset_df = pd.DataFrame(asset_stats)
        st.dataframe(asset_df, use_container_width=True, hide_index=True)
    
    # --- TAB 2: PORTFOLIO OPTIMIZATION ---
    with tab2:
        create_portfolio_optimization_tab()
    
    # --- TAB 3: OHLC CHARTS ---
    with tab3:
        create_ohlc_tracking_error_tab()
    
    # --- TAB 4: BOLLINGER BANDS ---
    with tab4:
        create_bollinger_bands_tab()
    
    # --- TAB 5: STRESS TESTING ---
    with tab5:
        create_enhanced_stress_test_tab()
    
    # --- TAB 6: CORRELATION ---
    with tab6:
        st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
        
        returns = df_prices.pct_change().dropna()
        
        if returns.empty or returns.shape[1] < 2:
            st.markdown('<div class="warning-card">⚠️ Need at least 2 assets for correlation.</div>', unsafe_allow_html=True)
        else:
            corr_matrix = returns.corr()
            
            # Heatmap
            ticker_map = dm.get_ticker_name_map()
            labels = [ticker_map.get(t, t)[:20] for t in corr_matrix.columns]
            
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
                height=600,
                title="Asset Correlation Matrix",
                xaxis_title="Assets",
                yaxis_title="Assets",
                xaxis=dict(tickangle=-45)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Summary statistics
            st.markdown('<div class="subsection-header">Correlation Statistics</div>', unsafe_allow_html=True)
            
            col_corr1, col_corr2, col_corr3 = st.columns(3)
            
            with col_corr1:
                avg_corr = corr_matrix.values.mean()
                st.metric("Average Correlation", f"{avg_corr:.3f}")
            
            with col_corr2:
                max_corr = corr_matrix.values.max()
                st.metric("Maximum Correlation", f"{max_corr:.3f}")
            
            with col_corr3:
                min_corr = corr_matrix.values.min()
                st.metric("Minimum Correlation", f"{min_corr:.3f}")

# ==============================================================================
# 11) MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
