# ==============================================================================
# QUANTUM | Global Institutional Terminal - INSTITUTIONAL ENHANCED (Streamlit)
# ------------------------------------------------------------------------------
# Features (NO FEATURE REMOVALS; only additions/enhancements):
#  - Robust multi-asset data manager (50+ universe) with alignment & NaN-free series
#  - Portfolio Optimization: Equal Weight, Mean-Variance (MV), HRP, Black-Litterman (BL)
#  - Advanced VaR/CVaR/ES: Historical, Parametric, Monte-Carlo, EVT + visuals + surface
#  - Stress Testing: Historical crises + custom scenarios + Monte-Carlo stress sim
#  - Correlation & Risk: correlation heatmaps, rolling vol, drawdown analytics
#  - Rolling Beta: line + HEATMAP (assets × time buckets)
#  - Rolling CAPM Alpha panel: rolling regression + confidence intervals (CI)
#  - Advanced Performance Metrics Engine: 50+ institutional metrics + rolling charts
#  - Advanced Efficient Frontier: multi-panel + 3D frontier + CML + risk contributions
#  - Posterior weight stability / turnover diagnostics for BL vs MV vs HRP
# ------------------------------------------------------------------------------
# Deploy:
#  - streamlit run QUANTUM_Global_Institutional_Terminal_ENHANCED.py
#  - Streamlit Cloud: requirements.txt -> streamlit yfinance pandas numpy plotly scipy PyPortfolioOpt scikit-learn
# ==============================================================================

import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="QUANTUM | Advanced Risk Analytics",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

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

# ==============================================================================
# Styling: Light Grey background + Dark Blue titles (Institutional)
# ==============================================================================
st.markdown("""
<style>
    .stApp { background-color:#f5f7fa; font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
    .main-header { font-size:2.8rem; font-weight:800; color:#1a237e; text-align:center; border-bottom:3px solid #303f9f; padding-bottom:12px; margin-bottom:18px; }
    .section-header { font-size:1.7rem; font-weight:700; color:#1a237e; border-bottom:2px solid #e8eaf6; padding-bottom:8px; margin:22px 0 14px 0; }
    .subsection-header { font-size:1.25rem; font-weight:700; color:#283593; margin:16px 0 10px 0; }
    .card { background:white; border:1px solid #e1e5eb; border-radius:12px; padding:16px; margin:12px 0; box-shadow:0 2px 10px rgba(0,0,0,0.04); }
    .info-card { background:#f8fafc; border:1px solid #e5e7eb; border-left:4px solid #1f2937; border-radius:10px; padding:12px 14px; margin:10px 0; color:#111827; }
    .success-card { background:#f0fdf4; border:1px solid #dcfce7; border-left:4px solid #16a34a; border-radius:10px; padding:12px 14px; margin:10px 0; color:#052e16; }
    .warning-card { background:#fffbeb; border:1px solid #fde68a; border-left:4px solid #d97706; border-radius:10px; padding:12px 14px; margin:10px 0; color:#451a03; }
    .stTabs [data-baseweb="tab-list"] { gap:5px; background:#f0f2f6; padding:6px; border-radius:12px; }
    .stTabs [data-baseweb="tab"] { border-radius:10px; padding:10px 16px; font-weight:700; background:white; border:1px solid #e0e0e0; color:#5c6bc0; }
    .stTabs [aria-selected="true"] { background:#ffffff; color:#111827; border-color:#111827; box-shadow:0 4px 12px rgba(17,24,39,.10); }
    .stButton>button { background:#111827; color:white; border:none; border-radius:10px; padding:10px 18px; font-weight:700; }
    .stButton>button:hover { transform:translateY(-1px); box-shadow:0 6px 14px rgba(17,24,39,.18); }
    ::-webkit-scrollbar { width:8px; height:8px; }
    ::-webkit-scrollbar-track { background:#f1f1f1; border-radius:4px; }
    ::-webkit-scrollbar-thumb { background:#c5cae9; border-radius:4px; }
    ::-webkit-scrollbar-thumb:hover { background:#7986cb; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Universe / Data Manager (50+ instruments across regions)
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
            },
            "US ETFs (Major)": {
                "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM", "VTI": "VTI",
                "ARKK": "ARKK", "XLF": "XLF", "XLK": "XLK", "XLE": "XLE", "XLV": "XLV",
                "GLD": "GLD", "SLV": "SLV", "GDX": "GDX",
                "TLT": "TLT", "IEF": "IEF", "AGG": "AGG", "BND": "BND",
                "HYG": "HYG", "LQD": "LQD", "JNK": "JNK",
            },
            "Global Mega Caps": {
                "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN",
                "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA", "Berkshire": "BRK-B",
                "TSMC": "TSM", "Samsung": "005930.KS",
                "Tencent": "0700.HK", "Alibaba": "BABA",
                "LVMH": "MC.PA", "ASML": "ASML", "Novo Nordisk": "NVO",
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
                "TIP": "TIP",
                "SHY": "SHY",
            }
        }

    def ticker_name_map(self) -> Dict[str, str]:
        m = {}
        for cat, d in self.universe.items():
            for name, t in d.items():
                m[t] = name
        return m

data_manager = EnhancedDataManager()
TICKER_NAME_MAP = data_manager.ticker_name_map()

# ==============================================================================
# Robust data fetch + alignment (no NaNs; equal length)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(
    tickers: Tuple[str, ...],
    start: str,
    end: str,
    min_points: int = 200
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    if not tickers:
        return pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["No tickers selected."]}

    benchmark = "^GSPC"
    all_tickers = list(dict.fromkeys(list(tickers) + [benchmark]))

    try:
        data = yf.download(
            all_tickers, start=start, end=end,
            group_by="ticker", auto_adjust=True,
            threads=True, progress=False, timeout=30
        )
    except TypeError:
        data = yf.download(
            all_tickers, start=start, end=end,
            group_by="ticker", auto_adjust=True,
            threads=True, progress=False
        )

    def extract_close(t: str) -> Optional[pd.Series]:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if t in data.columns.levels[0] and "Close" in data[t].columns:
                    s = data[t]["Close"].copy()
                    s.name = t
                    return s
            else:
                if "Close" in data.columns:
                    s = data["Close"].copy()
                    s.name = t
                    return s
        except Exception:
            return None
        return None

    closes = {t: extract_close(t) for t in all_tickers}
    closes = {t: s for t, s in closes.items() if s is not None and s.count() > 0}

    report: Dict[str, Any] = {"warnings": [], "infos": [], "ticker_details": {}}

    if not closes:
        report["warnings"].append("No data fetched. Try a different date range or tickers.")
        return pd.DataFrame(), pd.Series(dtype=float), report

    df = pd.DataFrame(closes)
    # Ensure proper date index
    df = df.sort_index()

    # Filter insufficient
    good_cols = []
    for c in df.columns:
        non_na = int(df[c].count())
        report["ticker_details"][c] = {
            "non_na": non_na,
            "na_pct": float((1 - non_na / max(1, len(df))) * 100),
            "start": str(df[c].dropna().index.min().date()) if non_na > 0 else None,
            "end": str(df[c].dropna().index.max().date()) if non_na > 0 else None,
        }
        if non_na >= min_points:
            good_cols.append(c)
        else:
            report["warnings"].append(f"Removing {c}: insufficient data ({non_na} < {min_points}).")

    df = df[good_cols].copy()
    if df.empty or len(df) < min_points:
        report["warnings"].append("Insufficient data after filtering. Expand the date range.")
        return pd.DataFrame(), pd.Series(dtype=float), report

    # Fill missing values carefully and drop remaining NaNs
    df = df.ffill().bfill().dropna()
    report["infos"].append(f"Aligned data shape: {df.shape}")

    bench = pd.Series(dtype=float)
    if benchmark in df.columns:
        bench = df[benchmark].copy()

    # Portfolio assets only (exclude benchmark unless user selected)
    portfolio_cols = [t for t in tickers if t in df.columns]
    df_port = df[portfolio_cols].copy()

    report["start_date"] = str(df.index.min().date())
    report["end_date"] = str(df.index.max().date())
    report["rows"] = int(len(df))
    report["assets"] = int(len(df_port.columns))
    report["alignment_status"] = "SUCCESS"

    return df_port, bench, report

# ==============================================================================
# Optional optimization dependencies (PyPortfolioOpt + solvers)
#  - Robust to PyPortfolioOpt version differences
#  - If missing, optimization tabs degrade gracefully
# ==============================================================================
import importlib

def _import_attr(module_candidates, attr_name):
    """Try importing attr_name from first module that provides it."""
    last_err = ""
    for mod_name in module_candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, attr_name):
                return getattr(mod, attr_name), ""
        except Exception as _e:
            last_err = str(_e)
            continue
    return None, last_err or "Not found"

try:
    from pypfopt import expected_returns, risk_models, objective_functions
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns, market_implied_risk_aversion

    CLA, _cla_err = _import_attr(["pypfopt.cla"], "CLA")
    DiscreteAllocation, _da_err = _import_attr(["pypfopt.discrete_allocation"], "DiscreteAllocation")
    get_latest_prices, _lp_err = _import_attr(["pypfopt.discrete_allocation"], "get_latest_prices")

    EfficientSemivariance, _sv_err = _import_attr(["pypfopt.efficient_semivariance", "pypfopt.efficient_frontier"], "EfficientSemivariance")
    EfficientCVaR, _cvar_err = _import_attr(["pypfopt.efficient_cvar", "pypfopt.efficient_frontier"], "EfficientCVaR")
    EfficientCDaR, _cdar_err = _import_attr(["pypfopt.efficient_cdar", "pypfopt.efficient_frontier"], "EfficientCDaR")

    OPTIMIZATION_AVAILABLE = True
    OPTIMIZATION_IMPORT_ERROR = ""
except Exception as _e:
    OPTIMIZATION_AVAILABLE = False
    OPTIMIZATION_IMPORT_ERROR = str(_e)

# ==============================================================================
# 4B) OHLC + Benchmark Helpers (NEW)
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(tickers: Tuple[str, ...], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch daily OHLC data for one or more tickers (robust to yfinance column layouts)."""
    if not tickers:
        return {}

    tickers = tuple(dict.fromkeys(list(tickers)))

    try:
        data = yf.download(
            list(tickers),
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
            timeout=30,
        )
    except TypeError:
        data = yf.download(
            list(tickers),
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )

    out: Dict[str, pd.DataFrame] = {}
    if data is None or len(data) == 0:
        return out

    def _extract_one(t: str) -> pd.DataFrame:
        # group_by='ticker' usually => top-level tickers (MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            if t in data.columns.get_level_values(0):
                sub = data[t].copy()
            elif t in data.columns.get_level_values(-1):
                try:
                    sub = data.xs(t, level=-1, axis=1).copy()
                except Exception:
                    sub = pd.DataFrame()
            else:
                sub = pd.DataFrame()
        else:
            sub = data.copy()

        need = ["Open", "High", "Low", "Close"]
        if sub is None or sub.empty:
            return pd.DataFrame()
        # Some tickers return lowercase columns in rare cases
        cols = [c for c in need if c in sub.columns]
        if len(cols) < 4:
            # Try case-insensitive mapping
            lower_map = {str(c).lower(): c for c in sub.columns}
            cols2 = []
            for c in need:
                if c.lower() in lower_map:
                    cols2.append(lower_map[c.lower()])
            if len(cols2) >= 4:
                sub = sub[cols2].copy()
                sub.columns = need
                return sub.dropna(how="all")
            return pd.DataFrame()

        return sub[cols].copy().dropna(how="all")

    for t in tickers:
        df = _extract_one(t)
        if df is not None and not df.empty:
            out[t] = df

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close_series(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch adjusted close series for a single ticker."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=30,
        )
    except TypeError:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        return df["Close"].rename(ticker).dropna()
    # Fallback
    return df.iloc[:, 0].rename(ticker).dropna()


def bollinger_bands(price: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (mid, upper, lower)."""
    p = price.dropna()
    mid = p.rolling(window).mean()
    std = p.rolling(window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def tracking_error(active_returns: pd.Series, annualization: int = 252) -> float:
    """Annualized tracking error."""
    ar = active_returns.dropna()
    if len(ar) < 5:
        return float("nan")
    return float(ar.std() * np.sqrt(annualization))


def compute_portfolio_index(prices: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Portfolio index (base 100) from price panel + weights."""
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    w = pd.Series(weights).copy()
    w = w[w.index.isin(prices.columns)]
    if w.empty:
        # equal weight fallback
        w = pd.Series({c: 1.0 / prices.shape[1] for c in prices.columns})
    w = w / w.sum()
    ret = prices.pct_change().dropna()
    pr = portfolio_returns_from_weights(ret, w.to_dict())
    idx = (1 + pr).cumprod() * 100
    idx.name = "Portfolio Index"
    return idx


def compute_portfolio_ohlc_index(ohlc_map: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    """Build a portfolio OHLC index (base 100) using weighted normalized OHLC of constituents."""
    if not ohlc_map:
        return pd.DataFrame()

    w = pd.Series(weights).copy()
    w = w[w.index.isin(list(ohlc_map.keys()))]
    if w.empty:
        keys = list(ohlc_map.keys())
        w = pd.Series({k: 1.0 / len(keys) for k in keys})
    w = w / w.sum()

    # Align on common index
    common_index = None
    for t, df in ohlc_map.items():
        if df is None or df.empty:
            continue
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    if common_index is None or len(common_index) < 20:
        return pd.DataFrame()

    port = None
    for t, df in ohlc_map.items():
        if t not in w.index or df is None or df.empty:
            continue
        d = df.loc[common_index, ["Open", "High", "Low", "Close"]].copy()
        if d["Close"].iloc[0] == 0 or not np.isfinite(d["Close"].iloc[0]):
            continue
        d = d / float(d["Close"].iloc[0]) * 100.0
        d = d * float(w.loc[t])
        port = d if port is None else port.add(d, fill_value=0.0)

    if port is None or port.empty:
        return pd.DataFrame()

    port.columns = ["Open", "High", "Low", "Close"]
    port = port.dropna()
    return port


def fetch_market_caps(tickers: Tuple[str, ...]) -> pd.Series:
    caps = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            fi = tk.fast_info if hasattr(tk, "fast_info") else None
            mc = None
            if isinstance(fi, dict):
                mc = fi.get("marketCap", None)
            if mc is None:
                info = tk.info
                if isinstance(info, dict):
                    mc = info.get("marketCap", None)
            caps[t] = float(mc) if (mc is not None and np.isfinite(mc) and mc > 0) else np.nan
        except Exception:
            caps[t] = np.nan
    return pd.Series(caps, dtype="float64")

# ==============================================================================
# Portfolio Engine: returns, optimization methods, utility functions
# ==============================================================================
def _clean_weights(w: Dict[str, float]) -> Dict[str, float]:
    w2 = {k: float(v) for k, v in w.items() if np.isfinite(v)}
    s = sum(w2.values())
    if s <= 0:
        n = len(w2) if len(w2) > 0 else 1
        return {k: 1.0 / n for k in w2.keys()}
    return {k: v / s for k, v in w2.items()}

def portfolio_returns_from_weights(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    w = pd.Series(_clean_weights(weights)).reindex(returns.columns).fillna(0.0)
    return returns.dot(w)

def max_drawdown(r: pd.Series) -> float:
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())

def annualize_return(r: pd.Series, periods: int = 252) -> float:
    if r.empty:
        return float("nan")
    total = (1 + r).prod()
    years = len(r) / periods
    return float(total ** (1 / max(1e-9, years)) - 1)

def annualize_vol(r: pd.Series, periods: int = 252) -> float:
    return float(r.std() * np.sqrt(periods)) if len(r) > 1 else float("nan")

def sharpe_ratio(r: pd.Series, rf_annual: float = 0.04, periods: int = 252) -> float:
    mu = r.mean() * periods
    vol = r.std() * np.sqrt(periods)
    return float((mu - rf_annual) / vol) if vol > 0 else float("nan")

class PortfolioEngine:
    def __init__(self, prices: pd.DataFrame):
        if prices is None or prices.empty:
            raise ValueError("Empty prices.")
        self.prices = prices.copy()
        self.returns = self.prices.pct_change().dropna()
        self.assets = list(self.prices.columns)

        if OPTIMIZATION_AVAILABLE:
            self.mu = expected_returns.mean_historical_return(self.prices)
            self.S = risk_models.sample_cov(self.prices)
        else:
            self.mu = self.returns.mean() * 252
            self.S = self.returns.cov() * 252

    def equal_weight(self) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
        n = len(self.assets)
        w = {a: 1.0 / n for a in self.assets}
        pr = portfolio_returns_from_weights(self.returns, w)
        return w, (annualize_return(pr), annualize_vol(pr), sharpe_ratio(pr))

    def mean_variance(self, objective: str = "max_sharpe", rf: float = 0.04, gamma: Optional[float] = None) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
        if not OPTIMIZATION_AVAILABLE or EfficientFrontier is None:
            return self.equal_weight()
        ef = EfficientFrontier(self.mu, self.S)
        if gamma is not None and gamma > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=float(gamma))
        try:
            if objective == "min_volatility":
                ef.min_volatility()
            elif objective == "max_quadratic_utility":
                ef.max_quadratic_utility(risk_aversion=1)
            else:
                ef.max_sharpe(risk_free_rate=rf)
            w = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
            return _clean_weights(w), (float(perf[0]), float(perf[1]), float(perf[2]))
        except Exception:
            return self.equal_weight()

    def hrp(self) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
        if not HRP_AVAILABLE or HRPOpt is None:
            return self.equal_weight()
        try:
            hrp = HRPOpt(returns=self.returns)
            hrp.optimize()
            w = hrp.clean_weights()
            pr = portfolio_returns_from_weights(self.returns, w)
            return _clean_weights(w), (annualize_return(pr), annualize_vol(pr), sharpe_ratio(pr))
        except Exception:
            return self.equal_weight()

    def black_litterman(
        self,
        rf: float = 0.04,
        tau: float = 0.05,
        objective: str = "max_sharpe",
        views_abs: Optional[Dict[str, float]] = None,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        view_confidences: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
        benchmark_returns: Optional[pd.Series] = None,
        delta_override: Optional[float] = None,
        use_market_caps: bool = True
    ) -> Tuple[Dict[str, float], Tuple[float, float, float], pd.Series, pd.Series]:
        """
        Returns: weights, (ret,vol,sharpe), prior(pi), posterior(returns)
        """
        if not BLACK_LITTERMAN_AVAILABLE or BlackLittermanModel is None or not OPTIMIZATION_AVAILABLE:
            w, perf = self.mean_variance(objective="max_sharpe", rf=rf)
            prior = pd.Series(self.mu).reindex(self.assets)
            post = prior.copy()
            return w, perf, prior, post

        # market caps
        mkt_caps = None
        if use_market_caps:
            caps = fetch_market_caps(tuple(self.assets))
            if caps.notna().sum() >= max(2, int(0.5 * len(caps))):
                mkt_caps = caps.fillna(caps.median())
        if mkt_caps is None:
            mkt_caps = pd.Series({a: 1.0 for a in self.assets})

        # delta
        if delta_override is not None and np.isfinite(delta_override) and float(delta_override) > 0:
            delta = float(delta_override)
        else:
            delta = 2.5
            if benchmark_returns is not None and market_implied_risk_aversion is not None:
                try:
                    b = benchmark_returns.dropna()
                    if len(b) > 60:
                        delta = float(market_implied_risk_aversion(b, risk_free_rate=rf))
                except Exception:
                    delta = 2.5

        # prior
        try:
            if market_implied_prior_returns is not None:
                pi = market_implied_prior_returns(mkt_caps, delta, self.S)
            else:
                pi = self.mu.copy()
        except Exception:
            pi = self.mu.copy()

        # confidences
        abs_views = {}
        abs_conf = None
        if views_abs:
            abs_views = {k: float(v) for k, v in views_abs.items() if k in self.assets and np.isfinite(v)}
            if isinstance(view_confidences, dict):
                abs_conf = {k: float(view_confidences.get(k, 0.5)) for k in abs_views.keys()}
                abs_conf = {k: float(np.clip(v, 0.05, 0.95)) for k, v in abs_conf.items()}

        rel_conf = None
        if P is not None and Q is not None:
            try:
                if isinstance(view_confidences, (list, tuple, np.ndarray)):
                    rel_conf = np.clip(np.array(view_confidences, dtype=float), 0.05, 0.95)
                else:
                    rel_conf = np.full(shape=(len(Q),), fill_value=0.5, dtype=float)
            except Exception:
                rel_conf = np.full(shape=(len(Q),), fill_value=0.5, dtype=float)

        try:
            if P is not None and Q is not None:
                bl = BlackLittermanModel(
                    self.S, pi=pi, P=P, Q=Q,
                    omega="idzorek", view_confidences=rel_conf,
                    tau=float(tau)
                )
            else:
                bl = BlackLittermanModel(
                    self.S, pi=pi, absolute_views=abs_views if abs_views else None,
                    omega="idzorek" if abs_views else None,
                    view_confidences=abs_conf if abs_views else None,
                    tau=float(tau)
                )
            post = pd.Series(bl.bl_returns()).reindex(self.assets)
            cov = bl.bl_cov()

            ef = EfficientFrontier(post, cov)
            if objective == "min_volatility":
                ef.min_volatility()
            elif objective == "max_quadratic_utility":
                ef.max_quadratic_utility(risk_aversion=1)
            else:
                ef.max_sharpe(risk_free_rate=rf)
            w = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
            prior = pd.Series(pi).reindex(self.assets)
            return _clean_weights(w), (float(perf[0]), float(perf[1]), float(perf[2])), prior, post
        except Exception:
            w, perf = self.mean_variance(objective="max_sharpe", rf=rf)
            prior = pd.Series(pi).reindex(self.assets)
            post = prior.copy()
            return w, perf, prior, post

# --- Advanced PyPortfolioOpt strategies (robust imports) ---
def cla(self) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Critical Line Algorithm (CLA)."""
    if not OPTIMIZATION_AVAILABLE or CLA is None:
        return self.equal_weight()
    try:
        cla = CLA(self.mu, self.S)
        if hasattr(cla, "max_sharpe"):
            cla.max_sharpe()
        w = cla.clean_weights() if hasattr(cla, "clean_weights") else {a: float(x) for a, x in zip(self.assets, cla.weights)}
        perf = cla.portfolio_performance(verbose=False) if hasattr(cla, "portfolio_performance") else None
        if perf is not None:
            return _clean_weights(w), (float(perf[0]), float(perf[1]), float(perf[2]))
        pr = portfolio_returns_from_weights(self.returns, _clean_weights(w))
        return _clean_weights(w), (annualize_return(pr), annualize_vol(pr), sharpe_ratio(pr))
    except Exception:
        return self.equal_weight()

def semivariance(self, benchmark: float = 0.0) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Efficient Semivariance optimization (downside risk)."""
    if not OPTIMIZATION_AVAILABLE or EfficientSemivariance is None:
        # fallback: semicovariance risk matrix + min volatility
        try:
            if OPTIMIZATION_AVAILABLE and EfficientFrontier is not None:
                S_semi = risk_models.semicovariance(self.returns, benchmark=benchmark)
                ef = EfficientFrontier(self.mu, S_semi)
                ef.min_volatility()
                w = ef.clean_weights()
                perf = ef.portfolio_performance(verbose=False)
                return _clean_weights(w), (float(perf[0]), float(perf[1]), float(perf[2]))
        except Exception:
            pass
        return self.equal_weight()

    try:
        try:
            ef = EfficientSemivariance(self.mu, self.returns, benchmark=benchmark)
        except TypeError:
            ef = EfficientSemivariance(self.mu, self.returns)

        for _m in ["min_semivariance", "min_volatility", "max_sharpe"]:
            if hasattr(ef, _m):
                try:
                    getattr(ef, _m)()
                    break
                except TypeError:
                    getattr(ef, _m)(risk_free_rate=0.0)
                    break

        w = ef.clean_weights() if hasattr(ef, "clean_weights") else ef.weights
        pr = portfolio_returns_from_weights(self.returns, _clean_weights(w))
        return _clean_weights(w), (annualize_return(pr), annualize_vol(pr), sharpe_ratio(pr))
    except Exception:
        return self.equal_weight()

def cvar(self) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Efficient CVaR (Expected Shortfall) optimization."""
    if not OPTIMIZATION_AVAILABLE or EfficientCVaR is None:
        return self.equal_weight()
    try:
        try:
            ef = EfficientCVaR(self.mu, self.returns)
        except TypeError:
            ef = EfficientCVaR(self.mu, self.returns.values)

        for _m in ["min_cvar", "min_volatility", "max_sharpe"]:
            if hasattr(ef, _m):
                try:
                    getattr(ef, _m)()
                    break
                except TypeError:
                    getattr(ef, _m)(risk_free_rate=0.0)
                    break

        w = ef.clean_weights() if hasattr(ef, "clean_weights") else ef.weights
        pr = portfolio_returns_from_weights(self.returns, _clean_weights(w))
        return _clean_weights(w), (annualize_return(pr), annualize_vol(pr), sharpe_ratio(pr))
    except Exception:
        return self.equal_weight()

def cdar(self) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Efficient CDaR (Conditional Drawdown-at-Risk) optimization."""
    if not OPTIMIZATION_AVAILABLE or EfficientCDaR is None:
        return self.equal_weight()
    try:
        try:
            ef = EfficientCDaR(self.mu, self.returns)
        except TypeError:
            ef = EfficientCDaR(self.mu, self.returns.values)

        for _m in ["min_cdar", "min_volatility", "max_sharpe"]:
            if hasattr(ef, _m):
                try:
                    getattr(ef, _m)()
                    break
                except TypeError:
                    getattr(ef, _m)(risk_free_rate=0.0)
                    break

        w = ef.clean_weights() if hasattr(ef, "clean_weights") else ef.weights
        pr = portfolio_returns_from_weights(self.returns, _clean_weights(w))
        return _clean_weights(w), (annualize_return(pr), annualize_vol(pr), sharpe_ratio(pr))
    except Exception:
        return self.equal_weight()

def discrete_allocation(self, weights: Dict[str, float], portfolio_value: float = 1000000.0) -> Tuple[pd.DataFrame, float]:
    """Discrete allocation (integer shares). Returns (allocation_df, leftover_cash)."""
    if not OPTIMIZATION_AVAILABLE or DiscreteAllocation is None or get_latest_prices is None:
        return pd.DataFrame(), float(portfolio_value)
    try:
        latest_prices = get_latest_prices(self.prices)
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=float(portfolio_value))
        alloc, leftover = da.lp_portfolio()
        df = pd.DataFrame({"Ticker": list(alloc.keys()), "Shares": list(alloc.values())})
        df["AssetName"] = df["Ticker"].map(lambda t: TICKER_NAME_MAP.get(t, t))
        df = df.sort_values("Shares", ascending=False)
        return df, float(leftover)
    except Exception:
        return pd.DataFrame(), float(portfolio_value)

# ==============================================================================
# Advanced Performance Metrics Engine (50+)
# ==============================================================================
class AdvancedPerformanceMetrics:
    """
    Institutional 50+ metric engine for portfolio (and optional benchmark).
    """
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None, rf_annual: float = 0.04, periods: int = 252):
        self.rp = portfolio_returns.dropna().astype(float)
        self.rb = benchmark_returns.dropna().astype(float) if benchmark_returns is not None else None
        self.rf = float(rf_annual)
        self.periods = int(periods)

        if self.rb is not None:
            df = pd.concat([self.rp, self.rb], axis=1).dropna()
            self.rp = df.iloc[:, 0]
            self.rb = df.iloc[:, 1]

    def _dd_series(self) -> pd.Series:
        cum = (1 + self.rp).cumprod()
        peak = cum.cummax()
        return (cum - peak) / peak

    def _drawdown_durations(self) -> Tuple[int, float]:
        dd = self._dd_series()
        in_dd = (dd < 0).astype(int)
        # duration: max consecutive drawdown days
        max_dur = 0
        cur = 0
        for v in in_dd.values:
            if v == 1:
                cur += 1
                max_dur = max(max_dur, cur)
            else:
                cur = 0
        # recovery time approx: last dd<0 segment length (if still in dd -> NaN)
        if dd.iloc[-1] < 0:
            rec = np.nan
        else:
            # find last time dd hit 0 after being negative
            zeros = np.where(dd.values == 0)[0]
            rec = float(len(dd) - zeros[-1]) if len(zeros) else np.nan
        return int(max_dur), float(rec)

    def _capm(self) -> Dict[str, float]:
        if self.rb is None or len(self.rb) < 20:
            return {"alpha": np.nan, "beta": np.nan, "r2": np.nan, "tracking_error": np.nan, "info_ratio": np.nan}
        x = self.rb.values
        y = self.rp.values
        x = x - x.mean()
        y = y - y.mean()
        varx = float(np.dot(x, x))
        beta = float(np.dot(x, y) / varx) if varx > 0 else np.nan
        # alpha from original (annualized)
        beta2 = beta if np.isfinite(beta) else 0.0
        alpha_daily = float(self.rp.mean() - beta2 * self.rb.mean())
        alpha = alpha_daily * self.periods
        # r2
        yhat = beta2 * (self.rb - self.rb.mean()) + (self.rp.mean())
        resid = self.rp - yhat
        sst = float(((self.rp - self.rp.mean()) ** 2).sum())
        ssr = float((resid ** 2).sum())
        r2 = float(1 - ssr / sst) if sst > 0 else np.nan
        te = float((self.rp - self.rb).std() * np.sqrt(self.periods))
        ir = float((annualize_return(self.rp, self.periods) - annualize_return(self.rb, self.periods)) / te) if te > 0 else np.nan
        return {"alpha": alpha, "beta": beta, "r2": r2, "tracking_error": te, "info_ratio": ir}

    def compute(self) -> pd.DataFrame:
        rp = self.rp
        periods = self.periods
        rf = self.rf
        out: Dict[str, float] = {}

        # Return metrics
        out["Total Return"] = float((1 + rp).prod() - 1)
        out["CAGR"] = annualize_return(rp, periods)
        out["Annualized Mean Return"] = float(rp.mean() * periods)
        out["Median Daily Return"] = float(rp.median())
        out["Best Day"] = float(rp.max())
        out["Worst Day"] = float(rp.min())
        out["Positive Days %"] = float((rp > 0).mean())
        out["Negative Days %"] = float((rp < 0).mean())

        # Risk metrics
        out["Annualized Volatility"] = annualize_vol(rp, periods)
        out["Downside Deviation"] = float(rp[rp < 0].std() * np.sqrt(periods)) if (rp < 0).any() else np.nan
        out["Upside Deviation"] = float(rp[rp > 0].std() * np.sqrt(periods)) if (rp > 0).any() else np.nan
        out["Skewness"] = float(rp.skew())
        out["Kurtosis"] = float(rp.kurtosis())
        out["Jarque-Bera Stat"] = float(stats.jarque_bera(rp).statistic) if len(rp) > 20 else np.nan
        out["Jarque-Bera p-value"] = float(stats.jarque_bera(rp).pvalue) if len(rp) > 20 else np.nan

        dd = self._dd_series()
        out["Max Drawdown"] = float(dd.min())
        out["Avg Drawdown"] = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
        out["Ulcer Index"] = float(np.sqrt(np.mean((dd.values * 100) ** 2)))

        max_dd = abs(out["Max Drawdown"]) if np.isfinite(out["Max Drawdown"]) else np.nan

        # Risk-adjusted ratios
        out["Sharpe Ratio"] = sharpe_ratio(rp, rf, periods)
        out["Sortino Ratio"] = float((out["CAGR"] - rf) / out["Downside Deviation"]) if np.isfinite(out["Downside Deviation"]) and out["Downside Deviation"] > 0 else np.nan
        out["Calmar Ratio"] = float(out["CAGR"] / max_dd) if (np.isfinite(max_dd) and max_dd > 0) else np.nan

        # Omega / Gain-to-pain / Profit factor
        thr = rf / periods
        excess = rp - thr
        gains = float(excess[excess > 0].sum())
        losses = float(abs(excess[excess < 0].sum()))
        out["Omega Ratio"] = float(gains / losses) if losses > 0 else np.inf

        gross_profit = float(rp[rp > 0].sum())
        gross_loss = float(abs(rp[rp < 0].sum()))
        out["Profit Factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
        out["Gain-to-Pain Ratio"] = float(rp.sum() / gross_loss) if gross_loss > 0 else np.inf

        # VaR / CVaR / Tail metrics (multiple levels)
        for cl in [0.90, 0.95, 0.99]:
            alpha = 1 - cl
            var = -float(np.percentile(rp, alpha * 100)) if len(rp) > 30 else np.nan
            tail = rp[rp <= -var] if np.isfinite(var) else pd.Series(dtype=float)
            cvar = -float(tail.mean()) if len(tail) > 0 else np.nan
            out[f"VaR ({int(cl*100)}%)"] = var
            out[f"CVaR/ES ({int(cl*100)}%)"] = cvar
            out[f"Tail Loss Count ({int(cl*100)}%)"] = float(len(tail))

        out["Tail Ratio (95/5)"] = float(np.percentile(rp, 95) / abs(np.percentile(rp, 5))) if len(rp) > 30 and np.percentile(rp, 5) != 0 else np.nan

        # Win/Loss analysis
        wins = rp[rp > 0]
        losses_r = rp[rp < 0]
        out["Win Rate"] = float((rp > 0).mean())
        out["Avg Win"] = float(wins.mean()) if len(wins) else np.nan
        out["Avg Loss"] = float(losses_r.mean()) if len(losses_r) else np.nan
        out["Win/Loss Ratio"] = float(abs(out["Avg Win"] / out["Avg Loss"])) if (np.isfinite(out["Avg Win"]) and np.isfinite(out["Avg Loss"]) and out["Avg Loss"] != 0) else np.nan

        # Autocorrelation
        out["Autocorr(1)"] = float(rp.autocorr(lag=1)) if len(rp) > 10 else np.nan
        out["Autocorr(5)"] = float(rp.autocorr(lag=5)) if len(rp) > 20 else np.nan

        # Drawdown durations/recovery
        max_dur, rec = self._drawdown_durations()
        out["Max Drawdown Duration (days)"] = float(max_dur)
        out["Recovery Days (approx)"] = float(rec)

        # Benchmark metrics
        capm = self._capm()
        out["CAPM Alpha (ann)"] = float(capm["alpha"])
        out["CAPM Beta"] = float(capm["beta"])
        out["CAPM R^2"] = float(capm["r2"])
        out["Tracking Error (ann)"] = float(capm["tracking_error"])
        out["Information Ratio"] = float(capm["info_ratio"])

        if self.rb is not None and len(self.rb) > 20:
            out["Benchmark CAGR"] = annualize_return(self.rb, periods)
            out["Benchmark Volatility"] = annualize_vol(self.rb, periods)
            out["Correlation vs Benchmark"] = float(rp.corr(self.rb))
            out["Up Capture"] = float(rp[self.rb > 0].mean() / self.rb[self.rb > 0].mean()) if (self.rb > 0).any() else np.nan
            out["Down Capture"] = float(rp[self.rb < 0].mean() / self.rb[self.rb < 0].mean()) if (self.rb < 0).any() else np.nan

        # Additional institutional diagnostics (to exceed 50)
        out["Volatility of Volatility (ann)"] = float(rp.rolling(21).std().std() * np.sqrt(periods)) if len(rp) > 50 else np.nan
        out["Rolling 1M Return (last)"] = float((1 + rp.tail(21)).prod() - 1) if len(rp) > 21 else np.nan
        out["Rolling 3M Return (last)"] = float((1 + rp.tail(63)).prod() - 1) if len(rp) > 63 else np.nan
        out["Rolling 6M Return (last)"] = float((1 + rp.tail(126)).prod() - 1) if len(rp) > 126 else np.nan
        out["Rolling 12M Return (last)"] = float((1 + rp.tail(252)).prod() - 1) if len(rp) > 252 else np.nan
        out["Monthly Return Std"] = float(rp.resample("M").apply(lambda x: (1+x).prod()-1).std()) if len(rp) > 50 else np.nan
        out["Monthly Return Mean"] = float(rp.resample("M").apply(lambda x: (1+x).prod()-1).mean()) if len(rp) > 50 else np.nan
        out["Max Monthly Loss"] = float(rp.resample("M").apply(lambda x: (1+x).prod()-1).min()) if len(rp) > 50 else np.nan
        out["Max Monthly Gain"] = float(rp.resample("M").apply(lambda x: (1+x).prod()-1).max()) if len(rp) > 50 else np.nan
        out["Semi-Variance"] = float(np.mean(np.minimum(rp, 0) ** 2))
        out["Expected Daily Shortfall (1%)"] = float(-rp[rp <= np.percentile(rp, 1)].mean()) if len(rp) > 100 else np.nan
        out["Expected Daily Shortfall (5%)"] = float(-rp[rp <= np.percentile(rp, 5)].mean()) if len(rp) > 60 else np.nan

        # Format
        df = pd.DataFrame({"Metric": list(out.keys()), "Value": list(out.values())})
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        return df

# ==============================================================================
# Rolling Beta Heatmap + Rolling CAPM with confidence intervals
# ==============================================================================
def compute_rolling_beta(asset_returns: pd.DataFrame, benchmark: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    Rolling beta for each asset vs benchmark.
    """
    df = pd.concat([asset_returns, benchmark.rename("BENCH")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    bench = df["BENCH"]
    betas = {}
    for col in asset_returns.columns:
        if col not in df.columns:
            continue
        x = bench
        y = df[col]
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var()
        beta = cov / var.replace(0, np.nan)
        betas[col] = beta
    return pd.DataFrame(betas).dropna(how="all")

def rolling_capm_with_ci(asset: pd.Series, benchmark: pd.Series, window: int = 126, alpha: float = 0.05) -> pd.DataFrame:
    """
    Rolling OLS: asset = a + b*bench + eps
    Returns alpha,beta and CI bands.
    """
    df = pd.concat([asset.rename("y"), benchmark.rename("x")], axis=1).dropna()
    if len(df) < window + 5:
        return pd.DataFrame()

    ys = df["y"].values
    xs = df["x"].values

    out = []
    z = stats.norm.ppf(1 - alpha / 2)

    for i in range(window, len(df) + 1):
        y = ys[i-window:i]
        x = xs[i-window:i]
        X = np.column_stack([np.ones_like(x), x])
        # OLS
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        a, b = beta_hat[0], beta_hat[1]
        yhat = X @ beta_hat
        resid = y - yhat
        dof = max(1, len(y) - 2)
        s2 = float((resid @ resid) / dof)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_a = float(np.sqrt(s2 * XtX_inv[0, 0]))
        se_b = float(np.sqrt(s2 * XtX_inv[1, 1]))
        # r2
        sst = float(((y - y.mean()) ** 2).sum())
        ssr = float((resid ** 2).sum())
        r2 = float(1 - ssr / sst) if sst > 0 else np.nan
        dt = df.index[i-1]
        out.append({
            "date": dt,
            "alpha_daily": float(a),
            "beta": float(b),
            "alpha_low": float(a - z * se_a),
            "alpha_high": float(a + z * se_a),
            "beta_low": float(b - z * se_b),
            "beta_high": float(b + z * se_b),
            "r2": float(r2),
        })

    res = pd.DataFrame(out).set_index("date")
    # annualize alpha (but keep daily too)
    res["alpha_ann"] = res["alpha_daily"] * 252
    res["alpha_low_ann"] = res["alpha_low"] * 252
    res["alpha_high_ann"] = res["alpha_high"] * 252
    return res

# ==============================================================================
# Advanced VaR Engine (Historical, Parametric, MC, EVT) + visualizations
# ==============================================================================
class AdvancedVaREngine:
    def __init__(self, returns: pd.Series):
        self.r = returns.dropna().astype(float)

    def historical(self, cl: float = 0.95, hp: int = 1) -> Dict[str, Any]:
        if len(self.r) < 50:
            return {"Method": "Historical", "VaR_1d": np.nan, "VaR_hp": np.nan, "CVaR": np.nan}
        a = 1 - cl
        var_1d = -np.percentile(self.r, a * 100)
        var_hp = var_1d * np.sqrt(hp)
        tail = self.r[self.r <= -var_1d]
        cvar = -tail.mean() if len(tail) else np.nan
        return {"Method": "Historical", "VaR_1d": float(var_1d), "VaR_hp": float(var_hp), "CVaR": float(cvar)}

    def parametric(self, cl: float = 0.95, hp: int = 1, dist: str = "normal") -> Dict[str, Any]:
        mu = float(self.r.mean())
        sig = float(self.r.std())
        if dist == "t":
            df = max(5, len(self.r) - 1)
            z = stats.t.ppf(1 - cl, df)
        elif dist == "laplace":
            z = np.log(2*cl) if cl > 0.5 else -np.log(2*(1-cl))
        else:
            z = stats.norm.ppf(1 - cl)
        var_1d = -(mu + z * sig)
        var_hp = -(mu * hp + z * sig * np.sqrt(hp))
        cvar = -(mu - sig * stats.norm.pdf(z) / (1 - cl)) if dist == "normal" else np.nan
        return {"Method": f"Parametric({dist})", "VaR_1d": float(var_1d), "VaR_hp": float(var_hp), "CVaR": float(cvar)}

    def monte_carlo(self, cl: float = 0.95, hp: int = 1, sims: int = 5000) -> Dict[str, Any]:
        mu = float(self.r.mean())
        sig = float(self.r.std())
        sim = np.random.normal(mu, sig, size=(sims, hp))
        cum = (1 + sim).prod(axis=1) - 1
        var_hp = -np.percentile(cum, (1 - cl) * 100)
        tail = cum[cum <= -var_hp]
        cvar = -tail.mean() if len(tail) else np.nan
        var_1d = var_hp / np.sqrt(hp) if hp > 1 else var_hp
        return {"Method": "MonteCarlo", "VaR_1d": float(var_1d), "VaR_hp": float(var_hp), "CVaR": float(cvar), "Sims": int(sims)}

    def evt_gpd(self, cl: float = 0.95, threshold_q: float = 0.90) -> Dict[str, Any]:
        from scipy.stats import genpareto
        r = np.sort(self.r.values)
        n = len(r)
        if n < 200:
            return {"Method": "EVT(GPD)", "VaR_1d": np.nan, "VaR_hp": np.nan, "CVaR": np.nan}
        th_idx = int(n * threshold_q)
        th = r[th_idx]
        exc = r[r < th] - th
        if len(exc) < 20:
            return {"Method": "EVT(GPD)", "VaR_1d": np.nan, "VaR_hp": np.nan, "CVaR": np.nan}
        try:
            params = genpareto.fit(-exc)
            xi, beta = float(params[0]), float(params[2])
            n_u = len(exc)
            var = th + (beta/xi) * (((n/n_u)*(1-cl))**(-xi) - 1) if xi != 0 else th + beta*np.log((n/n_u)*(1-cl))
            cvar = (var + beta - xi*th) / (1 - xi) if xi < 1 else np.inf
            return {"Method": "EVT(GPD)", "VaR_1d": float(-var), "VaR_hp": float(-var*np.sqrt(5)), "CVaR": float(-cvar), "xi": xi, "beta": beta, "th": float(th), "exc": int(n_u)}
        except Exception:
            return {"Method": "EVT(GPD)", "VaR_1d": np.nan, "VaR_hp": np.nan, "CVaR": np.nan}

    def compare_methods(self, cl: float = 0.95, hp: int = 1) -> pd.DataFrame:
        methods = [
            self.historical(cl, hp),
            self.parametric(cl, hp, "normal"),
            self.parametric(cl, hp, "t"),
            self.parametric(cl, hp, "laplace"),
            self.monte_carlo(cl, hp, sims=8000),
            self.evt_gpd(cl, threshold_q=0.90)
        ]
        df = pd.DataFrame(methods)
        df["VaR_1d_%"] = df["VaR_1d"] * 100
        df["CVaR_%"] = df["CVaR"] * 100
        return df

    def chart_distribution(self, cl: float = 0.95) -> go.Figure:
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Return Distribution", "Loss Distribution & VaR",
                                            "QQ Tail Diagnostic", "VaR/CVaR Method Comparison"),
                            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                                   [{"type": "scatter"}, {"type": "bar"}]],
                            vertical_spacing=0.15, horizontal_spacing=0.12)
        r = self.r * 100
        fig.add_trace(go.Histogram(x=r, nbinsx=60, name="Returns", opacity=0.75), row=1, col=1)

        x = np.linspace(r.min(), r.max(), 200)
        y = stats.norm.pdf(x, r.mean(), r.std())
        fig.add_trace(go.Scatter(x=x, y=y * len(r) * (x[1]-x[0]), mode="lines", name="Normal Fit"), row=1, col=1)

        losses = (-self.r[self.r < 0]) * 100
        fig.add_trace(go.Histogram(x=losses, nbinsx=50, name="Losses", opacity=0.75), row=1, col=2)

        var = self.historical(cl)["VaR_1d"] * 100
        fig.add_vline(x=var, line_dash="dash", annotation_text=f"VaR {int(cl*100)}%: {var:.2f}%", row=1, col=2)

        theo = stats.norm.ppf(np.linspace(0.01, 0.99, len(self.r)))
        samp = np.sort(self.r.values)
        fig.add_trace(go.Scatter(x=theo, y=samp, mode="markers", name="QQ Plot"), row=2, col=1)
        m1 = min(theo.min(), samp.min())
        m2 = max(theo.max(), samp.max())
        fig.add_trace(go.Scatter(x=[m1, m2], y=[m1, m2], mode="lines", name="45° Line", line=dict(dash="dash")), row=2, col=1)

        dfm = self.compare_methods(cl, 1).dropna(subset=["VaR_1d_%"])
        fig.add_trace(go.Bar(x=dfm["Method"], y=dfm["VaR_1d_%"], name="VaR(%)", text=dfm["VaR_1d_%"].map(lambda v: f"{v:.2f}%"), textposition="auto"), row=2, col=2)
        fig.add_trace(go.Scatter(x=dfm["Method"], y=dfm["CVaR_%"], mode="lines+markers", name="CVaR(%)"), row=2, col=2)

        fig.update_layout(height=820, template="plotly_white", title=f"Advanced VaR/CVaR/ES Diagnostics (CL={int(cl*100)}%)",
                          title_font_color="#1a237e")
        fig.update_xaxes(tickangle=45, row=2, col=2)
        return fig

    def var_surface(self) -> go.Figure:
        confs = np.linspace(0.90, 0.995, 18)
        hps = np.arange(1, 21)
        Z = np.zeros((len(confs), len(hps)))
        for i, c in enumerate(confs):
            for j, hp in enumerate(hps):
                Z[i, j] = self.historical(c, hp)["VaR_hp"] * 100
        fig = go.Figure(data=[go.Surface(z=Z, x=hps, y=confs*100, contours={"z": {"show": True}})])
        fig.update_layout(height=720, title="VaR Surface: Confidence vs Holding Period",
                          title_font_color="#1a237e",
                          scene=dict(xaxis_title="Holding Period (Days)", yaxis_title="Confidence (%)", zaxis_title="VaR (%)"))
        return fig

# ==============================================================================
# Stress Testing Engine (Historical + Custom scenarios)
# ==============================================================================

class StressTestEngine:
    HISTORICAL = {
        # Major equity drawdowns / shocks
        "Black Monday (1987)": ("1987-10-14", "1987-10-30", "1987 crash / liquidity shock", "Extreme"),
        "Asian Financial Crisis (1997)": ("1997-07-02", "1998-01-28", "Currency crisis + contagion", "Severe"),
        "Russia Default & LTCM (1998)": ("1998-08-17", "1998-10-08", "Sovereign default + hedge fund deleveraging", "High"),
        "Dot-com Bubble Burst (2000-2002)": ("2000-03-10", "2002-10-09", "Tech bubble collapse", "Severe"),
        "9/11 Shock (2001)": ("2001-09-10", "2001-09-21", "Terror attacks + market closure", "High"),
        "Global Financial Crisis (2008-2009)": ("2007-10-09", "2009-03-09", "Banking crisis / deleveraging", "Extreme"),
        "Flash Crash (2010)": ("2010-05-06", "2010-05-10", "Intraday liquidity shock", "High"),
        "Eurozone Debt Crisis (2011)": ("2011-05-02", "2011-10-03", "Sovereign risk + banking stress", "Moderate"),
        "US Debt Ceiling Crisis (2011)": ("2011-07-22", "2011-08-08", "Political standoff / downgrade fears", "Moderate"),
        "Taper Tantrum (2013)": ("2013-05-22", "2013-09-05", "Bond shock / rate repricing", "Moderate"),
        "China Market Crash (2015)": ("2015-06-12", "2015-08-26", "Equity bubble unwind / policy uncertainty", "High"),
        "Brexit Referendum (2016)": ("2016-06-23", "2016-06-27", "FX + risk-off repricing", "Moderate"),
        "Q4 Risk-Off Selloff (2018)": ("2018-09-20", "2018-12-24", "Growth scare / tightening", "Moderate"),
        "Turkey Currency Stress (2018)": ("2018-08-01", "2018-09-01", "TRY depreciation / funding stress", "Severe"),
        "COVID Crash (2020)": ("2020-02-19", "2020-03-23", "Pandemic shock", "Severe"),
        "Oil Price War (2020)": ("2020-03-06", "2020-04-20", "Oil demand shock + price war", "High"),
        "Inflation & Rate Hikes (2022)": ("2022-01-03", "2022-10-12", "Aggressive tightening", "Moderate"),
        "Russia-Ukraine War Shock (2022)": ("2022-02-24", "2022-03-08", "Geopolitical shock / energy risk", "High"),
        "US Banking Turmoil (2023)": ("2023-03-01", "2023-03-31", "Regional bank stress", "Moderate"),


# ----------------------------------------------------------------------
# User-Preset Historical Windows (Classic Episodes)
# NOTE: For pre-2010 scenarios, set the sidebar Start Date sufficiently
# early and choose tickers with long history (major indices / large caps).
# ----------------------------------------------------------------------
"1997 Asian Crisis": ("1997-07-01", "1997-12-31", "Asian currency + contagion (preset window)", "Severe"),
"1998 Russian Default": ("1998-08-01", "1998-10-01", "Russia default / LTCM stress (preset window)", "High"),
"1999 Dotcom Bubble": ("1999-10-01", "2000-03-10", "Late-stage dotcom melt-up to peak (preset window)", "High"),
"2001 9/11 Crash": ("2001-09-01", "2001-11-30", "9/11 shock + aftershock (preset window)", "High"),
"2002 Market Downturn": ("2002-04-01", "2002-10-01", "2002 bear leg / credit stress (preset window)", "Severe"),
"2006 EM Correction": ("2006-05-01", "2006-06-30", "EM correction / risk repricing (preset window)", "Moderate"),
"2008 Global Crisis": ("2008-09-01", "2009-03-09", "GFC acute phase (Lehman -> bottom) (preset window)", "Extreme"),
"2011 Euro Crisis": ("2011-05-01", "2011-10-01", "Euro debt stress (preset window)", "Moderate"),
"2015 China Crash": ("2015-06-01", "2015-08-31", "China equity crash (preset window)", "High"),
"2018 Trade War/TR": ("2018-08-01", "2018-12-31", "Trade-war + EM/TRY stress (preset window)", "Moderate"),
"2020 COVID Crash": ("2020-02-19", "2020-03-23", "COVID crash (preset window)", "Severe"),
"2022 Inflation/Rates": ("2022-01-03", "2022-10-14", "Inflation + rate hikes (preset window)", "Moderate"),
    }

    @staticmethod
    def slice_period(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        try:
            return prices.loc[start:end].copy()
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def scenario_simulation(r: pd.Series, magnitude: float = 0.30, duration: int = 30, vol_mult: float = 3.0, sims: int = 1500) -> Dict[str, Any]:
        base_vol = float(r.std())
        mu = float(r.mean())
        scen_vol = base_vol * float(vol_mult)
        crash_mu = -abs(float(magnitude)) / max(1, duration)
        paths = np.random.normal(crash_mu, scen_vol, size=(sims, duration))
        cum = (1 + paths).prod(axis=1) - 1
        var95 = -np.percentile(cum, 5)
        tail = cum[cum <= -var95]
        cvar95 = -tail.mean() if len(tail) else np.nan
        # drawdown proxy from path
        max_dd = []
        for p in paths:
            cp = (1 + p).cumprod()
            pk = np.maximum.accumulate(cp)
            dd = (cp - pk) / pk
            max_dd.append(dd.min())
        return {
            "Expected Loss %": float(np.mean(cum) * 100),
            "VaR 95% %": float(var95 * 100),
            "CVaR 95% %": float(cvar95 * 100),
            "Avg Max DD %": float(np.mean(max_dd) * 100),
            "Worst Max DD %": float(np.min(max_dd) * 100),
            "Simulations": int(sims)
        }

    @staticmethod
    def comparison_chart(rows: List[Dict[str, Any]]) -> go.Figure:
        df = pd.DataFrame(rows)
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Expected Loss", "VaR vs CVaR", "Drawdown Proxy", "Risk Profile"),
                            specs=[[{"type": "bar"}, {"type": "bar"}],
                                   [{"type": "bar"}, {"type": "scatter"}]],
                            vertical_spacing=0.15, horizontal_spacing=0.12)
        fig.add_trace(go.Bar(x=df["Scenario"], y=df["Expected Loss %"], name="Expected Loss %", text=df["Expected Loss %"].map(lambda v: f"{v:.1f}%"), textposition="auto"), row=1, col=1)
        fig.add_trace(go.Bar(x=df["Scenario"], y=df["VaR 95% %"], name="VaR 95%"), row=1, col=2)
        fig.add_trace(go.Bar(x=df["Scenario"], y=df["CVaR 95% %"], name="CVaR 95%"), row=1, col=2)
        fig.add_trace(go.Bar(x=df["Scenario"], y=df["Avg Max DD %"], name="Avg Max DD %"), row=2, col=1)

        risk = np.abs(df["Avg Max DD %"])
        ret = df["Expected Loss %"]
        fig.add_trace(go.Scatter(x=risk, y=ret, mode="markers+text", text=df["Scenario"], name="Risk Profile"), row=2, col=2)

        fig.update_layout(height=880, template="plotly_white", title="Stress Test Scenario Comparison", title_font_color="#1a237e")
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        return fig

# ==============================================================================
# Efficient Frontier (advanced multi-panel + 3D + CML + risk contributions)
# ==============================================================================
def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    w = weights.reshape(-1, 1)
    port_var = float(w.T @ cov @ w)
    if port_var <= 0:
        return np.zeros(len(weights))
    mrc = cov @ w  # marginal risk contribution
    rc = np.multiply(w, mrc) / port_var
    return rc.flatten()

def efficient_frontier_charts(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.04, n_random: int = 4000, n_frontier: int = 40) -> Dict[str, go.Figure]:
    assets = list(mu.index)
    mu_v = mu.values.astype(float)
    cov_m = cov.values.astype(float)

    # random portfolios
    W = np.random.dirichlet(np.ones(len(assets)), size=n_random)
    rets = W @ mu_v
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, cov_m, W))
    sharpes = (rets - rf) / np.where(vols == 0, np.nan, vols)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=vols, y=rets, mode="markers",
        marker=dict(size=5, color=sharpes, colorbar=dict(title="Sharpe"), showscale=True),
        name="Random Portfolios"
    ))

    # key portfolios
    ew = np.repeat(1/len(assets), len(assets))
    ew_ret = float(ew @ mu_v)
    ew_vol = float(np.sqrt(ew @ cov_m @ ew))
    fig_scatter.add_trace(go.Scatter(x=[ew_vol], y=[ew_ret], mode="markers+text", text=["Equal Weight"], textposition="top center", marker=dict(size=11, symbol="diamond"), name="Equal Weight"))

    # MV frontier if available
    frontier_pts = None
    w_minv = None
    w_maxsh = None
    if OPTIMIZATION_AVAILABLE and EfficientFrontier is not None:
        try:
            ef1 = EfficientFrontier(mu, cov)
            ef1.min_volatility()
            w_minv = np.array([ef1.clean_weights().get(a, 0.0) for a in assets], dtype=float)
            minv_perf = ef1.portfolio_performance(verbose=False, risk_free_rate=rf)

            ef2 = EfficientFrontier(mu, cov)
            ef2.max_sharpe(risk_free_rate=rf)
            w_maxsh = np.array([ef2.clean_weights().get(a, 0.0) for a in assets], dtype=float)
            maxsh_perf = ef2.portfolio_performance(verbose=False, risk_free_rate=rf)

            fig_scatter.add_trace(go.Scatter(x=[minv_perf[1]], y=[minv_perf[0]], mode="markers+text", text=["Min Vol"], textposition="top center", marker=dict(size=12, symbol="x"), name="Min Vol"))
            fig_scatter.add_trace(go.Scatter(x=[maxsh_perf[1]], y=[maxsh_perf[0]], mode="markers+text", text=["Max Sharpe"], textposition="top center", marker=dict(size=12, symbol="star"), name="Max Sharpe"))

            # frontier
            target_rets = np.linspace(np.nanmin(rets), np.nanmax(rets), n_frontier)
            pts = []
            for tr in target_rets:
                ef = EfficientFrontier(mu, cov)
                ef.efficient_return(target_return=float(tr))
                p = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
                pts.append((p[1], p[0]))  # vol, ret
            frontier_pts = np.array(pts)
            fig_scatter.add_trace(go.Scatter(x=frontier_pts[:, 0], y=frontier_pts[:, 1], mode="lines", name="Efficient Frontier", line=dict(width=3)))
        except Exception:
            frontier_pts = None

    # CML: tangent portfolio from random portfolios
    i_star = int(np.nanargmax(sharpes)) if np.isfinite(sharpes).any() else 0
    tan_vol, tan_ret = float(vols[i_star]), float(rets[i_star])
    cml_x = np.linspace(0, max(vols) * 1.05, 50)
    cml_y = rf + (tan_ret - rf) / max(1e-12, tan_vol) * cml_x
    fig_scatter.add_trace(go.Scatter(x=cml_x, y=cml_y, mode="lines", name="Capital Market Line", line=dict(dash="dash")))

    fig_scatter.update_layout(
        height=650, template="plotly_white",
        title="Efficient Frontier (Random Cloud + Key Portfolios + CML)",
        title_font_color="#1a237e",
        xaxis_title="Volatility (σ)", yaxis_title="Return (μ)",
        legend_orientation="h", legend_y=-0.2
    )

    # Sharpe distribution
    fig_sharpe = go.Figure()
    fig_sharpe.add_trace(go.Histogram(x=sharpes[np.isfinite(sharpes)], nbinsx=60, name="Sharpe"))
    fig_sharpe.update_layout(height=420, template="plotly_white", title="Sharpe Ratio Distribution", title_font_color="#1a237e", xaxis_title="Sharpe", yaxis_title="Count")

    # Risk contribution (use max sharpe if available else best sharpe from random)
    w_use = w_maxsh if w_maxsh is not None else W[i_star]
    rc = _risk_contributions(w_use, cov_m)
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Bar(x=assets, y=rc, name="Risk Contribution"))
    fig_rc.update_layout(height=420, template="plotly_white", title="Risk Contribution Decomposition", title_font_color="#1a237e", xaxis_tickangle=45, yaxis_title="Contribution")

    # 3D frontier cloud
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=vols, y=rets, z=sharpes,
        mode="markers",
        marker=dict(size=3, color=sharpes, colorbar=dict(title="Sharpe")),
        name="Random"
    ))
    if frontier_pts is not None:
        # z values for frontier (recompute sharpe)
        zf = (frontier_pts[:, 1] - rf) / np.where(frontier_pts[:, 0] == 0, np.nan, frontier_pts[:, 0])
        fig3d.add_trace(go.Scatter3d(x=frontier_pts[:, 0], y=frontier_pts[:, 1], z=zf, mode="lines", name="Frontier", line=dict(width=6)))
    fig3d.update_layout(height=650, template="plotly_white", title="3D Frontier: Volatility-Return-Sharpe", title_font_color="#1a237e",
                        scene=dict(xaxis_title="Vol", yaxis_title="Ret", zaxis_title="Sharpe"))

    return {"frontier": fig_scatter, "sharpe_dist": fig_sharpe, "risk_contrib": fig_rc, "frontier3d": fig3d}

# ==============================================================================
# Posterior weight stability / turnover diagnostics (BL vs MV vs HRP)
# ==============================================================================
def turnover(w_prev: pd.Series, w_now: pd.Series) -> float:
    w_prev = w_prev.fillna(0.0)
    w_now = w_now.fillna(0.0)
    return float(0.5 * np.abs(w_now - w_prev).sum())

@st.cache_data(ttl=1800, show_spinner=False)
def rolling_weight_paths(
    prices: pd.DataFrame,
    method: str,
    rf: float,
    lookback: int,
    rebalance: str,
    tau: float,
    bl_objective: str,
    mv_objective: str,
    views_payload: Optional[Dict[str, Any]] = None,
    benchmark: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      weights_df: index=rebal dates, columns=assets
      turnover_series: index=rebal dates
    """
    r = prices.pct_change().dropna()
    if len(r) < lookback + 30:
        return pd.DataFrame(), pd.Series(dtype=float)

    # rebalance dates: month-end / week-end / etc.
    if rebalance == "W":
        rebal_dates = r.resample("W-FRI").last().index
    elif rebalance == "Q":
        rebal_dates = r.resample("Q").last().index
    else:
        rebal_dates = r.resample("M").last().index

    rebal_dates = [d for d in rebal_dates if d in r.index]
    if len(rebal_dates) < 3:
        rebal_dates = list(r.index[lookback::21])

    assets = list(prices.columns)
    weights_hist = []
    dates_hist = []

    prev = pd.Series(0.0, index=assets)
    to = []

    for d in rebal_dates:
        end_loc = r.index.get_loc(d)
        if end_loc < lookback:
            continue
        window_prices = prices.iloc[end_loc - lookback:end_loc + 1].copy()
        try:
            pe = PortfolioEngine(window_prices)
            if method == "HRP":
                w, _ = pe.hrp()
            elif method == "BL":
                vp = views_payload or {}
                views_abs = vp.get("abs", {})
                P = vp.get("P", None)
                Q = vp.get("Q", None)
                conf = vp.get("conf", None)
                w, _, _, _ = pe.black_litterman(
                    rf=rf, tau=tau, objective=bl_objective,
                    views_abs=views_abs, P=P, Q=Q,
                    view_confidences=conf,
                    benchmark_returns=benchmark.iloc[end_loc - lookback:end_loc + 1] if benchmark is not None and len(benchmark) > 0 else None
                )
            else:
                w, _ = pe.mean_variance(objective=mv_objective, rf=rf)
        except Exception:
            w = {a: 1/len(assets) for a in assets}

        w_s = pd.Series(w, index=assets).fillna(0.0)
        w_s = w_s / max(1e-12, w_s.sum())
        weights_hist.append(w_s.values)
        dates_hist.append(d)
        to.append(turnover(prev, w_s))
        prev = w_s

    weights_df = pd.DataFrame(weights_hist, index=pd.DatetimeIndex(dates_hist), columns=assets)
    turnover_s = pd.Series(to, index=weights_df.index, name="Turnover")
    return weights_df, turnover_s

# ==============================================================================
# Black-Litterman View Builder (Absolute, Relative, Ranking) + Templates
# ==============================================================================
def build_relative_view_matrix(assets: List[str], views: List[Tuple[str, str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    views: [(A, B, spread)] meaning A outperforms B by spread (annualized).
    Q is annualized view return difference.
    """
    P = np.zeros((len(views), len(assets)))
    Q = np.zeros((len(views),))
    idx = {a: i for i, a in enumerate(assets)}
    for k, (a, b, spread) in enumerate(views):
        if a in idx and b in idx:
            P[k, idx[a]] = 1.0
            P[k, idx[b]] = -1.0
            Q[k] = float(spread)
    return P, Q

def build_ranking_views(rank_assets: List[str], default_spread: float = 0.02) -> List[Tuple[str, str, float]]:
    """
    ranking view: A1 > A2 > A3 ... create pairwise adjacent relative views.
    """
    out = []
    for i in range(len(rank_assets) - 1):
        out.append((rank_assets[i], rank_assets[i+1], float(default_spread)))
    return out

BL_TEMPLATES = {
    "Tech Outperformance (US Large Tech)": {
        "type": "relative",
        "views": [("NVDA", "SPY", 0.03), ("MSFT", "SPY", 0.015), ("AAPL", "SPY", 0.01)]
    },
    "Risk-Off (Bonds Defensive)": {
        "type": "relative",
        "views": [("TLT", "SPY", 0.02), ("IEF", "SPY", 0.015)]
    },
    "Gold Hedge": {
        "type": "absolute",
        "views": {"GLD": 0.10}
    },
    "Emerging Markets Rebound": {
        "type": "absolute",
        "views": {"EEM": 0.12}
    }
}

# ==============================================================================
# UI Helpers
# ==============================================================================
def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:.{digits}f}%"

def fmt_num(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"

def kpi_row(items: List[Tuple[str, str, str]]):
    cols = st.columns(len(items))
    for c, (label, value, delta) in zip(cols, items):
        c.metric(label, value, delta)

def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=True).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )

# ==============================================================================
# Tabs
# ==============================================================================
def tab_market_overview(prices: pd.DataFrame, bench: pd.Series, report: Dict[str, Any]):
    st.markdown('<div class="section-header">Market Overview & Data Quality</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("Assets", str(report.get("assets", prices.shape[1])))
    c2.metric("Rows", str(report.get("rows", len(prices))))
    c3.metric("Range", f'{report.get("start_date","")} → {report.get("end_date","")}')

    if report.get("warnings"):
        st.markdown('<div class="warning-card"><b>Data Warnings</b><br>' + "<br>".join(report["warnings"]) + "</div>", unsafe_allow_html=True)

    # Price chart
    if not prices.empty:
        norm = prices / prices.iloc[0]
        fig = go.Figure()
        for col in norm.columns:
            fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=TICKER_NAME_MAP.get(col, col)))
        fig.update_layout(height=520, template="plotly_white", title="Normalized Price Performance", title_font_color="#1a237e",
                          xaxis_title="Date", yaxis_title="Normalized (Start=1)")
        st.plotly_chart(fig, use_container_width=True)

    # Benchmark
    if bench is not None and not bench.empty:
        st.markdown('<div class="subsection-header">Benchmark</div>', unsafe_allow_html=True)
        figb = go.Figure(go.Scatter(x=bench.index, y=bench/bench.iloc[0], mode="lines", name="^GSPC"))
        figb.update_layout(height=280, template="plotly_white", title="Benchmark Normalized (^GSPC)", title_font_color="#1a237e")
        st.plotly_chart(figb, use_container_width=True)

    # Data quality table
    if report.get("ticker_details"):
        dq = pd.DataFrame(report["ticker_details"]).T
        dq["name"] = dq.index.map(lambda t: TICKER_NAME_MAP.get(t, t))
        dq = dq[["name", "non_na", "na_pct", "start", "end"]]
        st.markdown('<div class="subsection-header">Data Quality Snapshot</div>', unsafe_allow_html=True)
        st.dataframe(dq, use_container_width=True)
        download_csv(dq, "data_quality.csv", "Download data quality CSV")


def tab_portfolio_optimization(prices: pd.DataFrame, bench: pd.Series):
    st.markdown('<div class="section-header">Portfolio Optimization Suite (Equal, MV, HRP, BL)</div>', unsafe_allow_html=True)

    if prices.empty or len(prices.columns) < 2:
        st.info("Select at least 2 assets with sufficient data.")
        return

    pe = PortfolioEngine(prices)

    colA, colB, colC, colD = st.columns([1,1,1,1])
    rf = colA.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.30, value=0.04, step=0.005, format="%.3f", key="opt_rf_annual")
    mv_objective = colB.selectbox("MV Objective", ["max_sharpe", "min_volatility", "max_quadratic_utility"], index=0, key="opt_mv_objective")
    l2 = colC.slider("L2 Regularization (gamma)", 0.0, 2.0, 0.0, 0.05, key="opt_l2_gamma")
    bl_tau = colD.slider("BL Tau (confidence scaling)", 0.01, 0.30, 0.05, 0.01, key="opt_bl_tau")

    st.markdown('<div class="subsection-header">Run Optimization</div>', unsafe_allow_html=True)
    run_cols = st.columns([1,1,1,1])
    do_equal = run_cols[0].button("Run Equal Weight", key="opt_run_equal")
    do_mv = run_cols[1].button("Run Mean-Variance (MV)", key="opt_run_mv")
    do_hrp = run_cols[2].button("Run HRP", key="opt_run_hrp")
    do_bl = run_cols[3].button("Run Black-Litterman (BL)", key="opt_run_bl")
    adv_cols = st.columns([1,1,1,1,1])
    do_cla = adv_cols[0].button("Run CLA", key="opt_run_cla")
    do_semi = adv_cols[1].button("Run Semivariance", key="opt_run_semivariance")
    do_cvar = adv_cols[2].button("Run CVaR (ES)", key="opt_run_cvar")
    do_cdar = adv_cols[3].button("Run CDaR (Drawdown)", key="opt_run_cdar")
    do_all = adv_cols[4].button("Run ALL (Institutional)", key="opt_run_all")


    st.markdown('<div class="subsection-header">🧮 PyPortfolioOpt Risk Model Matrix (risk_models.risk_matrix)</div>', unsafe_allow_html=True)
    if not OPTIMIZATION_AVAILABLE:
        st.info("PyPortfolioOpt is not available, so risk_models.risk_matrix diagnostics are disabled.")
    else:
        with st.expander("Covariance / Correlation Matrix + Round Percentage (Donut) Chart", expanded=False):
            rm_col1, rm_col2, rm_col3, rm_col4 = st.columns([1,1,1,1])

            rm_method = rm_col1.selectbox(
                "Method",
                [
                    "sample_cov",
                    "semicovariance",
                    "exp_cov",
                    "ledoit_wolf",
                    "ledoit_wolf_constant_variance",
                    "ledoit_wolf_single_factor",
                    "ledoit_wolf_constant_correlation",
                    "oracle_approximating",
                ],
                index=0,
                key="opt_risk_matrix_method"
            )

            rm_view = rm_col2.selectbox("View", ["Correlation", "Covariance"], index=0, key="opt_risk_matrix_view")
            rm_freq = rm_col3.number_input("Frequency (annualization)", min_value=1, max_value=3650, value=252, step=1, key="opt_risk_matrix_freq")
            rm_span = rm_col4.slider("exp_cov span", 10, 500, 180, 5, key="opt_risk_matrix_span")

            rm_returns_data = st.checkbox("Input is returns (returns_data=True)", value=False, key="opt_risk_matrix_returns_data")

            rm_input = prices if not rm_returns_data else prices.pct_change().dropna()

            rm_kwargs = {"frequency": int(rm_freq)}
            if rm_method == "exp_cov":
                rm_kwargs["span"] = int(rm_span)

            try:
                S_rm = risk_models.risk_matrix(rm_input, method=rm_method, returns_data=rm_returns_data, **rm_kwargs)
                if not isinstance(S_rm, pd.DataFrame):
                    S_rm = pd.DataFrame(S_rm, index=prices.columns, columns=prices.columns)
            except Exception as e:
                st.warning(f"Risk matrix computation failed: {e}")
            else:
                # Build display matrix
                if rm_view == "Correlation":
                    S = S_rm.values.astype(float)
                    d = np.sqrt(np.clip(np.diag(S), 1e-18, None))
                    corr = S / np.outer(d, d)
                    corr = np.clip(corr, -1.0, 1.0)
                    M = pd.DataFrame(corr, index=S_rm.index, columns=S_rm.columns)
                    zmin, zmax = -1.0, 1.0
                    colorscale = "RdBu_r"
                    title = f"Correlation Matrix (derived from {rm_method})"
                else:
                    M = S_rm.copy()
                    zmin, zmax = None, None
                    colorscale = "Blues"
                    title = f"Covariance Matrix (annualized, {rm_method})"

                labels = [TICKER_NAME_MAP.get(t, t) for t in M.columns]
                fig_hm = go.Figure(data=go.Heatmap(
                    z=M.values,
                    x=labels,
                    y=labels,
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z:.6f}<extra></extra>"
                ))
                fig_hm.update_layout(
                    height=650,
                    template="plotly_white",
                    title=title,
                    title_font_color="#1a237e",
                    xaxis_tickangle=-45,
                    font_color="#424242"
                )
                st.plotly_chart(fig_hm, use_container_width=True)

                st.markdown('<div class="subsection-header">🟣 Round Percentage Chart (Risk Contribution)</div>', unsafe_allow_html=True)

                cov_for_rc = S_rm.copy()
                w_last = st.session_state.get("last_weights", None)

                try:
                    if isinstance(w_last, dict) and len(w_last) > 0:
                        wv = pd.Series(w_last).reindex(cov_for_rc.columns).fillna(0.0).astype(float)
                        w_label = f"using last strategy weights ({st.session_state.get('last_strategy','')})"
                    else:
                        wv = pd.Series(1.0 / len(cov_for_rc.columns), index=cov_for_rc.columns).astype(float)
                        w_label = "using equal weights (no optimized weights yet)"

                    port_var = float(wv.values @ cov_for_rc.values @ wv.values)
                    if not np.isfinite(port_var) or port_var <= 0:
                        raise ValueError("Non-positive portfolio variance (cannot compute contributions).")

                    mc = cov_for_rc.values @ wv.values
                    rc = wv.values * mc / port_var
                    rc = np.clip(rc, 0, None)
                    if rc.sum() <= 0:
                        raise ValueError("Zero/negative total contribution.")

                    rc_pct = rc / rc.sum() * 100.0
                    rc_df = pd.DataFrame({"Asset": cov_for_rc.columns, "RiskContributionPct": rc_pct})
                    rc_df["AssetName"] = rc_df["Asset"].map(lambda t: TICKER_NAME_MAP.get(t, t))
                    rc_df = rc_df.sort_values("RiskContributionPct", ascending=False)

                    fig_donut = px.pie(rc_df, values="RiskContributionPct", names="AssetName", hole=0.45)
                    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
                    fig_donut.update_layout(
                        height=520,
                        template="plotly_white",
                        title=f"Risk Contribution (% of portfolio variance) — {w_label}",
                        title_font_color="#1a237e",
                        showlegend=False,
                        font_color="#424242"
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                    with st.expander("Show risk contribution table", expanded=False):
                        show_tbl = rc_df.copy()
                        show_tbl["RiskContributionPct"] = show_tbl["RiskContributionPct"].map(lambda x: f"{x:.2f}%")
                        st.dataframe(show_tbl[["AssetName", "Asset", "RiskContributionPct"]], use_container_width=True)

                except Exception as e:
                    st.info(f"Risk contribution donut unavailable: {e}")


    # BL views from session (if user created in BL tab)
    views_payload = st.session_state.get("bl_views_payload", None)

    def show_solution(name: str, w: Dict[str, float], perf: Tuple[float, float, float], prior: Optional[pd.Series]=None, post: Optional[pd.Series]=None):
        st.markdown(f'<div class="card"><div class="subsection-header">{name} Results</div>', unsafe_allow_html=True)

        # weights table
        wdf = pd.DataFrame({"Asset": list(w.keys()), "Weight": list(w.values())})
        wdf["AssetName"] = wdf["Asset"].map(lambda t: TICKER_NAME_MAP.get(t, t))
        wdf = wdf.sort_values("Weight", ascending=False)
        st.dataframe(wdf, use_container_width=True)
        # weights wheel (donut) chart
        try:
            pie_df = wdf[["AssetName", "Weight"]].copy()
            pie_df = pie_df.sort_values("Weight", ascending=False)
            if pie_df.shape[0] > 12:
                top = pie_df.head(12)
                others_w = float(pie_df["Weight"].iloc[12:].sum())
                pie_df = pd.concat([top, pd.DataFrame([{ "AssetName": "Others", "Weight": others_w }])], ignore_index=True)
            fig_w = px.pie(pie_df, values="Weight", names="AssetName", hole=0.45, title="Portfolio Weights (Wheel %)")
            fig_w.update_layout(height=420, title_font_color="#1f2937")
            st.plotly_chart(fig_w, use_container_width=True)
        except Exception:
            pass

        kpi_row([
            ("Annual Return", fmt_pct(perf[0], 2), ""),
            ("Annual Vol", fmt_pct(perf[1], 2), ""),
            ("Sharpe", fmt_num(perf[2], 3), "")
        ])

        # store in session
        st.session_state["last_weights"] = w
        st.session_state["last_perf"] = perf
        st.session_state["last_strategy"] = name

        # posterior vs prior
        if prior is not None and post is not None:
            comp = pd.DataFrame({"Prior": prior, "Posterior": post}).dropna()
            comp["Δ"] = comp["Posterior"] - comp["Prior"]
            comp = comp.sort_values("Δ", ascending=False)
            st.markdown('<div class="subsection-header">BL: Prior vs Posterior Returns</div>', unsafe_allow_html=True)
            st.dataframe(comp, use_container_width=True)
            download_csv(comp, "bl_prior_posterior.csv", "Download prior/posterior CSV")

            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp.index, y=comp["Prior"], name="Prior"))
            fig.add_trace(go.Bar(x=comp.index, y=comp["Posterior"], name="Posterior"))
            fig.update_layout(barmode="group", height=420, template="plotly_white", title="Equilibrium vs Adjusted Returns", title_font_color="#1a237e", xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


        # Discrete allocation (integer shares) - optional
        with st.expander("Discrete Allocation (Integer Shares) — optional", expanded=False):
            pv = st.number_input("Total portfolio value ($)", min_value=1000.0, value=1_000_000.0, step=25_000.0, key=f"da_pv_{name}")
            alloc_df, leftover = pe.discrete_allocation(w, portfolio_value=pv)
            if alloc_df is not None and not alloc_df.empty:
                st.dataframe(alloc_df, use_container_width=True)
                st.caption(f"Leftover cash: ${leftover:,.2f}")
                download_csv(alloc_df, "discrete_allocation.csv", "Download discrete allocation CSV")
            else:
                st.info("Discrete allocation unavailable (requires PyPortfolioOpt discrete_allocation + solver).")

        st.markdown("</div>", unsafe_allow_html=True)

    if do_equal:
        w, perf = pe.equal_weight()
        show_solution("Equal Weight", w, perf)

    if do_mv:
        w, perf = pe.mean_variance(objective=mv_objective, rf=rf, gamma=l2 if l2 > 0 else None)
        show_solution(f"Mean-Variance ({mv_objective})", w, perf)

    if do_hrp:
        w, perf = pe.hrp()
        show_solution("HRP", w, perf)

    if do_bl:
        w, perf, prior, post = pe.black_litterman(
            rf=rf, tau=bl_tau, objective="max_sharpe",
            views_abs=(views_payload or {}).get("abs", None),
            P=(views_payload or {}).get("P", None),
            Q=(views_payload or {}).get("Q", None),
            view_confidences=(views_payload or {}).get("conf", None),
            benchmark_returns=bench
        )
        show_solution("Black-Litterman", w, perf, prior, post)

    # --- Advanced PyPortfolioOpt strategy runners (version-robust) ---
    if do_cla:
        w, perf = pe.cla()
        show_solution("CLA (Critical Line Algorithm)", w, perf)

    if do_semi:
        w, perf = pe.semivariance(benchmark=0.0)
        show_solution("Semivariance (Downside Risk)", w, perf)

    if do_cvar:
        w, perf = pe.cvar()
        show_solution("Efficient CVaR / ES", w, perf)

    if do_cdar:
        w, perf = pe.cdar()
        show_solution("Efficient CDaR", w, perf)

    if do_all:
        # Run a comprehensive institutional set and rank by Sharpe (then by Vol)
        results = {}
        runners = [
            ("Equal Weight", lambda: pe.equal_weight()),
            ("Mean-Variance (max_sharpe)", lambda: pe.mean_variance(objective="max_sharpe", rf=rf, gamma=l2 if l2 > 0 else None)),
            ("Mean-Variance (min_volatility)", lambda: pe.mean_variance(objective="min_volatility", rf=rf, gamma=l2 if l2 > 0 else None)),
            ("Mean-Variance (max_quadratic_utility)", lambda: pe.mean_variance(objective="max_quadratic_utility", rf=rf, gamma=l2 if l2 > 0 else None)),
            ("HRP", lambda: pe.hrp()),
            ("CLA", lambda: pe.cla()),
            ("Semivariance", lambda: pe.semivariance(benchmark=0.0)),
            ("CVaR / ES", lambda: pe.cvar()),
            ("CDaR", lambda: pe.cdar()),
            ("Black-Litterman (max_sharpe)", lambda: pe.black_litterman(
                rf=rf, tau=bl_tau, objective="max_sharpe",
                views_abs=(views_payload or {}).get("abs", None),
                P=(views_payload or {}).get("P", None),
                Q=(views_payload or {}).get("Q", None),
                view_confidences=(views_payload or {}).get("conf", None),
                benchmark_returns=bench
            )[:2])
        ]

        ret = prices.pct_change().dropna()
        for name, fn in runners:
            try:
                w_i, perf_i = fn()
                pr_i = portfolio_returns_from_weights(ret, w_i)
                ve = AdvancedVaREngine(pr_i)
                hist = ve.historical(0.95, 1)
                results[name] = {
                    "Ann.Return": float(perf_i[0]),
                    "Ann.Vol": float(perf_i[1]),
                    "Sharpe": float(perf_i[2]),
                    "MaxDD": float(max_drawdown(pr_i)),
                    "VaR95(1d)": float(hist.get("VaR_1d", np.nan)),
                    "ES95": float(hist.get("CVaR", np.nan)),
                }
            except Exception:
                continue

        if results:
            res_df = pd.DataFrame(results).T
            res_df = res_df.sort_values(["Sharpe", "Ann.Vol"], ascending=[False, True])
            st.markdown('<div class="subsection-header">Institutional Strategy Comparison</div>', unsafe_allow_html=True)
            st.dataframe(res_df.style.format({
                "Ann.Return": "{:.2%}",
                "Ann.Vol": "{:.2%}",
                "Sharpe": "{:.3f}",
                "MaxDD": "{:.2%}",
                "VaR95(1d)": "{:.2%}",
                "ES95": "{:.2%}",
            }), use_container_width=True)
            download_csv(res_df, "institutional_strategy_comparison.csv", "Download Strategy Comparison CSV")

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=res_df.index, y=res_df["Sharpe"], name="Sharpe"))
            fig1.update_layout(height=420, title="Sharpe by Strategy", xaxis_tickangle=35)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=res_df.index, y=res_df["MaxDD"], name="MaxDD"))
            fig2.update_layout(height=420, title="Max Drawdown by Strategy", xaxis_tickangle=35)
            st.plotly_chart(fig2, use_container_width=True)

            best_name = res_df.index[0]
            try:
                best_runner = dict(runners).get(best_name, None)
                if best_runner:
                    w_best, perf_best = best_runner()
                    show_solution(f"BEST → {best_name}", w_best, perf_best)
            except Exception:
                pass
        else:
            st.warning("No strategies produced a result (check data, solvers, or PyPortfolioOpt install).")

    # Mini-backtest if a solution exists
    if "last_weights" in st.session_state and st.session_state.get("last_weights"):
        st.markdown('<div class="subsection-header">Mini Backtest (Buy & Hold)</div>', unsafe_allow_html=True)
        w = st.session_state["last_weights"]
        ret = prices.pct_change().dropna()
        pr = portfolio_returns_from_weights(ret, w)
        cum = (1 + pr).cumprod()
        fig = go.Figure(go.Scatter(x=cum.index, y=cum, mode="lines", name="Portfolio"))
        fig.update_layout(height=320, template="plotly_white", title="Cumulative Growth of $1", title_font_color="#1a237e")
        st.plotly_chart(fig, use_container_width=True)

        # Factsheet table (Institutional Strategy Factsheet)
        apm = AdvancedPerformanceMetrics(pr, bench.pct_change().dropna() if bench is not None and not bench.empty else None, rf_annual=rf)
        mdf = apm.compute()
        key = ["CAGR", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "VaR (95%)", "CVaR/ES (95%)", "CAPM Alpha (ann)", "CAPM Beta", "Information Ratio"]
        facts = mdf[mdf["Metric"].isin(key)].copy()
        facts["Value"] = facts["Value"].astype(float)
        facts["Display"] = facts.apply(lambda r: fmt_pct(r["Value"]) if "Return" in r["Metric"] or "CAGR" in r["Metric"] or "Drawdown" in r["Metric"] or "VaR" in r["Metric"] or "CVaR" in r["Metric"] or "Alpha" in r["Metric"] else fmt_num(r["Value"]), axis=1)
        st.markdown('<div class="card"><b>Institutional Strategy Factsheet</b></div>', unsafe_allow_html=True)
        st.dataframe(facts[["Metric", "Display"]], use_container_width=True)
        download_csv(facts[["Metric", "Value"]], "strategy_factsheet.csv", "Download factsheet CSV")

def tab_advanced_var(prices: pd.DataFrame, bench: pd.Series):
    st.markdown('<div class="section-header">Advanced VaR / CVaR / ES Lab</div>', unsafe_allow_html=True)
    if "last_weights" not in st.session_state or not st.session_state.get("last_weights"):
        st.info("Run an optimization first (or Equal Weight) to generate a portfolio for VaR analysis.")
        return

    w = st.session_state["last_weights"]
    ret = prices.pct_change().dropna()
    pr = portfolio_returns_from_weights(ret, w)
    var_engine = AdvancedVaREngine(pr)

    c1, c2, c3 = st.columns([1,1,1])
    cl = c1.slider("Confidence Level", 0.90, 0.995, 0.95, 0.005)
    hp = c2.slider("Holding Period (days)", 1, 20, 5, 1)
    sims = c3.selectbox("Monte Carlo sims", [2000, 5000, 8000, 12000], index=2)

    dfm = var_engine.compare_methods(cl, hp)
    st.dataframe(dfm, use_container_width=True)
    download_csv(dfm, "var_methods_comparison.csv", "Download VaR comparison CSV")

    fig = var_engine.chart_distribution(cl)
    st.plotly_chart(fig, use_container_width=True)

    fig_surf = var_engine.var_surface()
    st.plotly_chart(fig_surf, use_container_width=True)

    # $ impact
    invest = st.number_input("Investment Amount ($)", min_value=1000.0, max_value=1e9, value=1_000_000.0, step=50_000.0, key="var_investment_amount")
    hist = var_engine.historical(cl, hp)
    if np.isfinite(hist.get("VaR_hp", np.nan)):
        st.markdown('<div class="info-card"><b>Dollar VaR/ES</b><br>'
                    f'VaR({int(cl*100)}%, {hp}d): <b>${invest*hist["VaR_hp"]:,.0f}</b><br>'
                    f'CVaR/ES({int(cl*100)}%): <b>${invest*hist["CVaR"]:,.0f}</b></div>', unsafe_allow_html=True)




    st.markdown('<div class="subsection-header">Relative VaR / CVaR / ES (Portfolio vs Benchmark)</div>', unsafe_allow_html=True)
    if bench is None or (isinstance(bench, pd.Series) and bench.dropna().empty):
        st.info("Benchmark series not available for relative risk. Select a benchmark in the sidebar.")
    else:
        rel_tabs = st.tabs(["Portfolio vs Benchmark", "Active (P − B)", "Comparative Charts"])
        try:
            br = bench.pct_change().dropna().rename("Benchmark")
            aligned = pd.concat([pr.rename("Portfolio"), br], axis=1).dropna()
            pr_a = aligned["Portfolio"]
            br_a = aligned["Benchmark"]

            with rel_tabs[0]:
                ve_p = AdvancedVaREngine(pr_a)
                ve_b = AdvancedVaREngine(br_a)
                df_p = ve_p.compare_methods(cl, hp).set_index("Method")
                df_b = ve_b.compare_methods(cl, hp).set_index("Method")
                comp = pd.DataFrame({
                    "VaR_hp_Port": df_p["VaR_hp"],
                    "VaR_hp_Bench": df_b["VaR_hp"],
                    "Δ VaR_hp (P−B)": df_p["VaR_hp"] - df_b["VaR_hp"],
                    "CVaR_Port": df_p["CVaR"],
                    "CVaR_Bench": df_b["CVaR"],
                    "Δ CVaR (P−B)": df_p["CVaR"] - df_b["CVaR"],
                }).reset_index()
                st.dataframe(comp.style.format({
                    "VaR_hp_Port": "{:.2%}",
                    "VaR_hp_Bench": "{:.2%}",
                    "Δ VaR_hp (P−B)": "{:.2%}",
                    "CVaR_Port": "{:.2%}",
                    "CVaR_Bench": "{:.2%}",
                    "Δ CVaR (P−B)": "{:.2%}",
                }), use_container_width=True)
                download_csv(comp, "relative_var_cvar_table.csv", "Download Relative VaR/CVaR Table")

            with rel_tabs[1]:
                active = (pr_a - br_a).rename("Active")
                ve_a = AdvancedVaREngine(active)
                df_a = ve_a.compare_methods(cl, hp)
                st.markdown('<div class="info-card"><b>Active risk</b><br>Active = Portfolio return − Benchmark return</div>', unsafe_allow_html=True)
                st.dataframe(df_a, use_container_width=True)
                fig_a = ve_a.chart_distribution(cl)
                st.plotly_chart(fig_a, use_container_width=True)

            with rel_tabs[2]:
                # comparative charts (VaR_hp and CVaR) across methods
                ve_p = AdvancedVaREngine(pr_a)
                ve_b = AdvancedVaREngine(br_a)
                df_p = ve_p.compare_methods(cl, hp).set_index("Method")
                df_b = ve_b.compare_methods(cl, hp).set_index("Method")
                methods = df_p.index.intersection(df_b.index)

                fig_var = go.Figure()
                fig_var.add_trace(go.Bar(x=methods, y=df_p.loc[methods, "VaR_hp"], name="Portfolio"))
                fig_var.add_trace(go.Bar(x=methods, y=df_b.loc[methods, "VaR_hp"], name="Benchmark"))
                fig_var.update_layout(barmode="group", height=420, title=f"VaR({int(cl*100)}%, {hp}d) — Portfolio vs Benchmark", xaxis_tickangle=35)
                st.plotly_chart(fig_var, use_container_width=True)

                fig_es = go.Figure()
                fig_es.add_trace(go.Bar(x=methods, y=df_p.loc[methods, "CVaR"], name="Portfolio"))
                fig_es.add_trace(go.Bar(x=methods, y=df_b.loc[methods, "CVaR"], name="Benchmark"))
                fig_es.update_layout(barmode="group", height=420, title=f"CVaR/ES({int(cl*100)}%) — Portfolio vs Benchmark", xaxis_tickangle=35)
                st.plotly_chart(fig_es, use_container_width=True)

                fig_delta = go.Figure()
                fig_delta.add_trace(go.Bar(x=methods, y=(df_p.loc[methods, "VaR_hp"] - df_b.loc[methods, "VaR_hp"]), name="Δ VaR_hp (P−B)"))
                fig_delta.update_layout(height=420, title="Relative VaR Difference (P−B)", xaxis_tickangle=35)
                st.plotly_chart(fig_delta, use_container_width=True)
        except Exception as _e:
            st.warning(f"Relative risk failed: {_e}")

def tab_stress_testing(prices: pd.DataFrame):
    st.markdown('<div class="section-header">Stress Testing Lab (Historical + Custom + Shock Simulator)</div>', unsafe_allow_html=True)

    if prices is None or prices.empty:
        st.markdown('<div class="warning-card">⚠️ No price data available.</div>', unsafe_allow_html=True)
        return

    # --- weights fallback (do NOT block the tab) ---
    w = st.session_state.get("last_weights", {})
    if not w:
        st.info("No optimized weights found in session_state. Using Equal-Weight portfolio for stress testing.")
        w = {c: 1.0 / prices.shape[1] for c in prices.columns}

    # --- Target selector (Portfolio vs Instrument) ---
    target_options = ["Portfolio (Weights)"] + list(prices.columns)
    target = st.selectbox(
        "Select target for stress analysis (Portfolio or single instrument)",
        options=target_options,
        index=0,
        key="stress_target_select_v2"
    )

    # Build target series
    start_str = prices.index.min().strftime("%Y-%m-%d")
    end_str = prices.index.max().strftime("%Y-%m-%d")

    if target == "Portfolio (Weights)":
        target_price = compute_portfolio_index(prices, w)
        target_ret = target_price.pct_change().dropna()
        target_label = "Portfolio"
    else:
        target_price = prices[target].dropna().copy()
        target_ret = target_price.pct_change().dropna()
        target_label = str(target)

    # -----------------------
    # A) Historical crises
    # -----------------------
    st.markdown('<div class="subsection-header">Historical Crises</div>', unsafe_allow_html=True)

    default_events = [
        "COVID Crash (2020)",
        "Inflation & Rate Hikes (2022)",
        "Global Financial Crisis (2008-2009)",
        "Dot-com Bubble Burst (2000-2002)",
    ]
    default_events = [e for e in default_events if e in StressTestEngine.HISTORICAL]
    if not default_events:
        default_events = list(StressTestEngine.HISTORICAL.keys())[:3]

    # --- Scenario availability (auto label: Available / Not Available) ---
    data_start = prices.index.min()
    data_end = prices.index.max()

    scenario_rows = []
    display_options = []
    display_to_name = {}

    for _name, (_s, _e, _desc, _sev) in StressTestEngine.HISTORICAL.items():
        _s_dt = pd.to_datetime(_s)
        _e_dt = pd.to_datetime(_e)

        # Overlap with current app window
        _in_window = bool((_e_dt >= data_start) and (_s_dt <= data_end))

        # Also check target data availability (more realistic for selected portfolio/instrument)
        _overlap_pts = 0
        if _in_window:
            try:
                _overlap_pts = int(target_price.loc[_s:_e].dropna().shape[0])
            except Exception:
                _overlap_pts = 0

        _available = bool(_in_window and (_overlap_pts >= 10))
        _status = "Available" if _available else "Not Available"

        _label = f"✅ {_name}" if _available else f"⛔ {_name}"
        _label = f"{_label}  ({_s} → {_e})"

        display_options.append(_label)
        display_to_name[_label] = _name

        scenario_rows.append({
            "Scenario": _name,
            "Start": _s,
            "End": _e,
            "Severity": _sev,
            "Status": _status,
            "WindowOverlap": "Yes" if _in_window else "No",
            "OverlapPoints": _overlap_pts,
        })

    # Table view (quick availability scan)
    df_scenarios = pd.DataFrame(scenario_rows)
    if not df_scenarios.empty:
        # Put available scenarios on top
        df_scenarios["_sort"] = (df_scenarios["Status"] == "Available").astype(int)
        df_scenarios = df_scenarios.sort_values(["_sort", "Severity", "Scenario"], ascending=[False, True, True]).drop(columns=["_sort"])

        st.dataframe(
            df_scenarios,
            use_container_width=True,
            height=320
        )

    # Map defaults to display labels
    default_display = []
    for _ev in default_events:
        for _lab, _nm in display_to_name.items():
            if _nm == _ev:
                default_display.append(_lab)

    if not default_display:
        # fallback: pick first few available labels
        default_display = [lab for lab in display_options if lab.startswith("✅")][:3] or display_options[:3]

    chosen_display = st.multiselect(
        "Select historical stress events (auto-labeled by availability)",
        options=display_options,
        default=default_display,
        key="stress_hist_events_v3"
    )

    chosen = [display_to_name[x] for x in chosen_display]

    rows = []
    for name in chosen:
        s, e, desc, sev = StressTestEngine.HISTORICAL[name]
        tp = target_price.loc[s:e].dropna()
        if tp.empty or len(tp) < 10:
            continue
        tr = tp.pct_change().dropna()
        rows.append({
            "Scenario": name,
            "Target": target_label,
            "Period": f"{s} → {e}",
            "Severity": sev,
            "Return %": float(((1 + tr).prod() - 1) * 100),
            "Max DD %": float(max_drawdown(tr) * 100),
            "Vol (ann) %": float(annualize_vol(tr) * 100),
            "Description": desc
        })

    if rows:
        dfh = pd.DataFrame(rows).sort_values("Max DD %")
        st.dataframe(dfh, use_container_width=True, height=360)
        download_csv(dfh, f"historical_stress_{target_label}.csv", "Download historical stress CSV")

        # Chart: Max drawdown vs scenario
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dfh["Scenario"], y=dfh["Max DD %"], name="Max DD %"))
        fig.update_layout(
            height=420,
            template="plotly_white",
            title=f"{target_label}: Historical Stress - Max Drawdown",
            title_font_color="#1a237e",
            xaxis_tickangle=45,
            margin=dict(l=10, r=10, t=60, b=90),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Chart: Return vs Vol scatter
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dfh["Vol (ann) %"],
            y=dfh["Return %"],
            mode="markers+text",
            text=dfh["Scenario"],
            textposition="top center",
            marker=dict(size=12),
            name="Scenario"
        ))
        fig2.update_layout(
            height=420,
            template="plotly_white",
            title=f"{target_label}: Historical Stress - Return vs Volatility",
            title_font_color="#1a237e",
            xaxis_title="Volatility (Ann, %)",
            yaxis_title="Return (%)",
            margin=dict(l=10, r=10, t=60, b=60),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.markdown('<div class="info-card">ℹ️ No overlapping data found for selected events and target.</div>', unsafe_allow_html=True)

    st.divider()

    # -----------------------
    # B) Custom Shock Simulation (existing MC summary)
    # -----------------------
    st.markdown('<div class="subsection-header">Custom Shock Simulation (Monte-Carlo)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        magnitude = st.slider("Shock magnitude (total loss)", 0.05, 0.80, 0.25, 0.01, key="shock_mag_v2")
    with c2:
        duration = st.slider("Duration (days)", 5, 252, 30, 1, key="shock_dur_v2")
    with c3:
        volm = st.slider("Volatility multiplier", 1.0, 6.0, 2.5, 0.1, key="shock_volm_v2")
    with c4:
        sims = st.number_input("Simulations", 200, 20000, 2000, 200, key="shock_sims_v2")

    if st.button("Run Shock Simulation", type="primary", use_container_width=True, key="run_shock_v2"):
        res = StressTestEngine.scenario_simulation(target_ret, magnitude=magnitude, duration=duration, vol_mult=volm, sims=int(sims))
        st.markdown(
            '<div class="success-card"><b>Custom Shock Results</b><br>' +
            "<br>".join([
                f"{k}: <b>{v:.4f}</b>" if isinstance(v, (int, float, np.floating)) else f"{k}: <b>{v}</b>"
                for k, v in res.items()
            ]) +
            "</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # -----------------------
    # C) Interactive Shock Simulator (NEW) with fan chart + distribution
    # -----------------------
    st.markdown('<div class="subsection-header">Interactive Stress Shock Simulator (Fan Chart + Distribution)</div>', unsafe_allow_html=True)

    with st.expander("💥 Shock Simulator Controls", expanded=True):
        colA, colB, colC, colD = st.columns(4)
        with colA:
            shock_type = st.selectbox(
                "Shock type",
                ["Instant Drop + Recovery", "Linear Drawdown", "Volatility Spike", "Liquidity Crunch"],
                key="shock_type_v2"
            )
        with colB:
            shock_mag = st.slider("Shock magnitude", 0.05, 0.90, 0.30, 0.01, key="shock_mag2_v2")
        with colC:
            shock_days = st.slider("Shock horizon (days)", 5, 504, 60, 1, key="shock_days_v2")
        with colD:
            simN = st.number_input("MC paths", 200, 50000, 5000, 200, key="shock_paths_v2")

        colE, colF, colG, colH = st.columns(4)
        with colE:
            vol_mult = st.slider("Vol multiplier", 0.5, 8.0, 2.0, 0.1, key="shock_volmult_v2")
        with colF:
            recovery_speed = st.select_slider("Recovery speed", ["slow", "medium", "fast"], value="medium", key="shock_rec_v2")
        with colG:
            tail_fat = st.slider("Tail fatness (t-df inverse)", 3, 30, 8, 1, key="shock_tdf_v2")
        with colH:
            seed = st.number_input("Random seed", 0, 10_000_000, 42, 1, key="shock_seed_v2")

        st.caption("This simulator generates *hypothetical* forward paths based on the historical volatility of the selected target.")

    if st.button("⚡ Run Shock Simulator", use_container_width=True, key="run_shock_sim_v2"):
        np.random.seed(int(seed))

        base_mu = float(target_ret.mean())
        base_sigma = float(target_ret.std())
        if not np.isfinite(base_sigma) or base_sigma <= 0:
            st.markdown('<div class="warning-card">⚠️ Insufficient volatility estimate for simulation.</div>', unsafe_allow_html=True)
            return

        # t innovations for fatter tails
        df_t = int(max(3, tail_fat))
        eps = stats.t.rvs(df_t, size=(int(simN), int(shock_days)))
        eps = eps / np.std(eps)  # scale to unit std

        mu = base_mu
        sigma = base_sigma * float(vol_mult)

        paths_ret = np.zeros((int(simN), int(shock_days)))

        if shock_type == "Instant Drop + Recovery":
            # Day 1: big drop; subsequent mean reversion drift depending on recovery_speed
            paths_ret[:, 0] = -float(shock_mag)
            if recovery_speed == "fast":
                rec_mu = abs(shock_mag) / max(shock_days, 1) * 0.9
            elif recovery_speed == "medium":
                rec_mu = abs(shock_mag) / max(shock_days, 1) * 0.6
            else:
                rec_mu = abs(shock_mag) / max(shock_days, 1) * 0.3

            if shock_days > 1:
                paths_ret[:, 1:] = (rec_mu + sigma * eps[:, 1:])

        elif shock_type == "Linear Drawdown":
            drift = -float(shock_mag) / max(shock_days, 1)
            paths_ret[:, :] = drift + sigma * eps

        elif shock_type == "Volatility Spike":
            # zero drift, high sigma
            paths_ret[:, :] = 0.0 + sigma * eps

        else:  # Liquidity Crunch
            # increasing sigma + negative drift
            drift = -float(shock_mag) / max(shock_days, 1) * 0.6
            sigma_path = np.linspace(base_sigma, sigma, int(shock_days))
            for j in range(int(shock_days)):
                paths_ret[:, j] = drift + sigma_path[j] * eps[:, j]

        # Convert to index paths
        idx_paths = 100.0 * np.cumprod(1.0 + paths_ret, axis=1)
        final_vals = idx_paths[:, -1]
        final_rets = final_vals / 100.0 - 1.0

        q = np.percentile(idx_paths, [5, 25, 50, 75, 95], axis=0)
        x = np.arange(1, int(shock_days) + 1)

        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(x=x, y=q[2], mode="lines", name="Median", line=dict(width=3)))
        fig_fan.add_trace(go.Scatter(x=x, y=q[1], mode="lines", name="25%", line=dict(width=1, dash="dash")))
        fig_fan.add_trace(go.Scatter(x=x, y=q[3], mode="lines", name="75%", line=dict(width=1, dash="dash")))
        fig_fan.add_trace(go.Scatter(x=x, y=q[0], mode="lines", name="5%", line=dict(width=1, dash="dot")))
        fig_fan.add_trace(go.Scatter(x=x, y=q[4], mode="lines", name="95%", line=dict(width=1, dash="dot")))

        fig_fan.update_layout(
            height=520,
            template="plotly_white",
            title=f"{target_label}: Shock Simulator Fan Chart ({shock_type})",
            title_font_color="#1a237e",
            xaxis_title="Simulation Day",
            yaxis_title="Index Level (Base 100)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=70, b=60),
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        # Distribution + risk metrics
        var95 = -np.percentile(final_rets, 5)
        tail = final_rets[final_rets <= np.percentile(final_rets, 5)]
        cvar95 = -float(np.mean(tail)) if len(tail) > 0 else float("nan")

        colm1, colm2, colm3, colm4 = st.columns(4)
        colm1.metric("Expected final return", f"{np.mean(final_rets)*100:.2f}%")
        colm2.metric("VaR 95% (horizon)", f"{var95*100:.2f}%")
        colm3.metric("CVaR 95% (horizon)", f"{cvar95*100:.2f}%")
        colm4.metric("Worst path return", f"{np.min(final_rets)*100:.2f}%")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=final_rets * 100, nbinsx=60, name="Final return (%)", opacity=0.8))
        fig_hist.add_vline(x=-var95 * 100, line_dash="dash", annotation_text="VaR 95%", annotation_position="top right")
        fig_hist.update_layout(
            height=420,
            template="plotly_white",
            title=f"{target_label}: Distribution of Final Returns",
            title_font_color="#1a237e",
            xaxis_title="Final return (%)",
            yaxis_title="Frequency",
            margin=dict(l=10, r=10, t=70, b=60),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


def tab_correlation_risk(prices: pd.DataFrame):
    st.markdown('<div class="section-header">Correlation & Risk Analytics</div>', unsafe_allow_html=True)
    if prices.empty:
        return

    ret = prices.pct_change().dropna()
    corr = ret.corr()

    fig = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation Heatmap")
    fig.update_layout(height=650, title_font_color="#1a237e")
    st.plotly_chart(fig, use_container_width=True)

    # rolling vol + drawdowns
    st.markdown('<div class="subsection-header">Rolling Volatility & Drawdown</div>', unsafe_allow_html=True)
    wlen = st.slider("Rolling window (days)", 21, 252, 63, 21, key="perf_roll_window_days")
    vol = ret.rolling(wlen).std() * np.sqrt(252)
    figv = go.Figure()
    for c in vol.columns:
        figv.add_trace(go.Scatter(x=vol.index, y=vol[c]*100, mode="lines", name=TICKER_NAME_MAP.get(c, c)))
    figv.update_layout(height=420, template="plotly_white", title=f"Rolling Annualized Volatility ({wlen}d)", title_font_color="#1a237e", yaxis_title="Vol %")
    st.plotly_chart(figv, use_container_width=True)

    dd = (prices / prices.cummax()) - 1
    figd = go.Figure()
    for c in dd.columns:
        figd.add_trace(go.Scatter(x=dd.index, y=dd[c]*100, mode="lines", name=TICKER_NAME_MAP.get(c, c)))
    figd.update_layout(height=420, template="plotly_white", title="Underwater (Drawdown) Curves", title_font_color="#1a237e", yaxis_title="Drawdown %")
    st.plotly_chart(figd, use_container_width=True)

def tab_rolling_beta_capm(prices: pd.DataFrame, bench: pd.Series):
    st.markdown('<div class="section-header">Rolling Beta + CAPM Alpha Panel (with CI)</div>', unsafe_allow_html=True)
    if prices.empty or bench is None or bench.empty:
        st.info("Benchmark (^GSPC) not available for rolling beta/CAPM.")
        return

    ret = prices.pct_change().dropna()
    bret = bench.pct_change().dropna()

    # Align
    df = pd.concat([ret, bret.rename("BENCH")], axis=1).dropna()
    ret = df[prices.columns]
    bret = df["BENCH"]

    win_beta = st.slider("Rolling beta window (days)", 20, 252, 60, 5, key="roll_beta_window_days")
    betas = compute_rolling_beta(ret, bret, window=win_beta)

    if betas.empty:
        st.warning("Could not compute rolling betas (insufficient aligned data).")
        return

    # Line plot
    fig = go.Figure()
    for c in betas.columns:
        fig.add_trace(go.Scatter(x=betas.index, y=betas[c], mode="lines", name=TICKER_NAME_MAP.get(c, c)))
    fig.update_layout(height=420, template="plotly_white", title=f"Rolling Beta vs ^GSPC ({win_beta}d)", title_font_color="#1a237e", yaxis_title="Beta")
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: assets × time buckets (monthly or quarterly)
    bucket = st.selectbox("Heatmap time buckets", ["Monthly", "Quarterly"], index=0, key="beta_heatmap_bucket")
    if bucket == "Quarterly":
        bucketed = betas.resample("Q").mean()
    else:
        bucketed = betas.resample("M").mean()
    bucketed = bucketed.T  # assets x time

    fig_h = px.imshow(bucketed, aspect="auto", title="Rolling Beta Heatmap (Assets × Time Buckets)")
    fig_h.update_layout(height=650, title_font_color="#1a237e")
    st.plotly_chart(fig_h, use_container_width=True)
    download_csv(bucketed, "rolling_beta_heatmap_matrix.csv", "Download beta heatmap matrix CSV")

    # Rolling CAPM Alpha panel with confidence intervals
    st.markdown('<div class="subsection-header">Rolling CAPM Alpha Regression (with CI)</div>', unsafe_allow_html=True)
    sel = st.multiselect("Select assets for CAPM rolling regression", list(prices.columns), default=list(prices.columns)[:min(3, len(prices.columns))], key="capm_assets_select")
    capm_win = st.slider("CAPM regression window (days)", 60, 504, 126, 21, key="capm_reg_window_days")
    ci_level = st.slider("CI level", 0.80, 0.99, 0.95, 0.01, key="capm_ci_level")
    alpha = 1 - ci_level

    for a in sel:
        res = rolling_capm_with_ci(ret[a], bret, window=capm_win, alpha=alpha)
        if res.empty:
            st.warning(f"Insufficient data for rolling CAPM on {a}.")
            continue

        # Alpha chart with CI band
        figA = go.Figure()
        figA.add_trace(go.Scatter(x=res.index, y=res["alpha_ann"]*100, mode="lines", name="Alpha (ann, %)", line=dict(width=2)))
        figA.add_trace(go.Scatter(x=res.index, y=res["alpha_low_ann"]*100, mode="lines", name="CI Low", line=dict(dash="dash")))
        figA.add_trace(go.Scatter(x=res.index, y=res["alpha_high_ann"]*100, mode="lines", name="CI High", line=dict(dash="dash")))
        figA.update_layout(height=320, template="plotly_white", title=f"{TICKER_NAME_MAP.get(a,a)} Rolling Alpha (annualized) with {int(ci_level*100)}% CI", title_font_color="#1a237e", yaxis_title="Alpha %")
        st.plotly_chart(figA, use_container_width=True)

        # Beta chart with CI band
        figB = go.Figure()
        figB.add_trace(go.Scatter(x=res.index, y=res["beta"], mode="lines", name="Beta", line=dict(width=2)))
        figB.add_trace(go.Scatter(x=res.index, y=res["beta_low"], mode="lines", name="CI Low", line=dict(dash="dash")))
        figB.add_trace(go.Scatter(x=res.index, y=res["beta_high"], mode="lines", name="CI High", line=dict(dash="dash")))
        figB.update_layout(height=320, template="plotly_white", title=f"{TICKER_NAME_MAP.get(a,a)} Rolling Beta with {int(ci_level*100)}% CI", title_font_color="#1a237e", yaxis_title="Beta")
        st.plotly_chart(figB, use_container_width=True)

        # Summary table
        latest = res.iloc[-1]
        sm = pd.DataFrame({
            "Metric": ["Alpha (ann)", "Beta", "R^2"],
            "Value": [latest["alpha_ann"], latest["beta"], latest["r2"]]
        })
        st.dataframe(sm, use_container_width=True)


def tab_technicals_tracking(prices: pd.DataFrame, bench: pd.Series):
    st.markdown('<div class="section-header">📐 Technicals & Tracking (Bollinger + OHLC + Tracking Error)</div>', unsafe_allow_html=True)

    if prices is None or prices.empty:
        st.markdown('<div class="warning-card">⚠️ No price data available.</div>', unsafe_allow_html=True)
        return

    start_str = prices.index.min().strftime("%Y-%m-%d")
    end_str = prices.index.max().strftime("%Y-%m-%d")

    # Target selection
    w = st.session_state.get("last_weights", {})
    if not w:
        w = {c: 1.0 / prices.shape[1] for c in prices.columns}

    target_options = ["Portfolio (Weights)"] + list(prices.columns)
    target = st.selectbox(
        "Select instrument / portfolio",
        options=target_options,
        index=0,
        key="tech_target_select_v1"
    )

    if target == "Portfolio (Weights)":
        target_close = compute_portfolio_index(prices, w)
        target_name = "Portfolio"
    else:
        target_close = prices[target].dropna().copy()
        target_name = str(target)

    # Benchmark selection (combo boxes)
    st.markdown('<div class="subsection-header">Benchmark Selection</div>', unsafe_allow_html=True)
    colb1, colb2, colb3 = st.columns([1.2, 1.2, 1])
    with colb1:
        bench_cat = st.selectbox(
            "Benchmark Category",
            options=list(data_manager.universe.keys()),
            index=0,
            key="tech_bench_cat_v1"
        )
    with colb2:
        bench_name = st.selectbox(
            "Benchmark",
            options=list(data_manager.universe[bench_cat].keys()),
            index=0,
            key="tech_bench_name_v1"
        )
    with colb3:
        te_window = st.slider("Rolling TE window", 20, 252, 60, 5, key="tech_te_window_v1")

    bench_ticker = data_manager.universe[bench_cat][bench_name]
    bench_close = None

    # Prefer already-provided 'bench' if it matches, otherwise fetch chosen benchmark
    if isinstance(bench, pd.Series) and not bench.empty and getattr(bench, "name", None) == bench_ticker:
        bench_close = bench.dropna().copy()
    else:
        bench_close = fetch_close_series(bench_ticker, start_str, end_str)

    if bench_close is None or bench_close.empty:
        st.markdown('<div class="warning-card">⚠️ Benchmark price series could not be fetched. Please choose another benchmark.</div>', unsafe_allow_html=True)
        return

    # Sub-tabs inside Technicals
    t1, t2, t3 = st.tabs(["📎 Bollinger Bands", "🕯️ OHLC + Tracking Error", "📊 Breach & Signals Overview"])

    # -------------------------
    # Bollinger Bands
    # -------------------------
    with t1:
        st.markdown('<div class="subsection-header">Bollinger Bands (Interactive)</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            bb_window = st.slider("Window", 10, 200, 20, 1, key="bb_window_v1")
        with c2:
            bb_nstd = st.slider("Std dev", 1.0, 4.0, 2.0, 0.1, key="bb_nstd_v1")
        with c3:
            show_breaches = st.checkbox("Show breach markers", value=True, key="bb_show_breaches_v1")
        with c4:
            show_mid = st.checkbox("Show mid band (MA)", value=True, key="bb_show_mid_v1")

        mid, upper, lower = bollinger_bands(target_close, window=int(bb_window), n_std=float(bb_nstd))
        df_bb = pd.DataFrame({"Price": target_close, "Mid": mid, "Upper": upper, "Lower": lower}).dropna()

        if df_bb.empty or len(df_bb) < int(bb_window) + 5:
            st.markdown('<div class="warning-card">⚠️ Not enough data to compute Bollinger Bands for this window.</div>', unsafe_allow_html=True)
        else:
            last = df_bb.iloc[-1]
            above = bool(last["Price"] > last["Upper"])
            below = bool(last["Price"] < last["Lower"])

            cM1, cM2, cM3, cM4 = st.columns(4)
            cM1.metric("Last Price", f"{last['Price']:.2f}")
            cM2.metric("Upper Band", f"{last['Upper']:.2f}")
            cM3.metric("Lower Band", f"{last['Lower']:.2f}")
            if above:
                cM4.markdown('<span class="risk-badge risk-high">Above Upper (Overbought)</span>', unsafe_allow_html=True)
            elif below:
                cM4.markdown('<span class="risk-badge risk-high">Below Lower (Oversold)</span>', unsafe_allow_html=True)
            else:
                cM4.markdown('<span class="risk-badge risk-low">Inside Bands</span>', unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_bb.index, y=df_bb["Price"],
                mode="lines",
                name=f"{target_name} Price",
                line=dict(width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_bb.index, y=df_bb["Upper"],
                mode="lines",
                name="Upper Band",
                line=dict(width=1, dash="dash")
            ))
            if show_mid:
                fig.add_trace(go.Scatter(
                    x=df_bb.index, y=df_bb["Mid"],
                    mode="lines",
                    name="Mid (MA)",
                    line=dict(width=1)
                ))
            fig.add_trace(go.Scatter(
                x=df_bb.index, y=df_bb["Lower"],
                mode="lines",
                name="Lower Band",
                line=dict(width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(30, 60, 120, 0.08)"
            ))

            if show_breaches:
                breach_up = df_bb[df_bb["Price"] > df_bb["Upper"]]
                breach_dn = df_bb[df_bb["Price"] < df_bb["Lower"]]
                if not breach_up.empty:
                    fig.add_trace(go.Scatter(
                        x=breach_up.index, y=breach_up["Price"],
                        mode="markers",
                        name="Upper Breach",
                        marker=dict(size=8, symbol="triangle-up")
                    ))
                if not breach_dn.empty:
                    fig.add_trace(go.Scatter(
                        x=breach_dn.index, y=breach_dn["Price"],
                        mode="markers",
                        name="Lower Breach",
                        marker=dict(size=8, symbol="triangle-down")
                    ))

            fig.update_layout(
                template="plotly_white",
                height=560,
                title=f"Bollinger Bands — {target_name} (Window={bb_window}, σ={bb_nstd})",
                title_font_color="#1a237e",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=70, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # OHLC + Tracking Error
    # -------------------------
    with t2:
        st.markdown('<div class="subsection-header">OHLC (Daily) + Tracking Error vs Benchmark</div>', unsafe_allow_html=True)

        # OHLC Chart (instrument or portfolio index)
        ohlc_df = pd.DataFrame()
        if target == "Portfolio (Weights)":
            needed = tuple(dict.fromkeys(list(w.keys())))
            ohlc_map = fetch_ohlc(needed, start_str, end_str)
            ohlc_df = compute_portfolio_ohlc_index(ohlc_map, w)
        else:
            ohlc_map = fetch_ohlc((target,), start_str, end_str)
            ohlc_df = ohlc_map.get(target, pd.DataFrame())

        if ohlc_df is None or ohlc_df.empty:
            st.markdown('<div class="info-card">ℹ️ OHLC data not available for this target (some indexes/ETFs do not provide OHLC reliably). Showing Close chart instead.</div>', unsafe_allow_html=True)
            fig_close = px.line(
                target_close.to_frame("Close"),
                title=f"Close Price — {target_name}",
                labels={"value": "Price", "index": "Date"}
            )
            fig_close.update_layout(template="plotly_white", height=520, title_font_color="#1a237e", margin=dict(l=10, r=10, t=70, b=50))
            st.plotly_chart(fig_close, use_container_width=True)
        else:
            fig_ohlc = go.Figure(data=[
                go.Candlestick(
                    x=ohlc_df.index,
                    open=ohlc_df["Open"],
                    high=ohlc_df["High"],
                    low=ohlc_df["Low"],
                    close=ohlc_df["Close"],
                    name=target_name
                )
            ])
            fig_ohlc.update_layout(
                template="plotly_white",
                height=620,
                title=f"Daily OHLC — {target_name}",
                title_font_color="#1a237e",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=70, b=50),
            )
            st.plotly_chart(fig_ohlc, use_container_width=True)

        # Tracking error
        tr = target_close.pct_change().dropna()
        br = bench_close.pct_change().dropna()
        aligned = pd.concat([tr.rename("target"), br.rename("bench")], axis=1).dropna()
        if aligned.empty or len(aligned) < 20:
            st.markdown('<div class="warning-card">⚠️ Not enough overlapping data to compute tracking error.</div>', unsafe_allow_html=True)
        else:
            active = aligned["target"] - aligned["bench"]
            te = tracking_error(active)
            te_roll = active.rolling(int(te_window)).std() * np.sqrt(252)

            cT1, cT2, cT3, cT4 = st.columns(4)
            cT1.metric("Tracking Error (Ann.)", f"{te*100:.2f}%")
            cT2.metric("Active Return (Ann.)", f"{(active.mean()*252)*100:.2f}%")
            cT3.metric("Correlation (Target vs Bench)", f"{aligned.corr().iloc[0,1]:.3f}")
            cT4.metric("Overlap Days", f"{len(aligned):,}")

            fig_te = go.Figure()
            fig_te.add_trace(go.Scatter(
                x=te_roll.index,
                y=te_roll * 100,
                mode="lines",
                name=f"Rolling TE ({te_window}d)"
            ))
            fig_te.update_layout(
                template="plotly_white",
                height=420,
                title=f"Rolling Tracking Error vs {bench_name} ({bench_ticker})",
                title_font_color="#1a237e",
                xaxis_title="Date",
                yaxis_title="Tracking Error (%)",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=70, b=50),
            )
            st.plotly_chart(fig_te, use_container_width=True)

    # -------------------------
    # Breach & Signals Overview
    # -------------------------
    with t3:
        st.markdown('<div class="subsection-header">Band Breaches & Quick Signals</div>', unsafe_allow_html=True)
        bb_window2 = st.slider("BB window (signals)", 10, 200, 20, 1, key="bb_window_sig_v1")
        bb_nstd2 = st.slider("Std dev (signals)", 1.0, 4.0, 2.0, 0.1, key="bb_nstd_sig_v1")

        mid2, upper2, lower2 = bollinger_bands(target_close, window=int(bb_window2), n_std=float(bb_nstd2))
        df_sig = pd.DataFrame({"Price": target_close, "Upper": upper2, "Lower": lower2}).dropna()
        if df_sig.empty:
            st.markdown('<div class="warning-card">⚠️ Not enough data to compute signals.</div>', unsafe_allow_html=True)
        else:
            df_sig["AboveUpper"] = df_sig["Price"] > df_sig["Upper"]
            df_sig["BelowLower"] = df_sig["Price"] < df_sig["Lower"]
            df_sig["Breach"] = np.where(df_sig["AboveUpper"], "Above Upper", np.where(df_sig["BelowLower"], "Below Lower", "Inside"))

            last_n = st.slider("Lookback (days)", 30, 504, 180, 10, key="bb_lookback_v1")
            df_last = df_sig.tail(int(last_n)).copy()
            breach_counts = df_last["Breach"].value_counts()

            cS1, cS2, cS3 = st.columns(3)
            cS1.metric("Above Upper (count)", int(breach_counts.get("Above Upper", 0)))
            cS2.metric("Below Lower (count)", int(breach_counts.get("Below Lower", 0)))
            cS3.metric("Inside (count)", int(breach_counts.get("Inside", 0)))

            st.dataframe(
                df_last[["Price", "Upper", "Lower", "Breach"]].tail(250),
                use_container_width=True,
                height=360
            )

            # Quick breach heat strip
            fig_strip = go.Figure()
            fig_strip.add_trace(go.Scatter(
                x=df_last.index,
                y=[1]*len(df_last),
                mode="markers",
                marker=dict(
                    size=10,
                    color=np.where(df_last["AboveUpper"], 2, np.where(df_last["BelowLower"], 0, 1)),
                    colorscale=[[0, "#c62828"], [0.5, "#fbc02d"], [1.0, "#2e7d32"]],
                    showscale=False
                ),
                hovertext=df_last["Breach"],
                name="Breach"
            ))
            fig_strip.update_layout(
                template="plotly_white",
                height=180,
                title="Bollinger Breach Heat Strip (Red=Below, Yellow=Inside, Green=Above)",
                title_font_color="#1a237e",
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(fig_strip, use_container_width=True)



def tab_advanced_performance(prices: pd.DataFrame, bench: pd.Series):
    st.markdown('<div class="section-header">Advanced Performance Metrics (50+)</div>', unsafe_allow_html=True)

    if "last_weights" not in st.session_state or not st.session_state.get("last_weights"):
        st.info("Run an optimization first to generate a portfolio for performance analytics.")
        return

    w = st.session_state["last_weights"]
    rf = st.session_state.get("rf_last", 0.04)

    ret = prices.pct_change().dropna()
    pr = portfolio_returns_from_weights(ret, w)

    bench_ret = bench.pct_change().dropna() if bench is not None and not bench.empty else None
    apm = AdvancedPerformanceMetrics(pr, bench_ret, rf_annual=rf)

    mdf = apm.compute().sort_values("Metric")
    st.dataframe(mdf, use_container_width=True)
    download_csv(mdf, "advanced_performance_metrics.csv", "Download all metrics CSV")

    # Rolling metrics charts
    st.markdown('<div class="subsection-header">Rolling Metrics Dashboard</div>', unsafe_allow_html=True)
    win = st.slider("Rolling window (days)", 21, 252, 63, 21, key="rollingbeta_bucket_window_days")

    rolling_ret = pr.rolling(win).apply(lambda x: (1 + x).prod() - 1, raw=False)
    rolling_vol = pr.rolling(win).std() * np.sqrt(252)
    rolling_sh = (pr.rolling(win).mean() * 252 - rf) / (pr.rolling(win).std() * np.sqrt(252))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"Rolling Return ({win}d)", f"Rolling Volatility ({win}d)", f"Rolling Sharpe ({win}d)"))
    fig.add_trace(go.Scatter(x=rolling_ret.index, y=rolling_ret*100, mode="lines", name="Roll Ret %"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol*100, mode="lines", name="Roll Vol %"), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolling_sh.index, y=rolling_sh, mode="lines", name="Roll Sharpe"), row=3, col=1)
    fig.update_layout(height=820, template="plotly_white", title="Rolling Performance Analytics", title_font_color="#1a237e", showlegend=False)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart (normalized metrics)
    st.markdown('<div class="subsection-header">Performance Radar (Quick Assessment)</div>', unsafe_allow_html=True)
    key = {
        "CAGR": ("Return", True),
        "Annualized Volatility": ("Risk", False),
        "Sharpe Ratio": ("Risk-Adj", True),
        "Sortino Ratio": ("Risk-Adj", True),
        "Max Drawdown": ("Risk", False),
        "Omega Ratio": ("Tail", True),
        "Information Ratio": ("Benchmark", True),
        "CAPM Alpha (ann)": ("Benchmark", True)
    }
    sub = mdf[mdf["Metric"].isin(key.keys())].copy()
    if len(sub) >= 5:
        # normalize (higher is better)
        vals = []
        cats = []
        for _, r in sub.iterrows():
            m = r["Metric"]
            v = float(r["Value"]) if np.isfinite(r["Value"]) else 0.0
            # invert bad metrics
            if m in ["Annualized Volatility"]:
                v = -v
            if m in ["Max Drawdown"]:
                v = -abs(v)
            cats.append(m)
            vals.append(v)
        vals = np.array(vals)
        # scale 0..1
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        scaled = (vals - vmin) / (vmax - vmin + 1e-12)

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=scaled, theta=cats, fill="toself", name="Portfolio"))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=520, template="plotly_white",
                            title="Radar (normalized)", title_font_color="#1a237e")
        st.plotly_chart(radar, use_container_width=True)


def tab_black_litterman(prices: pd.DataFrame, bench: pd.Series):
    st.markdown('<div class="section-header">Black-Litterman Optimization Lab (Multi-View)</div>', unsafe_allow_html=True)

    if prices.empty or len(prices.columns) < 2:
        st.info("Select at least 2 assets to build BL views and optimize.")
        return

    assets = list(prices.columns)
    rf = st.number_input("Risk-free rate (annual)", 0.0, 0.30, 0.04, 0.005, format="%.3f", key="bl_rf_main")
    tau = st.slider("Tau", 0.01, 0.30, 0.05, 0.01, key="bl_tau")
    objective = st.selectbox("Optimization objective", ["max_sharpe", "min_volatility", "max_quadratic_utility"], index=0, key="bl_objective")

    st.markdown('<div class="subsection-header">Quick Templates</div>', unsafe_allow_html=True)
    tpl = st.selectbox("Pick a template (optional)", ["(None)"] + list(BL_TEMPLATES.keys()), key="bl_template")
    template_payload = None
    if tpl != "(None)":
        template_payload = BL_TEMPLATES[tpl]

    tabs = st.tabs(["Absolute Views", "Relative Views", "Ranking Views", "Run BL"])
    abs_views = {}
    abs_conf = {}
    rel_views = []
    rel_conf = []
    rank_assets = []

    with tabs[0]:
        st.write("Absolute views: set expected return for an asset (annualized).")
        if template_payload and template_payload.get("type") == "absolute":
            st.info(f"Template loaded: {tpl}")
        n = st.number_input("Number of absolute views", 0, min(10, len(assets)), 2, 1, key="bl_n_abs_views")
        for i in range(int(n)):
            c1, c2, c3 = st.columns([1.2, 1, 1])
            a = c1.selectbox(f"Asset #{i+1}", assets, key=f"abs_asset_{i}")
            v = c2.number_input(f"Expected return (ann) #{i+1}", -0.50, 1.00, 0.10, 0.01, key=f"abs_ret_{i}")
            conf = c3.slider(f"Confidence #{i+1}", 0.05, 0.95, 0.60, 0.05, key=f"abs_conf_{i}")
            abs_views[a] = float(v)
            abs_conf[a] = float(conf)
        if template_payload and template_payload.get("type") == "absolute":
            for k, v in template_payload.get("views", {}).items():
                if k in assets:
                    abs_views[k] = float(v)
                    abs_conf[k] = abs_conf.get(k, 0.60)
            st.success("Applied template to absolute views (where assets exist in your selection).")

    with tabs[1]:
        st.write("Relative views: Asset A will outperform Asset B by spread (annualized).")
        if template_payload and template_payload.get("type") == "relative":
            st.info(f"Template loaded: {tpl}")
        n = st.number_input("Number of relative views", 0, 10, 2, 1, key="bl_n_rel_views")
        for i in range(int(n)):
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            a = c1.selectbox(f"A #{i+1}", assets, key=f"rel_a_{i}")
            b = c2.selectbox(f"B #{i+1}", assets, key=f"rel_b_{i}")
            sp = c3.number_input(f"Spread A-B (ann) #{i+1}", -0.50, 0.50, 0.02, 0.005, key=f"rel_sp_{i}")
            conf = c4.slider(f"Confidence #{i+1}", 0.05, 0.95, 0.60, 0.05, key=f"rel_conf_{i}")
            if a != b:
                rel_views.append((a, b, float(sp)))
                rel_conf.append(float(conf))
        if template_payload and template_payload.get("type") == "relative":
            for (a, b, sp) in template_payload.get("views", []):
                if a in assets and b in assets and a != b:
                    rel_views.append((a, b, float(sp)))
                    rel_conf.append(0.60)
            st.success("Applied template to relative views (where assets exist in your selection).")

    with tabs[2]:
        st.write("Ranking views: create a priority order; converted into adjacent relative views.")
        rank_assets = st.multiselect("Ranked assets (top → bottom)", assets, default=assets[:min(5, len(assets))], key="bl_rank_assets")
        spread = st.number_input("Default spread per rank step (annualized)", -0.20, 0.20, 0.02, 0.005, key="bl_rank_spread")
        conf_rank = st.slider("Confidence for ranking-derived views", 0.05, 0.95, 0.60, 0.05, key="bl_rank_conf")

    with tabs[3]:
        st.write("Run Black-Litterman with your configured views.")
        run = st.button("Run BL Optimization (Institutional)")
        if run:
            pe = PortfolioEngine(prices)
            payload = {"abs": {}, "P": None, "Q": None, "conf": None}

            # precedence: if relative views exist, use P/Q; else absolute.
            if rel_views:
                P, Q = build_relative_view_matrix(assets, rel_views)
                payload["P"], payload["Q"], payload["conf"] = P, Q, np.array(rel_conf, dtype=float)
            elif rank_assets and len(rank_assets) >= 2:
                rv = build_ranking_views(rank_assets, default_spread=float(spread))
                P, Q = build_relative_view_matrix(assets, rv)
                payload["P"], payload["Q"], payload["conf"] = P, Q, np.array([conf_rank]*len(rv), dtype=float)
            else:
                payload["abs"] = abs_views
                payload["conf"] = abs_conf

            w, perf, prior, post = pe.black_litterman(
                rf=rf, tau=tau, objective=objective,
                views_abs=payload["abs"] if payload["abs"] else None,
                P=payload["P"], Q=payload["Q"], view_confidences=payload["conf"],
                benchmark_returns=bench.pct_change().dropna() if bench is not None and not bench.empty else None
            )

            st.session_state["bl_views_payload"] = payload
            st.session_state["last_weights"] = w
            st.session_state["last_perf"] = perf
            st.session_state["last_strategy"] = "Black-Litterman"
            st.session_state["rf_last"] = rf

            st.success("BL optimization completed and stored in session (shared across tabs).")

            wdf = pd.DataFrame({"Asset": list(w.keys()), "Weight": list(w.values())}).sort_values("Weight", ascending=False)
            st.dataframe(wdf, use_container_width=True)

            kpi_row([
                ("Annual Return", fmt_pct(perf[0]), ""),
                ("Annual Vol", fmt_pct(perf[1]), ""),
                ("Sharpe", fmt_num(perf[2]), "")
            ])

            comp = pd.DataFrame({"Prior": prior, "Posterior": post}).dropna()
            comp["Δ"] = comp["Posterior"] - comp["Prior"]
            st.dataframe(comp.sort_values("Δ", ascending=False), use_container_width=True)
            download_csv(comp, "bl_equilibrium_vs_posterior.csv", "Download equilibrium vs adjusted returns")

def tab_efficient_frontier(prices: pd.DataFrame):
    st.markdown('<div class="section-header">Advanced Efficient Frontier (2D + 3D + CML + Risk Contributions)</div>', unsafe_allow_html=True)
    if prices.empty or len(prices.columns) < 2:
        st.info("Select at least 2 assets.")
        return

    rf = st.number_input("Risk-free rate (annual)", 0.0, 0.30, 0.04, 0.005, format="%.3f", key="frontier_rf")
    n_rand = st.slider("Random portfolios", 500, 10000, 4000, 500, key="frontier_n_rand")
    n_front = st.slider("Frontier points", 10, 80, 40, 5, key="frontier_n_front")

    pe = PortfolioEngine(prices)
    mu = pd.Series(pe.mu).reindex(pe.assets)
    cov = pd.DataFrame(pe.S, index=pe.assets, columns=pe.assets)

    figs = efficient_frontier_charts(mu, cov, rf=rf, n_random=int(n_rand), n_frontier=int(n_front))

    st.plotly_chart(figs["frontier"], use_container_width=True)

    c1, c2 = st.columns([1,1])
    with c1:
        st.plotly_chart(figs["sharpe_dist"], use_container_width=True)
    with c2:
        st.plotly_chart(figs["risk_contrib"], use_container_width=True)

    st.plotly_chart(figs["frontier3d"], use_container_width=True)


def tab_weight_stability(prices: pd.DataFrame, bench: pd.Series):
    st.markdown('<div class="section-header">Posterior Weight Stability & Turnover Diagnostics (BL vs MV vs HRP)</div>', unsafe_allow_html=True)
    if prices.empty or len(prices.columns) < 2:
        st.info("Select at least 2 assets.")
        return

    rf = st.number_input("Risk-free rate (annual)", 0.0, 0.30, 0.04, 0.005, format="%.3f", key="diag_rf")
    lookback = st.slider("Lookback window (days)", 60, 756, 252, 21, key="stability_lookback")
    rebalance = st.selectbox("Rebalance frequency", ["M", "W", "Q"], index=0, key="stability_rebalance")
    tau = st.slider("BL Tau", 0.01, 0.30, 0.05, 0.01, key="diag_tau")
    mv_obj = st.selectbox("MV objective", ["max_sharpe", "min_volatility", "max_quadratic_utility"], index=0, key="diag_mv")
    bl_obj = st.selectbox("BL objective", ["max_sharpe", "min_volatility", "max_quadratic_utility"], index=0, key="diag_bl")

    views_payload = st.session_state.get("bl_views_payload", None)

    methods = st.multiselect("Methods to compare", ["MV", "HRP", "BL"], default=["BL", "MV", "HRP"], key="stability_methods")

    run = st.button("Run Stability Diagnostics")
    if not run:
        st.info("Configure and run diagnostics to compute rolling weights & turnover.")
        return

    results = {}
    for m in methods:
        wdf, to = rolling_weight_paths(
            prices=prices,
            method=m,
            rf=rf,
            lookback=int(lookback),
            rebalance=rebalance,
            tau=float(tau),
            bl_objective=bl_obj,
            mv_objective=mv_obj,
            views_payload=views_payload,
            benchmark=bench.pct_change().dropna() if bench is not None and not bench.empty else None
        )
        results[m] = (wdf, to)

    # Turnover chart
    st.markdown('<div class="subsection-header">Turnover Over Time</div>', unsafe_allow_html=True)
    figT = go.Figure()
    for m, (wdf, to) in results.items():
        if to is None or to.empty:
            continue
        figT.add_trace(go.Scatter(x=to.index, y=to, mode="lines", name=f"{m} Turnover"))
    figT.update_layout(height=420, template="plotly_white", title="Turnover (0.5 * Σ|Δw|)", title_font_color="#1a237e", yaxis_title="Turnover")
    st.plotly_chart(figT, use_container_width=True)

    # Summary table
    rows = []
    for m, (wdf, to) in results.items():
        if wdf is None or wdf.empty:
            continue
        rows.append({
            "Method": m,
            "Avg Turnover": float(to.mean()) if to is not None and not to.empty else np.nan,
            "Max Turnover": float(to.max()) if to is not None and not to.empty else np.nan,
            "Weight Std (avg)": float(wdf.std().mean()),
            "HHI (avg)": float((wdf**2).sum(axis=1).mean())
        })
    dsum = pd.DataFrame(rows)
    st.dataframe(dsum, use_container_width=True)
    download_csv(dsum, "weight_stability_summary.csv", "Download stability summary CSV")

    # Weight stability heatmaps (std of weights)
    st.markdown('<div class="subsection-header">Weight Stability (Std Dev Heatmap)</div>', unsafe_allow_html=True)
    for m, (wdf, to) in results.items():
        if wdf is None or wdf.empty:
            continue
        stds = wdf.std().to_frame(name="Weight Std").sort_values("Weight Std", ascending=False)
        figH = px.imshow(stds.T, aspect="auto", title=f"{m} - Weight Std Across Rebalances (higher=less stable)")
        figH.update_layout(height=220, title_font_color="#1a237e")
        st.plotly_chart(figH, use_container_width=True)

        # Weight path (stacked area) for top-N assets
        topn = st.slider(f"{m}: show top-N assets (by avg weight)", 5, min(25, len(wdf.columns)), min(10, len(wdf.columns)), 1, key=f"topn_{m}")
        avg_w = wdf.mean().sort_values(ascending=False)
        cols = list(avg_w.head(topn).index)
        w_small = wdf[cols].copy()
        w_small["Other"] = 1 - w_small.sum(axis=1)
        figA = go.Figure()
        for c in w_small.columns:
            figA.add_trace(go.Scatter(x=w_small.index, y=w_small[c], stackgroup="one", mode="lines", name=TICKER_NAME_MAP.get(c, c)))
        figA.update_layout(height=420, template="plotly_white", title=f"{m} Weight Allocation Over Time (Top {topn} + Other)", title_font_color="#1a237e", yaxis_title="Weight")
        st.plotly_chart(figA, use_container_width=True)

        download_csv(wdf, f"{m.lower()}_weights_path.csv", f"Download {m} weights path CSV")

# ==============================================================================
# Main App
# ==============================================================================
def main():
    st.markdown('<div class="main-header">QUANTUM | Global Institutional Terminal</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>Institutional-Grade Portfolio Analytics</b><br>'
                'Advanced optimization (MV/HRP/BL), VaR/CVaR/ES, stress testing, rolling beta heatmaps, rolling CAPM alpha with CI, efficient frontier, and stability/turnover diagnostics.</div>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### Universe & Data")
    category = st.sidebar.selectbox("Universe Category", list(data_manager.universe.keys()))
    universe_dict = data_manager.universe[category]

    default_tickers = list(universe_dict.values())[:min(8, len(universe_dict))]
    tickers = st.sidebar.multiselect("Select assets (50+ supported)", list(universe_dict.values()), default=default_tickers)
    st.sidebar.caption("Tip: pick 8–20 assets for fastest diagnostics; you can still select more.")

    start = st.sidebar.date_input("Start date", value=(datetime.now() - timedelta(days=365*5)).date())
    end = st.sidebar.date_input("End date", value=datetime.now().date())
    min_points = st.sidebar.slider("Minimum data points per asset", 60, 1000, 200, 20)

    if not tickers:
        st.warning("Select at least one asset.")
        return

    # Fetch
    with st.spinner("Fetching and aligning market data..."):
        prices, bench, report = fetch_prices(tuple(tickers), str(start), str(end), int(min_points))

    if prices.empty:
        st.error("No usable price data. Adjust tickers/date range/min points.")
        if report.get("warnings"):
            st.write(report["warnings"])
        return

    # Tabs
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

    with tabs[0]:
        tab_market_overview(prices, bench, report)
    with tabs[1]:
        tab_portfolio_optimization(prices, bench)
    with tabs[2]:
        tab_advanced_var(prices, bench)
    with tabs[3]:
        tab_stress_testing(prices)
    with tabs[4]:
        tab_correlation_risk(prices)
    with tabs[5]:
        tab_rolling_beta_capm(prices, bench)
    with tabs[6]:
        tab_technicals_tracking(prices, bench)
    with tabs[7]:
        tab_advanced_performance(prices, bench)
    with tabs[8]:
        tab_black_litterman(prices, bench)
    with tabs[9]:
        tab_efficient_frontier(prices)
    with tabs[10]:
        tab_weight_stability(prices, bench)

    # Footer
    st.markdown('<div class="card"><b>Notes</b><br>'
                '• BL views are shared across tabs via session_state (build in BL Lab).<br>'
                '• For large universes (30+), diagnostics may be computationally heavy; reduce assets or increase caching TTL if needed.<br>'
                '• For indexes/ETFs without market caps, BL falls back to equal-weight market prior.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
