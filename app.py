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
    .info-card { background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%); border-left:4px solid #2196f3; color:#0d47a1; border-radius:10px; padding:12px 14px; margin:10px 0; }
    .success-card { background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%); border-left:4px solid #4caf50; color:#1b5e20; border-radius:10px; padding:12px 14px; margin:10px 0; }
    .warning-card { background:linear-gradient(135deg,#fff8e1 0%,#ffecb3 100%); border-left:4px solid #ff9800; color:#5d4037; border-radius:10px; padding:12px 14px; margin:10px 0; }
    .stTabs [data-baseweb="tab-list"] { gap:5px; background:#f0f2f6; padding:6px; border-radius:12px; }
    .stTabs [data-baseweb="tab"] { border-radius:10px; padding:10px 16px; font-weight:700; background:white; border:1px solid #e0e0e0; color:#5c6bc0; }
    .stTabs [aria-selected="true"] { background:linear-gradient(135deg,#1a237e 0%,#283593 100%); color:white; border-color:#1a237e; box-shadow:0 4px 12px rgba(26,35,126,.15); }
    .stButton>button { background:linear-gradient(135deg,#303f9f 0%,#1a237e 100%); color:white; border:none; border-radius:10px; padding:10px 18px; font-weight:700; }
    .stButton>button:hover { transform:translateY(-1px); box-shadow:0 6px 14px rgba(26,35,126,.22); }
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
# Optional optimization dependencies
# ==============================================================================
OPTIMIZATION_AVAILABLE = False
HRP_AVAILABLE = False
BLACK_LITTERMAN_AVAILABLE = False

EfficientFrontier = None
risk_models = None
expected_returns = None
objective_functions = None
HRPOpt = None
BlackLittermanModel = None
market_implied_prior_returns = None
market_implied_risk_aversion = None

try:
    from pypfopt.efficient_frontier import EfficientFrontier as _EF
    from pypfopt import risk_models as _rm, expected_returns as _er, objective_functions as _of
    EfficientFrontier, risk_models, expected_returns, objective_functions = _EF, _rm, _er, _of
    OPTIMIZATION_AVAILABLE = True

    # HRP
    try:
        from pypfopt.hierarchical_portfolio import HRPOpt as _HRP
        HRPOpt = _HRP
        HRP_AVAILABLE = True
    except Exception:
        try:
            from pypfopt.hierarchical_risk_parity import HRPOpt as _HRP2
            HRPOpt = _HRP2
            HRP_AVAILABLE = True
        except Exception:
            HRP_AVAILABLE = False

    # BL
    try:
        from pypfopt.black_litterman import (
            BlackLittermanModel as _BL,
            market_implied_prior_returns as _mipr,
            market_implied_risk_aversion as _mira
        )
        BlackLittermanModel = _BL
        market_implied_prior_returns = _mipr
        market_implied_risk_aversion = _mira
        BLACK_LITTERMAN_AVAILABLE = True
    except Exception:
        try:
            from pypfopt.black_litterman import BlackLittermanModel as _BL2
            BlackLittermanModel = _BL2
            BLACK_LITTERMAN_AVAILABLE = True
        except Exception:
            BLACK_LITTERMAN_AVAILABLE = False

except Exception as e:
    OPTIMIZATION_AVAILABLE = False
    _OPT_IMPORT_ERROR = str(e)

@st.cache_data(ttl=3600, show_spinner=False)
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
        "Dot-com Bubble Burst (2000-2002)": ("2000-03-10", "2002-10-09", "Tech bubble collapse", "Severe"),
        "9/11 Shock (2001)": ("2001-09-10", "2001-09-21", "Terror attacks + market closure", "High"),
        "Global Financial Crisis (2008-2009)": ("2007-10-09", "2009-03-09", "Banking crisis", "Extreme"),
        "COVID Crash (2020)": ("2020-02-19", "2020-03-23", "Pandemic shock", "Severe"),
        "Inflation & Rate Hikes (2022)": ("2022-01-03", "2022-10-12", "Aggressive tightening", "Moderate"),
        "US Banking Turmoil (2023)": ("2023-03-01", "2023-03-31", "Regional bank stress", "Moderate"),
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
    rf = colA.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.30, value=0.04, step=0.005, format="%.3f")
    mv_objective = colB.selectbox("MV Objective", ["max_sharpe", "min_volatility", "max_quadratic_utility"], index=0)
    l2 = colC.slider("L2 Regularization (gamma)", 0.0, 2.0, 0.0, 0.05)
    bl_tau = colD.slider("BL Tau (confidence scaling)", 0.01, 0.30, 0.05, 0.01)

    st.markdown('<div class="subsection-header">Run Optimization</div>', unsafe_allow_html=True)
    run_cols = st.columns([1,1,1,1])
    do_equal = run_cols[0].button("Run Equal Weight")
    do_mv = run_cols[1].button("Run Mean-Variance (MV)")
    do_hrp = run_cols[2].button("Run HRP")
    do_bl = run_cols[3].button("Run Black-Litterman (BL)")

    # BL views from session (if user created in BL tab)
    views_payload = st.session_state.get("bl_views_payload", None)

    def show_solution(name: str, w: Dict[str, float], perf: Tuple[float, float, float], prior: Optional[pd.Series]=None, post: Optional[pd.Series]=None):
        st.markdown(f'<div class="card"><div class="subsection-header">{name} Results</div>', unsafe_allow_html=True)

        # weights table
        wdf = pd.DataFrame({"Asset": list(w.keys()), "Weight": list(w.values())})
        wdf["AssetName"] = wdf["Asset"].map(lambda t: TICKER_NAME_MAP.get(t, t))
        wdf = wdf.sort_values("Weight", ascending=False)
        st.dataframe(wdf, use_container_width=True)

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
    invest = st.number_input("Investment Amount ($)", min_value=1000.0, max_value=1e9, value=1_000_000.0, step=50_000.0)
    hist = var_engine.historical(cl, hp)
    if np.isfinite(hist.get("VaR_hp", np.nan)):
        st.markdown('<div class="info-card"><b>Dollar VaR/ES</b><br>'
                    f'VaR({int(cl*100)}%, {hp}d): <b>${invest*hist["VaR_hp"]:,.0f}</b><br>'
                    f'CVaR/ES({int(cl*100)}%): <b>${invest*hist["CVaR"]:,.0f}</b></div>', unsafe_allow_html=True)


def tab_stress_testing(prices: pd.DataFrame):
    st.markdown('<div class="section-header">Stress Testing Lab (Historical + Custom)</div>', unsafe_allow_html=True)

    if "last_weights" not in st.session_state or not st.session_state.get("last_weights"):
        st.info("Run an optimization first to define portfolio weights.")
        return

    w = st.session_state["last_weights"]
    ret = prices.pct_change().dropna()
    pr = portfolio_returns_from_weights(ret, w)

    st.markdown('<div class="subsection-header">Historical Crises</div>', unsafe_allow_html=True)
    chosen = st.multiselect("Select historical stress events", list(StressTestEngine.HISTORICAL.keys()),
                            default=["Dot-com Bubble Burst (2000-2002)", "9/11 Shock (2001)"] if "9/11 Shock (2001)" in StressTestEngine.HISTORICAL else list(StressTestEngine.HISTORICAL.keys())[:2])

    rows = []
    for name in chosen:
        s, e, desc, sev = StressTestEngine.HISTORICAL[name]
        slice_prices = StressTestEngine.slice_period(prices, s, e)
        if slice_prices.empty:
            continue
        r = slice_prices.pct_change().dropna()
        p = portfolio_returns_from_weights(r, w)
        rows.append({
            "Scenario": name,
            "Period": f"{s} → {e}",
            "Severity": sev,
            "Return %": float(((1+p).prod()-1)*100),
            "Max DD %": float(max_drawdown(p)*100),
            "Vol (ann) %": float(annualize_vol(p)*100)
        })

    if rows:
        dfh = pd.DataFrame(rows).sort_values("Max DD %")
        st.dataframe(dfh, use_container_width=True)
        download_csv(dfh, "historical_stress_results.csv", "Download historical stress CSV")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=dfh["Scenario"], y=dfh["Max DD %"], name="Max DD %"))
        fig.update_layout(height=420, template="plotly_white", title="Historical Stress: Max Drawdown", title_font_color="#1a237e", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="subsection-header">Custom Shock Simulation (Monte-Carlo)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    magnitude = c1.slider("Shock magnitude (total)", 0.05, 0.60, 0.30, 0.01)
    duration = c2.slider("Shock duration (days)", 5, 180, 30, 5)
    volm = c3.slider("Volatility multiplier", 1.0, 8.0, 3.0, 0.25)
    sims = c4.selectbox("Simulations", [500, 1000, 1500, 2500], index=2)

    if st.button("Run Custom Shock Simulation"):
        res = StressTestEngine.scenario_simulation(pr, magnitude=magnitude, duration=duration, vol_mult=volm, sims=sims)
        st.markdown('<div class="success-card"><b>Custom Shock Results</b><br>' +
                    "<br>".join([f"{k}: <b>{v:.2f}</b>" if isinstance(v, (int,float)) else f"{k}: <b>{v}</b>" for k, v in res.items()]) +
                    "</div>", unsafe_allow_html=True)


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
    wlen = st.slider("Rolling window (days)", 21, 252, 63, 21)
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

    win_beta = st.slider("Rolling beta window (days)", 20, 252, 60, 5)
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
    bucket = st.selectbox("Heatmap time buckets", ["Monthly", "Quarterly"], index=0)
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
    sel = st.multiselect("Select assets for CAPM rolling regression", list(prices.columns), default=list(prices.columns)[:min(3, len(prices.columns))])
    capm_win = st.slider("CAPM regression window (days)", 60, 504, 126, 21)
    ci_level = st.slider("CI level", 0.80, 0.99, 0.95, 0.01)
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
    win = st.slider("Rolling window (days)", 21, 252, 63, 21)

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
    rf = st.number_input("Risk-free rate (annual)", 0.0, 0.30, 0.04, 0.005, format="%.3f")
    tau = st.slider("Tau", 0.01, 0.30, 0.05, 0.01)
    objective = st.selectbox("Optimization objective", ["max_sharpe", "min_volatility", "max_quadratic_utility"], index=0)

    st.markdown('<div class="subsection-header">Quick Templates</div>', unsafe_allow_html=True)
    tpl = st.selectbox("Pick a template (optional)", ["(None)"] + list(BL_TEMPLATES.keys()))
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
        n = st.number_input("Number of absolute views", 0, min(10, len(assets)), 2, 1)
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
        n = st.number_input("Number of relative views", 0, 10, 2, 1)
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
        rank_assets = st.multiselect("Ranked assets (top → bottom)", assets, default=assets[:min(5, len(assets))])
        spread = st.number_input("Default spread per rank step (annualized)", -0.20, 0.20, 0.02, 0.005)
        conf_rank = st.slider("Confidence for ranking-derived views", 0.05, 0.95, 0.60, 0.05)

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

    rf = st.number_input("Risk-free rate (annual)", 0.0, 0.30, 0.04, 0.005, format="%.3f")
    n_rand = st.slider("Random portfolios", 500, 10000, 4000, 500)
    n_front = st.slider("Frontier points", 10, 80, 40, 5)

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
    lookback = st.slider("Lookback window (days)", 60, 756, 252, 21)
    rebalance = st.selectbox("Rebalance frequency", ["M", "W", "Q"], index=0)
    tau = st.slider("BL Tau", 0.01, 0.30, 0.05, 0.01, key="diag_tau")
    mv_obj = st.selectbox("MV objective", ["max_sharpe", "min_volatility", "max_quadratic_utility"], index=0, key="diag_mv")
    bl_obj = st.selectbox("BL objective", ["max_sharpe", "min_volatility", "max_quadratic_utility"], index=0, key="diag_bl")

    views_payload = st.session_state.get("bl_views_payload", None)

    methods = st.multiselect("Methods to compare", ["MV", "HRP", "BL"], default=["BL", "MV", "HRP"])

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
        tab_advanced_performance(prices, bench)
    with tabs[7]:
        tab_black_litterman(prices, bench)
    with tabs[8]:
        tab_efficient_frontier(prices)
    with tabs[9]:
        tab_weight_stability(prices, bench)

    # Footer
    st.markdown('<div class="card"><b>Notes</b><br>'
                '• BL views are shared across tabs via session_state (build in BL Lab).<br>'
                '• For large universes (30+), diagnostics may be computationally heavy; reduce assets or increase caching TTL if needed.<br>'
                '• For indexes/ETFs without market caps, BL falls back to equal-weight market prior.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
