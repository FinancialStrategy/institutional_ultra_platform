# ==============================================================================
# QUANTUM | Global Institutional Terminal
# Robust Streamlit App (Fixed st.set_page_config ordering + stability enhancements)
# NOTE: st.set_page_config MUST be the first Streamlit command executed.
# ==============================================================================

import streamlit as st

# --- 1) STREAMLIT PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="QUANTUM | Global Institutional Terminal",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# --- 2) STANDARD LIBRARIES & THIRD-PARTY IMPORTS ---
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats


# --- 3) CUSTOM CSS (Professional White Theme) ---
st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
            font-family: 'Roboto', sans-serif;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f2937;
            border-bottom: 3px solid #0056b3;
            padding-bottom: 10px;
            margin-bottom: 20px;
            letter-spacing: 1px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            color: #000000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0056b3;
            color: white;
        }
        /* Tables */
        div[data-testid="stDataFrame"] {
            width: 100%;
        }
        .risk-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .risk-low { background-color: #d1fae5; color: #065f46; }
        .risk-medium { background-color: #fef3c7; color: #92400e; }
        .risk-high { background-color: #fee2e2; color: #991b1b; }
        .data-warning { 
            background-color: #fef3c7; 
            border-left: 4px solid #f59e0b;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .data-success { 
            background-color: #d1fae5; 
            border-left: 4px solid #10b981;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 4) ENHANCED DATA MANAGER WITH GLOBAL ASSETS & ROBUST DATA ALIGNMENT
#   - Fix: keep cached function "pure" (no st.* calls inside cache)
#   - UI messages are returned via a report and rendered outside cache
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
                # Equity ETFs
                "SPDR S&P 500 ETF (SPY)": "SPY",
                "Invesco QQQ Trust (QQQ)": "QQQ",
                "iShares Russell 2000 (IWM)": "IWM",
                "Vanguard Total Stock Market (VTI)": "VTI",
                "ARK Innovation ETF (ARKK)": "ARKK",
                # Sector ETFs
                "Financial Select Sector SPDR (XLF)": "XLF",
                "Technology Select Sector SPDR (XLK)": "XLK",
                "Energy Select Sector SPDR (XLE)": "XLE",
                "Health Care Select Sector SPDR (XLV)": "XLV",
                # Commodity ETFs
                "SPDR Gold Shares (GLD)": "GLD",
                "iShares Silver Trust (SLV)": "SLV",
                "United States Copper Index (CPER)": "CPER",
                "VanEck Gold Miners (GDX)": "GDX",
                # Bond/Fixed Income
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
                # Financials
                "Mitsubishi UFJ Financial Group": "8306.T",
                "Sumitomo Mitsui Financial Group": "8316.T",
                "Mizuho Financial Group": "8411.T",
                "Nomura Holdings": "8604.T",
                "Daiwa Securities Group": "8601.T",
                # Industrials & Manufacturers
                "Toyota Motor": "7203.T",
                "Sony Group": "6758.T",
                "Hitachi": "6501.T",
                "Mitsubishi Corporation": "8058.T",
                "Honda Motor": "7267.T",
                "Nintendo": "7974.T",
                "Panasonic": "6752.T",
                "Canon": "7751.T",
                # Holdings
                "SoftBank Group": "9984.T",
                "Mitsubishi Estate": "8802.T",
                "Mitsui & Co.": "8031.T"
            },
            "Australia (Major Stocks)": {
                # Banks & Financials
                "Commonwealth Bank of Australia": "CBA.AX",
                "Westpac Banking Corporation": "WBC.AX",
                "Australia and New Zealand Banking Group": "ANZ.AX",
                "National Australia Bank": "NAB.AX",
                "Macquarie Group": "MQG.AX",
                # Industrial Holdings
                "BHP Group": "BHP.AX",
                "Rio Tinto": "RIO.AX",
                "Fortescue Metals Group": "FMG.AX",
                "CSL Limited": "CSL.AX",
                "Wesfarmers": "WES.AX",
                "Woolworths Group": "WOW.AX",
                "Transurban Group": "TCL.AX",
                # Mining & Metals
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
    IMPORTANT: No Streamlit UI calls inside cached function.

    Returns:
        df_prices: Cleaned price DataFrame with aligned dates
        benchmark_series: S&P 500 benchmark data (if available)
        data_quality_report: Dictionary with data quality metrics + warnings
    """
    warnings_list: List[str] = []
    infos_list: List[str] = []

    if not selected_tickers:
        return pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["No tickers selected."]}

    benchmark_ticker = "^GSPC"
    all_tickers = list(set(list(selected_tickers) + [benchmark_ticker]))

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Download data (with timeout fallback for yfinance version differences)
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
        # Some yfinance versions may not support `timeout`
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
                # Single-ticker or flat columns
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

    # Remove tickers with insufficient data
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

    # Fill missing values and drop remaining NA rows
    df_filled = df_filtered.ffill().bfill()
    df_clean = df_filled.dropna()

    if len(df_clean) < int(min_data_length):
        return pd.DataFrame(), pd.Series(dtype=float), {
            "warnings": [f"Insufficient data after cleaning. Only {len(df_clean)} data points available."],
            "total_rows": int(len(df_clean))
        }

    # Separate benchmark from portfolio assets
    selected = list(dict.fromkeys(selected_tickers))  # de-duplicate while preserving order
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

    # Final data quality report
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


class EnhancedDataQualityUI:
    @staticmethod
    def display_data_quality_report(dm: "EnhancedDataManager", report: Dict):
        """Display comprehensive data quality report."""
        if not report:
            return

        with st.expander("📊 Data Quality & Alignment Report", expanded=True):
            if report.get("alignment_status") == "SUCCESS":
                st.markdown('<div class="data-success">✅ Data Alignment Successful</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="data-warning">⚠️ Data Alignment Warning</div>', unsafe_allow_html=True)

            # Show info/warning messages captured from cached function
            if report.get("infos"):
                for msg in report["infos"][:6]:
                    st.info(msg)

            if report.get("warnings"):
                for msg in report["warnings"][:10]:
                    st.warning(msg)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Tickers", report.get("initial_tickers", 0))
                st.metric("Valid Tickers", report.get("valid_tickers", 0))
            with col2:
                st.metric("Final Tickers", report.get("final_tickers", 0))
                st.metric("Data Points", report.get("total_rows", 0))
            with col3:
                st.metric("Start Date", report.get("start_date", "N/A"))
                st.metric("End Date", report.get("end_date", "N/A"))
            with col4:
                st.metric("Data Range", f"{report.get('data_range_days', 0):,} days")
                st.metric("Alignment", report.get("alignment_status", "UNKNOWN"))

            st.subheader("📈 Data Completeness by Asset")
            if "missing_data_summary" in report:
                completeness_data = []
                ticker_map = dm.get_ticker_name_map()

                for ticker, missing_pct in report["missing_data_summary"].items():
                    details = report.get("ticker_details", {}).get(ticker, {})
                    completeness_data.append({
                        "Ticker": ticker,
                        "Asset Name": ticker_map.get(ticker, ticker),
                        "Data Points": details.get("non_na_count", 0),
                        "Missing %": float(missing_pct),
                        "Start Date": details.get("start_date", "N/A"),
                        "End Date": details.get("end_date", "N/A"),
                    })

                if completeness_data:
                    df_completeness = pd.DataFrame(completeness_data).sort_values("Missing %", ascending=True)

                    def color_missing(val):
                        if val < 5:
                            return "background-color: #d1fae5; color: #065f46"
                        elif val < 20:
                            return "background-color: #fef3c7; color: #92400e"
                        return "background-color: #fee2e2; color: #991b1b"

                    styled_df = (
                        df_completeness.style
                        .format({"Data Points": "{:,}", "Missing %": "{:.1f}%"})
                        .applymap(color_missing, subset=["Missing %"])
                    )
                    st.dataframe(styled_df, use_container_width=True, height=300)

            st.subheader("🔗 Time Series Alignment Statistics")
            if "ticker_details" in report and "missing_data_summary" in report:
                alignment_stats = []
                for ticker, details in report["ticker_details"].items():
                    if ticker in report.get("missing_data_summary", {}):
                        alignment_stats.append({
                            "Ticker": ticker,
                            "Length": details.get("original_length", 0),
                            "Valid Points": details.get("non_na_count", 0),
                            "Missing": details.get("na_count", 0),
                            "Data Days": details.get("data_days", 0),
                        })

                if alignment_stats:
                    df_stats = pd.DataFrame(alignment_stats)
                    st.dataframe(
                        df_stats.style.format({
                            "Length": "{:,}",
                            "Valid Points": "{:,}",
                            "Missing": "{:,}",
                            "Data Days": "{:,}",
                        }),
                        use_container_width=True
                    )


# ==============================================================================
# 5) TRY TO IMPORT OPTIMIZATION LIBRARIES WITH MULTI-PATH FALLBACKS
#    - No assets removed; only import robustness improved
# ==============================================================================

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
    BLACK_LITTERMAN_AVAILABLE = False
    HRP_AVAILABLE = False
    # We intentionally do NOT call st.error here at import time beyond set_page_config + CSS.
    # We'll surface this message in the UI within main().
    _OPT_IMPORT_ERROR = str(e)


# ==============================================================================
# 6) ENHANCED OPTIMIZATION ENGINE WITH DATA VALIDATION
# ==============================================================================

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

    def optimize_black_litterman(self, views_dict, confidences):
        if not BLACK_LITTERMAN_AVAILABLE:
            st.error("Black-Litterman model not available. Install a compatible PyPortfolioOpt version.")
            return self.optimize_mean_variance("max_sharpe")

        try:
            # omega="idzorek" expects confidences aligned with views; keep order stable
            view_assets = list(views_dict.keys())
            view_conf = [float(confidences[a]) for a in view_assets]

            bl = BlackLittermanModel(
                cov_matrix=self.S,
                pi="equal",
                absolute_views=views_dict,
                omega="idzorek",
                view_confidences=view_conf
            )
            bl_returns = bl.bl_returns()
            ef = EfficientFrontier(bl_returns, self.S)
            ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False)

            return cleaned_weights, perf, bl_returns

        except Exception as e:
            st.error(f"Black-Litterman optimization failed: {e}")
            return self.optimize_mean_variance("max_sharpe")


# ==============================================================================
# 7) ENHANCED RISK METRICS ENGINE
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
# 8) ENHANCED STRESS TESTING MODULE
# ==============================================================================

class EnhancedStressTestLab:
    @staticmethod
    def run_custom_stress_test(portfolio_returns: pd.Series, shock_params: Dict) -> Dict:
        base_vol = float(portfolio_returns.std())

        if shock_params["shock_type"] == "market_crash":
            shock_return = -shock_params["magnitude"] / shock_params["duration"]
            simulated = pd.Series([shock_return] * shock_params["duration"])
            total_return = (1 + simulated).prod() - 1
            max_dd = total_return

        elif shock_params["shock_type"] == "volatility_spike":
            shock_vol = base_vol * (1 + shock_params["magnitude"])
            simulated = pd.Series(np.random.normal(portfolio_returns.mean(), shock_vol, shock_params["duration"]))
            total_return = (1 + simulated).prod() - 1
            max_dd = ((1 + simulated).cumprod() / (1 + simulated).cumprod().cummax() - 1).min()

        else:
            simulated = portfolio_returns.tail(shock_params["duration"]).copy()
            total_return = (1 + simulated).prod() - 1
            max_dd = ((1 + simulated).cumprod() / (1 + simulated).cumprod().cummax() - 1).min()

        vol = float(simulated.std() * np.sqrt(252)) if len(simulated) > 0 else 0.0

        return {
            "Scenario": f"Custom: {shock_params['shock_type'].replace('_', ' ').title()}",
            "Magnitude": f"{shock_params['magnitude']:.1%}",
            "Duration": f"{shock_params['duration']} days",
            "Total Return": float(total_return),
            "Max Drawdown": float(max_dd),
            "Volatility (Ann.)": float(vol),
        }

    @staticmethod
    def run_historical_stress_test(portfolio_returns: pd.Series) -> pd.DataFrame:
        historical_scenarios = {
            "COVID-19 Crash (2020)": {"start": "2020-02-19", "end": "2020-03-23"},
            "2022 Inflation/Correction": {"start": "2022-01-03", "end": "2022-06-30"},
            "2018 Trade War": {"start": "2018-10-01", "end": "2018-12-24"},
            "Turkey 2018 Currency Crisis": {"start": "2018-08-01", "end": "2018-09-01"},
            "Banking Turmoil (2023)": {"start": "2023-03-01", "end": "2023-03-31"},
        }

        results = []
        for name, params in historical_scenarios.items():
            mask = (portfolio_returns.index >= params["start"]) & (portfolio_returns.index <= params["end"])
            period_returns = portfolio_returns.loc[mask]
            if len(period_returns) > 5:
                total_return = (1 + period_returns).prod() - 1
                max_dd = ((1 + period_returns).cumprod() / (1 + period_returns).cumprod().cummax() - 1).min()
                vol = period_returns.std() * np.sqrt(252)
                results.append({
                    "Scenario": name,
                    "Period": f"{params['start']} to {params['end']}",
                    "Total Return": float(total_return),
                    "Max Drawdown": float(max_dd),
                    "Volatility (Ann.)": float(vol),
                })

        return pd.DataFrame(results)


# ==============================================================================
# 9) VISUALIZATION ENGINE
# ==============================================================================

class EnhancedChartFactory:
    @staticmethod
    def plot_efficient_frontier(mu: pd.Series, S: pd.DataFrame, optimal_weights: Optional[np.ndarray] = None):
        """
        Plot efficient frontier with feasible set + optional optimal portfolio star.
        Fix: correct portfolio vol calculation (vectorized quadratic form).
        """
        mu_vec = mu.values if isinstance(mu, pd.Series) else np.asarray(mu)
        S_mat = S.values if isinstance(S, pd.DataFrame) else np.asarray(S)

        n_assets = len(mu_vec)
        n_samples = 5000

        w = np.random.dirichlet(np.ones(n_assets), n_samples)
        rets = w @ mu_vec
        vars_ = np.einsum("ij,jk,ik->i", w, S_mat, w)
        stds = np.sqrt(np.maximum(vars_, 0))
        sharpes = np.divide(rets, stds, out=np.zeros_like(rets), where=stds > 0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stds,
            y=rets,
            mode="markers",
            marker=dict(size=5, color=sharpes, colorscale="Viridis", showscale=True),
            name="Feasible Set"
        ))

        if optimal_weights is not None:
            ow = np.asarray(optimal_weights).reshape(-1)
            opt_ret = float(ow @ mu_vec)
            opt_var = float(ow.T @ S_mat @ ow)
            opt_vol = float(np.sqrt(max(opt_var, 0)))
            fig.add_trace(go.Scatter(
                x=[opt_vol], y=[opt_ret],
                mode="markers",
                marker=dict(symbol="star", size=25, color="#e63946"),
                name="Optimal Portfolio"
            ))

        fig.update_layout(
            title="Efficient Frontier Analysis",
            xaxis_title="Volatility",
            yaxis_title="Return",
            template="plotly_white",
            height=500
        )
        return fig

    @staticmethod
    def plot_drawdown(returns_series: pd.Series, title: str = "Portfolio Drawdown"):
        wealth_index = (1 + returns_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", fillcolor="rgba(230, 57, 70, 0.2)",
            line=dict(color="#e63946", width=2),
            name="Drawdown"
        ))
        fig.update_layout(
            title=title,
            yaxis_tickformat=".1%",
            template="plotly_white",
            height=400
        )
        return fig

@staticmethod
def plot_alignment_heatmap(df_prices: pd.DataFrame):
    availability = pd.DataFrame(index=df_prices.index, columns=df_prices.columns)
    for col in df_prices.columns:
        availability[col] = df_prices[col].notna().astype(int)

    # Use Graph Objects to avoid Plotly Express/Narwhals duplicate-column constraints
    fig = go.Figure(data=go.Heatmap(
        z=availability.T.values,
        x=availability.index,
        y=availability.columns,
        colorscale=[[0, "red"], [1, "green"]],
        zmin=0, zmax=1,
        hovertemplate="Date: %{x}<br>Asset: %{y}<br>Available: %{z}<extra></extra>"
    ))
    fig.update_layout(
        title="Data Availability Heatmap (Green = Available)",
        height=400,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Asset"
    )
    return fig


# ==============================================================================
# 10) MAIN APPLICATION
# ==============================================================================

def main():
    st.markdown('<div class="main-header">🌐 QUANTUM | Global Institutional Analytics Platform</div>', unsafe_allow_html=True)

    # Initialize data manager
    dm = EnhancedDataManager()

    # Session state init
    if "custom_shocks" not in st.session_state:
        st.session_state.custom_shocks = []
    if "bl_views" not in st.session_state:
        st.session_state.bl_views = {}
    if "bl_conf" not in st.session_state:
        st.session_state.bl_conf = {}
    if "selected_assets_preset" not in st.session_state:
        st.session_state.selected_assets_preset = None

    with st.sidebar:
        st.header("🌍 Global Asset Selection")

        # Quick portfolio presets (kept, improved: feed into defaults)
        st.subheader("Quick Portfolios")
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
            # Do not remove default behavior; only add preset override
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

        # De-duplicate tickers (same ticker can appear in multiple universe groups, e.g., TLT)
        if selected_assets:
            before = len(selected_assets)
            selected_assets = list(dict.fromkeys(selected_assets))
            after = len(selected_assets)
            if after < before:
                st.info(
                    f"ℹ️ Removed {before-after} duplicate ticker selection(s) (same instrument selected from multiple groups)."
                )

        st.divider()

        # Data settings
        st.subheader("Data Settings")
        start_date = st.date_input("Start Date", value=datetime(2018, 1, 1))
        min_data_length = st.slider("Minimum Data Points", 100, 1000, 252,
                                   help="Assets with fewer data points will be removed")

        # Show regional exposure
        if selected_assets:
            exposure = dm.get_regional_exposure(selected_assets)
            st.subheader("🌐 Regional Exposure")
            for region, pct in exposure.items():
                st.progress(pct / 100, text=f"{region}: {pct:.1f}%")

        st.divider()
        st.caption("Tip: If Streamlit Cloud shows a redacted ModuleNotFoundError, add a requirements.txt with the packages used.")

    if not selected_assets:
        st.warning("Please select at least one asset from the sidebar.")
        return

    if not OPTIMIZATION_AVAILABLE:
        st.error("⚠️ PyPortfolioOpt is not installed/available. Optimization features will be disabled.")
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

    # Display data quality report
    if data_report:
        EnhancedDataQualityUI.display_data_quality_report(dm, data_report)

    if df_prices is None or df_prices.empty:
        st.error("❌ No valid data available after alignment. Please select different assets or adjust date range.")
        return

    st.success(f"✅ Data ready for analysis: {len(df_prices)} data points, {len(df_prices.columns)} assets")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview",
        "🎯 Portfolio Optimization",
        "⚠️ Stress Testing",
        "📊 Risk Analytics",
        "🔗 Correlation Analysis",
    ])

    # --- TAB 1: DATA OVERVIEW ---
    with tab1:
        st.subheader("📊 Data Overview & Visualization")

        st.markdown("### 📈 Normalized Price Performance")
        normalized = (df_prices / df_prices.iloc[0]) * 100

        fig_prices = px.line(
            normalized,
            title="All Assets Rebased to 100",
            labels={"value": "Index Value", "variable": "Asset"}
        )
        fig_prices.update_layout(
            template="plotly_white",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig_prices, use_container_width=True)

        st.markdown("### 🔍 Data Availability Heatmap")
        fig_heatmap = EnhancedChartFactory.plot_alignment_heatmap(df_prices)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("### 📊 Summary Statistics")
        returns = df_prices.pct_change().dropna()
        # Safety: ensure unique columns (same ticker can be selected multiple times)
        returns = returns.loc[:, ~returns.columns.duplicated()]
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Mean Return (Ann.)", f"{returns.mean().mean() * 252:.2%}")
        with col_stat2:
            st.metric("Avg Volatility (Ann.)", f"{returns.std().mean() * np.sqrt(252):.2%}")
        with col_stat3:
            st.metric("Avg Correlation", f"{returns.corr().values.mean():.2f}")

        with st.expander("🔎 Raw Data Preview (Prices & Returns)", expanded=False):
            st.dataframe(df_prices.tail(25), use_container_width=True, height=300)
            st.dataframe(returns.tail(25), use_container_width=True, height=300)

    # --- TAB 2: PORTFOLIO OPTIMIZATION ---
    with tab2:
        if not OPTIMIZATION_AVAILABLE:
            st.warning("Portfolio optimization requires PyPortfolioOpt. Add it to requirements.txt: PyPortfolioOpt")
        else:
            st.subheader("🎯 Portfolio Construction & Optimization")

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

            # Black-Litterman views (persisted)
            if strategy == "Black-Litterman (Views)":
                if not BLACK_LITTERMAN_AVAILABLE:
                    st.warning("Black-Litterman is not available in your installed PyPortfolioOpt version.")
                else:
                    st.subheader("Black-Litterman Views Configuration")
                    col_bl1, col_bl2 = st.columns(2)

                    with col_bl1:
                        st.markdown("**Add Your Views**")
                        view_asset = st.selectbox("Asset", options=df_prices.columns.tolist(), key="bl_asset")
                        view_return = st.number_input("Expected Return (%)", value=10.0, step=1.0, key="bl_ret") / 100
                        confidence = st.slider("Confidence Level", 0.1, 1.0, 0.7, key="bl_conf")

                        if st.button("Add View", use_container_width=True):
                            st.session_state.bl_views[view_asset] = float(view_return)
                            st.session_state.bl_conf[view_asset] = float(confidence)

                        if st.button("Clear All Views", use_container_width=True):
                            st.session_state.bl_views = {}
                            st.session_state.bl_conf = {}

                    with col_bl2:
                        if st.session_state.bl_views:
                            st.markdown("**Current Views**")
                            for asset, ret in st.session_state.bl_views.items():
                                conf = st.session_state.bl_conf.get(asset, 0.0)
                                st.write(f"{asset}: {ret:.2%} return (Confidence: {conf:.0%})")
                        else:
                            st.info("No views added yet.")

            if st.button("Run Optimization", type="primary", use_container_width=True):
                with st.spinner("Running optimization..."):
                    try:
                        engine = EnhancedOptimizationEngine(df_prices)

                        if strategy == "Hierarchical Risk Parity (HRP)":
                            weights, perf = engine.optimize_hrp()

                        elif strategy == "Black-Litterman (Views)" and BLACK_LITTERMAN_AVAILABLE and st.session_state.bl_views:
                            weights, perf, _ = engine.optimize_black_litterman(st.session_state.bl_views, st.session_state.bl_conf)

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

                        st.success("✅ Optimization completed!")

                        col_res1, col_res2 = st.columns([1, 1])
                        ticker_map = dm.get_ticker_name_map()

                        with col_res1:
                            st.subheader("📊 Portfolio Allocation")
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
                                st.warning("No non-trivial weights returned (all ~0).")
                            else:
                                display_df = portfolio_df[["Asset", "Ticker", "Weight", "Amount ($)"]].copy()
                                display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x:.2%}")
                                display_df["Amount ($)"] = display_df["Amount ($)"].apply(lambda x: f"${x:,.0f}")
                                st.dataframe(display_df, use_container_width=True, height=420)

                        with col_res2:
                            st.subheader("📈 Allocation Chart")
                            if not portfolio_df.empty:
                                fig_pie = px.pie(portfolio_df, values="Weight", names="Asset", hole=0.4)
                                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                                fig_pie.update_layout(template="plotly_white", height=420, showlegend=False)
                                st.plotly_chart(fig_pie, use_container_width=True)

                        st.divider()
                        st.subheader("🎯 Performance Summary")
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

                        # Efficient frontier visualization (kept as an enhancement; does not remove any features)
                        st.markdown("### 🧭 Efficient Frontier (Feasible Set + Optimal Point)")
                        w_vec = np.array([weights.get(t, 0.0) for t in engine.mu.index], dtype=float)
                        fig_ef = EnhancedChartFactory.plot_efficient_frontier(engine.mu, engine.S, optimal_weights=w_vec)
                        st.plotly_chart(fig_ef, use_container_width=True)

                        # Store session state for other tabs
                        st.session_state.portfolio_returns = portfolio_returns_series
                        st.session_state.weights = weights

                    except Exception as e:
                        st.error(f"Optimization failed: {e}")

    # --- TAB 3: STRESS TESTING ---
    with tab3:
        st.subheader("⚠️ Stress Testing Laboratory")

        if "portfolio_returns" not in st.session_state:
            st.warning("Please run portfolio optimization first to generate portfolio returns for stress testing.")
        else:
            portfolio_returns = st.session_state.portfolio_returns

            st.markdown("### 📜 Historical Stress Scenarios")
            historical_df = EnhancedStressTestLab.run_historical_stress_test(portfolio_returns)

            if not historical_df.empty:
                st.dataframe(
                    historical_df.style.format({
                        "Total Return": "{:.2%}",
                        "Max Drawdown": "{:.2%}",
                        "Volatility (Ann.)": "{:.2%}",
                    }).background_gradient(cmap="RdYlGn_r", subset=["Total Return"]),
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("No historical scenario windows matched the available portfolio return dates.")

            st.divider()
            st.markdown("### 🛠️ Custom Stress Test Builder")

            col_custom1, col_custom2, col_custom3 = st.columns(3)
            with col_custom1:
                shock_type = st.selectbox("Shock Type", ["market_crash", "volatility_spike", "sector_specific", "liquidity_crunch"])
            with col_custom2:
                magnitude = st.slider("Shock Magnitude", 0.1, 1.0, 0.3, 0.1, format="%.1f")
            with col_custom3:
                duration = st.slider("Duration (days)", 5, 250, 30)

            if st.button("Run Custom Stress Test", type="primary"):
                shock_params = {"shock_type": shock_type, "magnitude": float(magnitude), "duration": int(duration)}
                custom_result = EnhancedStressTestLab.run_custom_stress_test(portfolio_returns, shock_params)

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Scenario", custom_result["Scenario"])
                col_res2.metric("Estimated Loss", f"{custom_result['Total Return']:.2%}")
                col_res3.metric("Max Drawdown", f"{custom_result['Max Drawdown']:.2%}")

                st.session_state.custom_shocks.append(custom_result)

            if st.session_state.custom_shocks:
                st.markdown("### 🧾 Custom Stress Test History")
                df_custom = pd.DataFrame(st.session_state.custom_shocks)
                st.dataframe(
                    df_custom.style.format({
                        "Total Return": "{:.2%}",
                        "Max Drawdown": "{:.2%}",
                        "Volatility (Ann.)": "{:.2%}",
                    }),
                    use_container_width=True,
                    height=250
                )

            st.divider()
            st.markdown("### 📉 Drawdown Analysis")
            fig_dd = EnhancedChartFactory.plot_drawdown(portfolio_returns)
            st.plotly_chart(fig_dd, use_container_width=True)

    # --- TAB 4: RISK ANALYTICS ---
    with tab4:
        st.subheader("🛡️ Risk Analytics")

        if "portfolio_returns" not in st.session_state:
            st.warning("Please run portfolio optimization first.")
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

            st.markdown("### 📊 Risk & Performance Metrics")
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

            st.divider()
            st.markdown("### 💰 Value at Risk Analysis")

            col_var1, col_var2 = st.columns(2)
            with col_var1:
                confidence = st.select_slider("Confidence Level", [0.90, 0.95, 0.99], value=0.95)
            with col_var2:
                holding_days = st.selectbox("Holding Period", [1, 5, 10, 20], index=0)

            alpha = 1 - float(confidence)
            scale = float(np.sqrt(int(holding_days)))

            if len(portfolio_returns) >= 30:
                var_pct = float(-np.percentile(portfolio_returns, alpha * 100) * scale)
                var_value = float(var_pct * amt_for_var)
            else:
                var_pct = float("nan")
                var_value = float("nan")

            col_var_disp1, col_var_disp2 = st.columns(2)
            col_var_disp1.metric(f"{confidence:.0%} VaR", f"{var_pct:.2%}" if np.isfinite(var_pct) else "N/A")
            col_var_disp2.metric(f"{confidence:.0%} VaR (${amt_for_var:,.0f})", f"${var_value:,.0f}" if np.isfinite(var_value) else "N/A")

            # Distribution diagnostic (non-destructive enhancement)
            st.markdown("### 📉 Return Distribution Diagnostic")
            hist_fig = px.histogram(
                portfolio_returns.dropna(),
                nbins=60,
                title="Portfolio Daily Returns Histogram"
            )
            hist_fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(hist_fig, use_container_width=True)

    
    # --- TAB 5: CORRELATION ANALYSIS ---
    with tab5:
        st.subheader("🔗 Correlation Analysis")

        returns = df_prices.pct_change().dropna()
        # Safety: enforce unique columns (duplicate ticker selection can create duplicate columns)
        returns = returns.loc[:, ~returns.columns.duplicated()]

        if returns.empty or returns.shape[1] < 2:
            st.warning("Please select at least two assets with sufficient overlapping data to compute correlations.")
        else:
            st.markdown("### 📊 Correlation Matrix")
            corr_matrix = returns.corr()

            # Build unique display labels (avoid duplicates after mapping tickers -> names)
            ticker_map = dm.get_ticker_name_map()
            labels_raw = [ticker_map.get(t, t) for t in corr_matrix.columns]
            seen = {}
            labels = []
            for lab in labels_raw:
                if lab not in seen:
                    seen[lab] = 1
                    labels.append(lab)
                else:
                    seen[lab] += 1
                    labels.append(f"{lab} ({seen[lab]})")

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
                xaxis_title="Assets",
                yaxis_title="Assets",
                xaxis=dict(tickangle=-45)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("### 📈 Highest & Lowest Correlations")

            # Compute unique pairs (avoid counting (A,B) and (B,A) separately)
            pairs = {}
            for i, a in enumerate(corr_matrix.columns):
                for j, b in enumerate(corr_matrix.columns):
                    if j <= i:
                        continue
                    pairs[(a, b)] = corr_matrix.iloc[i, j]

            if not pairs:
                st.info("Not enough assets to compute pairwise correlations.")
            else:
                pairs_sorted = sorted(pairs.items(), key=lambda kv: kv[1])
                lowest = pairs_sorted[:10]
                highest = pairs_sorted[-10:][::-1]

                col_corr1, col_corr2 = st.columns(2)

                with col_corr1:
                    st.subheader("Highest Correlations")
                    for (a, b), value in highest:
                        name_a = ticker_map.get(a, a)
                        name_b = ticker_map.get(b, b)
                        st.metric(f"{name_a} & {name_b}", f"{value:.3f}")

                with col_corr2:
                    st.subheader("Lowest Correlations")
                    for (a, b), value in lowest:
                        name_a = ticker_map.get(a, a)
                        name_b = ticker_map.get(b, b)
                        st.metric(f"{name_a} & {name_b}", f"{value:.3f}")


if __name__ == "__main__":
    main()
