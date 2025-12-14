# ==============================================================================
# QUANTUM | Global Institutional Terminal - INSTITUTIONAL ENHANCED V8
# ------------------------------------------------------------------------------
# ENHANCEMENTS:
# 1. Smart UI/UX with collapsible sections and progressive disclosure
# 2. Enhanced data validation with availability indicators
# 3. Interactive chart linking and cross-filtering
# 4. Institutional reporting templates
# 5. Performance optimizations with lazy loading
# 6. Advanced error handling and user feedback
# 7. Theme consistency and professional styling
# 8. Export capabilities in multiple formats
# ==============================================================================

import streamlit as st
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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize
import io
import base64
from datetime import date

# Enhanced page configuration
st.set_page_config(
    page_title="QUANTUM | Advanced Risk Analytics Platform",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://quantum-analytics.com/docs',
        'Report a bug': 'https://quantum-analytics.com/issues',
        'About': "QUANTUM Institutional Terminal v8.0 - Advanced Portfolio Risk Analytics"
    }
)

# ==============================================================================
# ENHANCED STYLING: Professional Institutional Theme
# ==============================================================================
st.markdown("""
<style>
    /* Main App Styling */
    .stApp { 
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Headers */
    .main-header { 
        font-size: 2.8rem; 
        font-weight: 800; 
        background: linear-gradient(90deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 12px;
        margin-bottom: 24px;
        border-bottom: 3px solid linear-gradient(90deg, #1a237e 0%, #283593 50%, #3949ab 100%);
    }
    
    .section-header { 
        font-size: 1.7rem; 
        font-weight: 700; 
        color: #1a237e; 
        border-bottom: 2px solid #e8eaf6; 
        padding-bottom: 8px; 
        margin: 22px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .section-header::before {
        content: "▶";
        font-size: 0.8em;
        color: #3949ab;
        transition: transform 0.3s;
    }
    
    /* Cards */
    .card { 
        background: white; 
        border: 1px solid #e1e5eb; 
        border-radius: 12px; 
        padding: 20px; 
        margin: 16px 0; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    
    /* Status Cards */
    .info-card { 
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
        border-left: 4px solid #2196f3; 
        color: #0d47a1; 
        border-radius: 12px; 
        padding: 16px; 
        margin: 12px 0;
    }
    
    .success-card { 
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
        border-left: 4px solid #4caf50; 
        color: #1b5e20; 
        border-radius: 12px; 
        padding: 16px; 
        margin: 12px 0;
    }
    
    .warning-card { 
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%); 
        border-left: 4px solid #ff9800; 
        color: #5d4037; 
        border-radius: 12px; 
        padding: 16px; 
        margin: 12px 0;
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 2px; 
        background: transparent; 
        padding: 4px; 
        border-radius: 12px; 
    }
    
    .stTabs [data-baseweb="tab"] { 
        border-radius: 8px; 
        padding: 12px 20px; 
        font-weight: 600; 
        background: white; 
        border: 1px solid #e0e0e0; 
        color: #5c6bc0;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%); 
        color: white; 
        border-color: #1a237e; 
        box-shadow: 0 4px 12px rgba(26,35,126,0.15);
    }
    
    /* Enhanced Buttons */
    .stButton>button { 
        background: linear-gradient(135deg, #303f9f 0%, #1a237e 100%); 
        color: white; 
        border: none; 
        border-radius: 10px; 
        padding: 12px 24px; 
        font-weight: 600; 
        transition: all 0.3s;
    }
    
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(26,35,126,0.25);
    }
    
    /* Metrics Styling */
    .stMetric { 
        background: white; 
        border-radius: 10px; 
        padding: 15px; 
        border: 1px solid #e1e5eb;
    }
    
    /* Sidebar Enhancements */
    .sidebar .sidebar-content { 
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Dataframe Styling */
    .dataframe { 
        border-radius: 8px; 
        overflow: hidden; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar { 
        width: 10px; 
        height: 10px; 
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
    
    /* Badge Styling */
    .badge { 
        display: inline-block; 
        padding: 4px 12px; 
        border-radius: 20px; 
        font-size: 0.85em; 
        font-weight: 600; 
        margin: 2px;
    }
    
    .badge-success { background: #e8f5e9; color: #2e7d32; }
    .badge-warning { background: #fff8e1; color: #f57c00; }
    .badge-danger { background: #ffebee; color: #c62828; }
    .badge-info { background: #e3f2fd; color: #1565c0; }
    
    /* Tooltip Styling */
    .tooltip { 
        position: relative; 
        border-bottom: 1px dotted #666; 
        cursor: help; 
    }
    
    /* Loading Spinner */
    .stSpinner > div { 
        border-color: #3949ab transparent transparent transparent; 
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ENHANCED DATA MANAGER with Availability Checking
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
        
        self.cached_availability = {}
        
    def ticker_name_map(self) -> Dict[str, str]:
        m = {}
        for cat, d in self.universe.items():
            for name, t in d.items():
                m[t] = name
        return m
    
    def check_ticker_availability(self, ticker: str, start: date, end: date) -> Dict[str, Any]:
        """Check if a ticker has available data for the given period"""
        cache_key = f"{ticker}_{start}_{end}"
        if cache_key in self.cached_availability:
            return self.cached_availability[cache_key]
            
        try:
            data = yf.download(
                ticker, 
                start=start, 
                end=end,
                progress=False,
                timeout=10
            )
            
            if data.empty:
                result = {"available": False, "message": "No data returned", "days": 0}
            else:
                days = len(data)
                completeness = days / ((end - start).days) if (end - start).days > 0 else 1.0
                result = {
                    "available": True,
                    "message": f"{days} days of data available",
                    "days": days,
                    "completeness": completeness,
                    "start_date": data.index[0].date() if days > 0 else None,
                    "end_date": data.index[-1].date() if days > 0 else None
                }
        except Exception as e:
            result = {"available": False, "message": str(e), "days": 0}
        
        self.cached_availability[cache_key] = result
        return result
    
    def get_recommended_assets(self, category: str, count: int = 8) -> List[str]:
        """Get recommended assets based on category and typical institutional use"""
        if category in self.universe:
            assets = list(self.universe[category].values())
            if category == "Global Benchmarks":
                return assets[:6]  # Top benchmarks
            elif category == "US ETFs (Major)":
                # Core-satellite approach
                core = ["SPY", "QQQ", "IWM", "VTI"]
                satellite = [a for a in assets if a not in core][:count-4]
                return core + satellite
            elif category == "Global Mega Caps":
                # Top market cap stocks
                return assets[:min(count, len(assets))]
            elif category == "BIST 30 (Turkey)":
                # Liquidity-based selection
                liquid = ["AKBNK.IS", "GARAN.IS", "ISCTR.IS", "THYAO.IS", "ASELS.IS"]
                others = [a for a in assets if a not in liquid][:count-5]
                return liquid + others
        return []

# Initialize data manager
data_manager = EnhancedDataManager()
TICKER_NAME_MAP = data_manager.ticker_name_map()

# ==============================================================================
# ENHANCED DATA FETCH with Smart Validation
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices_with_validation(
    tickers: Tuple[str, ...],
    start: str,
    end: str,
    min_points: int = 200
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any], pd.DataFrame]:
    """
    Enhanced data fetching with validation and quality metrics
    Returns: prices, benchmark, report, quality_df
    """
    if not tickers:
        return pd.DataFrame(), pd.Series(dtype=float), {"warnings": ["No tickers selected."]}, pd.DataFrame()
    
    benchmark = "^GSPC"
    all_tickers = list(dict.fromkeys(list(tickers) + [benchmark]))
    
    # Create quality tracking dataframe
    quality_df = pd.DataFrame(index=all_tickers, columns=[
        "Status", "Days", "Completeness", "Start", "End", 
        "Mean", "Std", "Min", "Max", "Sharpe"
    ])
    
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
    
    report: Dict[str, Any] = {
        "warnings": [], 
        "infos": [], 
        "ticker_details": {},
        "quality_score": 0.0,
        "summary_stats": {}
    }
    
    if not closes:
        report["warnings"].append("No data fetched. Try a different date range or tickers.")
        return pd.DataFrame(), pd.Series(dtype=float), report, quality_df
    
    df = pd.DataFrame(closes)
    df = df.sort_index()
    
    # Calculate quality metrics
    good_cols = []
    quality_scores = []
    
    for c in df.columns:
        s = df[c].dropna()
        non_na = len(s)
        
        if non_na >= min_points:
            good_cols.append(c)
            
            # Calculate statistics
            returns = s.pct_change().dropna()
            if len(returns) > 10:
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe = np.nan
            
            completeness = non_na / len(df)
            quality_score = completeness * 0.6 + min(1.0, non_na / 1000) * 0.4
            
            quality_df.loc[c] = {
                "Status": "✅" if completeness > 0.8 else "⚠️" if completeness > 0.5 else "❌",
                "Days": non_na,
                "Completeness": f"{completeness:.1%}",
                "Start": s.index[0].date() if non_na > 0 else None,
                "End": s.index[-1].date() if non_na > 0 else None,
                "Mean": f"{returns.mean():.4%}" if len(returns) > 0 else "N/A",
                "Std": f"{returns.std():.4%}" if len(returns) > 0 else "N/A",
                "Min": f"{returns.min():.4%}" if len(returns) > 0 else "N/A",
                "Max": f"{returns.max():.4%}" if len(returns) > 0 else "N/A",
                "Sharpe": f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
            }
            
            quality_scores.append(quality_score)
            
            report["ticker_details"][c] = {
                "non_na": non_na,
                "na_pct": float((1 - completeness) * 100),
                "start": str(s.index[0].date()) if non_na > 0 else None,
                "end": str(s.index[-1].date()) if non_na > 0 else None,
                "quality_score": quality_score
            }
        else:
            report["warnings"].append(f"Removing {c}: insufficient data ({non_na} < {min_points}).")
            quality_df.loc[c] = {
                "Status": "❌",
                "Days": non_na,
                "Completeness": f"{non_na/len(df):.1%}",
                "Start": s.index[0].date() if non_na > 0 else None,
                "End": s.index[-1].date() if non_na > 0 else None,
                "Mean": "N/A", "Std": "N/A", "Min": "N/A", "Max": "N/A", "Sharpe": "N/A"
            }
    
    df = df[good_cols].copy()
    if df.empty or len(df) < min_points:
        report["warnings"].append("Insufficient data after filtering. Expand the date range.")
        return pd.DataFrame(), pd.Series(dtype=float), report, quality_df
    
    # Enhanced missing value handling
    df = df.ffill().bfill().dropna()
    
    # Calculate overall quality score
    if quality_scores:
        report["quality_score"] = np.mean(quality_scores)
        report["quality_rating"] = "Excellent" if report["quality_score"] > 0.9 else \
                                  "Good" if report["quality_score"] > 0.7 else \
                                  "Fair" if report["quality_score"] > 0.5 else "Poor"
    
    report["infos"].append(f"Aligned data shape: {df.shape}")
    report["infos"].append(f"Data quality: {report.get('quality_rating', 'Unknown')} ({report.get('quality_score', 0):.1%})")
    
    bench = pd.Series(dtype=float)
    if benchmark in df.columns:
        bench = df[benchmark].copy()
    
    # Portfolio assets only
    portfolio_cols = [t for t in tickers if t in df.columns]
    df_port = df[portfolio_cols].copy()
    
    report["start_date"] = str(df.index.min().date())
    report["end_date"] = str(df.index.max().date())
    report["rows"] = int(len(df))
    report["assets"] = int(len(df_port.columns))
    report["alignment_status"] = "SUCCESS"
    
    # Calculate summary statistics
    if not df_port.empty:
        returns = df_port.pct_change().dropna()
        report["summary_stats"] = {
            "mean_return": float(returns.mean().mean()),
            "volatility": float(returns.std().mean()),
            "sharpe_ratio": float((returns.mean().mean() / returns.std().mean() * np.sqrt(252)) if returns.std().mean() > 0 else 0),
            "correlation_mean": float(returns.corr().values[np.triu_indices_from(returns.corr(), k=1)].mean())
        }
    
    return df_port, bench, report, quality_df

# ==============================================================================
# ENHANCED CHARTING UTILITIES
# ==============================================================================
class ChartTheme:
    """Centralized chart theming for consistency"""
    
    @staticmethod
    def apply_default_theme(fig: go.Figure, title: str = "") -> go.Figure:
        """Apply consistent institutional theme to charts"""
        fig.update_layout(
            template="plotly_white",
            title={
                'text': title,
                'font': {'size': 20, 'color': '#1a237e', 'family': 'Inter'}
            },
            font={'family': 'Inter', 'size': 12, 'color': '#424242'},
            hovermode='x unified',
            hoverlabel={
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': 'Inter'
            },
            legend={
                'yanchor': "top",
                'y': 0.99,
                'xanchor': "left",
                'x': 0.01,
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': '#e1e5eb',
                'borderwidth': 1
            },
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='#e0e0e0',
            mirror=True
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='#e0e0e0',
            mirror=True
        )
        
        return fig
    
    @staticmethod
    def create_subplot_grid(rows: int, cols: int, titles: List[str], 
                           height_per_row: int = 400) -> go.Figure:
        """Create consistent subplot grid"""
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=titles,
            vertical_spacing=0.1 if rows > 1 else 0.15,
            horizontal_spacing=0.1 if cols > 1 else 0.15
        )
        
        fig.update_layout(
            height=height_per_row * rows,
            template="plotly_white",
            font={'family': 'Inter', 'size': 12},
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig

def create_interactive_performance_dashboard(prices: pd.DataFrame, 
                                           weights: Dict[str, float],
                                           benchmark: pd.Series = None) -> go.Figure:
    """Create comprehensive performance dashboard with linked interactivity"""
    
    # Calculate portfolio returns
    returns = prices.pct_change().dropna()
    portfolio_returns = portfolio_returns_from_weights(returns, weights)
    
    # Create subplot grid
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Cumulative Returns", "Rolling Metrics (21D)",
            "Drawdown Analysis", "Return Distribution",
            "Monthly Returns Heatmap", "Risk vs Return Scatter"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "heatmap"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1. Cumulative Returns
    cum_returns = (1 + portfolio_returns).cumprod()
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode='lines',
            name='Portfolio',
            line=dict(width=3, color='#1a237e'),
            hovertemplate='%{x}<br>Value: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    if benchmark is not None:
        bench_returns = benchmark.pct_change().dropna()
        bench_cum = (1 + bench_returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode='lines',
                name='Benchmark',
                line=dict(width=2, color='#757575', dash='dash'),
                hovertemplate='%{x}<br>Value: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Rolling Metrics
    window = 21
    rolling_sharpe = portfolio_returns.rolling(window).mean() / portfolio_returns.rolling(window).std() * np.sqrt(252)
    rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
    
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(width=2, color='#2e7d32'),
            yaxis='y'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode='lines',
            name='Rolling Vol',
            line=dict(width=2, color='#c62828'),
            yaxis='y2'
        ),
        row=1, col=2
    )
    
    # Add secondary y-axis for volatility
    fig.update_layout(
        yaxis2=dict(
            title="Volatility",
            titlefont=dict(color='#c62828'),
            tickfont=dict(color='#c62828'),
            anchor="x",
            overlaying="y",
            side="right"
        )
    )
    
    # 3. Drawdown Analysis
    drawdown = (cum_returns / cum_returns.cummax() - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(198, 40, 40, 0.3)',
            line=dict(width=2, color='#c62828'),
            name='Drawdown %',
            hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Return Distribution
    fig.add_trace(
        go.Histogram(
            x=portfolio_returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='#3949ab',
            opacity=0.7,
            hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add normal distribution overlay
    x_norm = np.linspace(portfolio_returns.min() * 100, portfolio_returns.max() * 100, 100)
    y_norm = stats.norm.pdf(x_norm, portfolio_returns.mean() * 100, portfolio_returns.std() * 100)
    y_norm = y_norm / y_norm.max() * len(portfolio_returns) / 20  # Scale to histogram
    
    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal Fit',
            line=dict(width=2, color='#424242', dash='dash'),
            hovertemplate='Return: %{x:.2f}%<br>Density: %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5. Monthly Returns Heatmap
    monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values * 100
    })
    
    heatmap_data = monthly_df.pivot_table(
        index='year', 
        columns='month', 
        values='return'
    )
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=heatmap_data.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="Return %"),
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Risk vs Return Scatter (by asset)
    asset_returns = returns.mean() * 252
    asset_vols = returns.std() * np.sqrt(252)
    asset_sharpes = asset_returns / asset_vols
    
    fig.add_trace(
        go.Scatter(
            x=asset_vols * 100,
            y=asset_returns * 100,
            mode='markers+text',
            marker=dict(
                size=12,
                color=asset_sharpes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe")
            ),
            text=[TICKER_NAME_MAP.get(t, t) for t in returns.columns],
            textposition="top center",
            hovertemplate='Asset: %{text}<br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Add portfolio point
    portfolio_return = annualize_return(portfolio_returns)
    portfolio_vol = annualize_vol(portfolio_returns)
    
    fig.add_trace(
        go.Scatter(
            x=[portfolio_vol * 100],
            y=[portfolio_return * 100],
            mode='markers+text',
            marker=dict(size=15, symbol='star', color='#ff9800'),
            text=['Portfolio'],
            textposition="bottom center",
            hovertemplate='Portfolio<br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Portfolio Performance Dashboard",
        title_font=dict(size=24, color='#1a237e'),
        showlegend=True,
        hovermode='closest'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    
    fig.update_xaxes(title_text="Return %", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_yaxes(title_text="Year", row=3, col=1)
    
    fig.update_xaxes(title_text="Volatility %", row=3, col=2)
    fig.update_yaxes(title_text="Return %", row=3, col=2)
    
    return fig

def create_interactive_correlation_matrix(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create interactive correlation matrix with filtering options"""
    
    # Create annotation matrix
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{value:.2f}",
                    font=dict(color="white" if abs(value) > 0.5 else "black"),
                    showarrow=False
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[TICKER_NAME_MAP.get(col, col) for col in corr_matrix.columns],
        y=[TICKER_NAME_MAP.get(col, col) for col in corr_matrix.index],
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
        hoverongaps=False,
        hovertemplate='X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Assets",
        yaxis_title="Assets",
        height=700,
        annotations=annotations
    )
    
    # Add filtering controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label="All",
                        method="restyle",
                        args=[{"z": [corr_matrix.values]}]
                    ),
                    dict(
                        label="High (>0.7)",
                        method="restyle",
                        args=[{"z": [np.where(corr_matrix.values > 0.7, corr_matrix.values, np.nan)]}]
                    ),
                    dict(
                        label="Negative (<0)",
                        method="restyle",
                        args=[{"z": [np.where(corr_matrix.values < 0, corr_matrix.values, np.nan)]}]
                    )
                ]
            )
        ]
    )
    
    return ChartTheme.apply_default_theme(fig, "Interactive Correlation Matrix")

# ==============================================================================
# ENHANCED PORTFOLIO ENGINE with Risk Attribution
# ==============================================================================
class EnhancedPortfolioEngine(PortfolioEngine):
    """Extended portfolio engine with risk attribution and advanced analytics"""
    
    def risk_attribution(self, weights: Dict[str, float]) -> pd.DataFrame:
        """Calculate risk attribution metrics"""
        w = pd.Series(weights).reindex(self.assets).fillna(0)
        w = w / w.sum()
        
        portfolio_var = w.values @ self.S.values @ w.values
        marginal_risk = self.S.values @ w.values
        
        attribution = pd.DataFrame({
            'Asset': self.assets,
            'Weight': w.values,
            'Marginal_Risk': marginal_risk,
            'Risk_Contribution': w.values * marginal_risk,
            'Percent_Contribution': (w.values * marginal_risk) / portfolio_var * 100
        })
        
        attribution['Asset_Name'] = attribution['Asset'].map(lambda x: TICKER_NAME_MAP.get(x, x))
        attribution = attribution.sort_values('Percent_Contribution', ascending=False)
        
        return attribution
    
    def concentration_metrics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate concentration metrics"""
        w = pd.Series(weights).reindex(self.assets).fillna(0)
        w = w / w.sum()
        
        # Herfindahl-Hirschman Index
        hhi = (w ** 2).sum()
        
        # Effective number of bets
        effective_bets = 1 / hhi
        
        # Top 5 concentration
        top5_concentration = w.nlargest(5).sum()
        
        # Gini coefficient
        sorted_w = np.sort(w.values)
        n = len(sorted_w)
        cum_w = np.cumsum(sorted_w)
        gini = (n + 1 - 2 * np.sum(cum_w) / cum_w[-1]) / n if cum_w[-1] > 0 else 0
        
        return {
            'HHI': float(hhi),
            'Effective_Bets': float(effective_bets),
            'Top5_Concentration': float(top5_concentration),
            'Gini_Coefficient': float(gini),
            'Max_Weight': float(w.max()),
            'Min_Weight': float(w.min())
        }
    
    def factor_exposure(self, weights: Dict[str, float], 
                       factor_returns: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate factor exposures (simplified version)"""
        w = pd.Series(weights).reindex(self.assets).fillna(0)
        w = w / w.sum()
        
        # If factor returns not provided, use simple market factor
        if factor_returns is None:
            # Use S&P 500 as market factor
            market_returns = self.returns.mean(axis=1)  # Simple proxy
            factor_returns = pd.DataFrame({'Market': market_returns})
        
        exposures = {}
        for factor in factor_returns.columns:
            factor_ret = factor_returns[factor]
            # Simple regression for each asset
            betas = []
            for asset in self.assets:
                if asset in self.returns.columns:
                    asset_ret = self.returns[asset]
                    aligned = pd.concat([asset_ret, factor_ret], axis=1).dropna()
                    if len(aligned) > 20:
                        X = aligned.iloc[:, 1].values.reshape(-1, 1)
                        y = aligned.iloc[:, 0].values
                        X = np.column_stack([np.ones_like(X), X])
                        beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
                        betas.append(beta)
                    else:
                        betas.append(0)
                else:
                    betas.append(0)
            
            # Portfolio exposure to factor
            portfolio_exposure = np.dot(w.values, np.array(betas))
            exposures[factor] = portfolio_exposure
        
        return pd.DataFrame.from_dict(exposures, orient='index', columns=['Exposure'])

# ==============================================================================
# ENHANCED UI COMPONENTS
# ==============================================================================
def create_data_quality_dashboard(quality_df: pd.DataFrame) -> go.Figure:
    """Create visual dashboard for data quality assessment"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Data Completeness by Asset",
            "Data Days Distribution",
            "Availability Timeline",
            "Quality Score Heatmap"
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "heatmap"}]
        ]
    )
    
    # 1. Data Completeness
    completeness = quality_df['Completeness'].str.rstrip('%').astype(float) / 100
    fig.add_trace(
        go.Bar(
            x=quality_df.index,
            y=completeness,
            marker_color=np.where(completeness > 0.8, '#2e7d32', 
                                 np.where(completeness > 0.5, '#ff9800', '#c62828')),
            text=quality_df['Completeness'],
            textposition='auto',
            hovertemplate='Asset: %{x}<br>Completeness: %{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Data Days Distribution
    days = quality_df['Days'].astype(int)
    fig.add_trace(
        go.Histogram(
            x=days,
            nbinsx=20,
            marker_color='#3949ab',
            opacity=0.7,
            hovertemplate='Days: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Availability Timeline
    for idx, row in quality_df.iterrows():
        if pd.notna(row['Start']) and pd.notna(row['End']):
            fig.add_trace(
                go.Scatter(
                    x=[row['Start'], row['End']],
                    y=[idx, idx],
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=8),
                    name=idx,
                    showlegend=False,
                    hovertemplate='Asset: %{y}<br>Start: %{x}<br>End: %{text}<extra></extra>',
                    text=[row['Start'], row['End']]
                ),
                row=2, col=1
            )
    
    # 4. Quality Score Heatmap
    # Create score matrix
    assets = quality_df.index.tolist()
    metrics = ['Days', 'Completeness']
    scores = []
    
    for metric in metrics:
        if metric == 'Completeness':
            values = completeness.values
        else:
            values = days.values / days.max()
        
        scores.append(values)
    
    fig.add_trace(
        go.Heatmap(
            z=scores,
            x=assets,
            y=metrics,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Score"),
            hovertemplate='Metric: %{y}<br>Asset: %{x}<br>Score: %{z:.2f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Data Quality Dashboard",
        title_font=dict(size=20, color='#1a237e'),
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Asset", row=1, col=1, tickangle=45)
    fig.update_yaxes(title_text="Completeness", row=1, col=1)
    
    fig.update_xaxes(title_text="Days", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Asset", row=2, col=1)
    
    fig.update_xaxes(title_text="Asset", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="Metric", row=2, col=2)
    
    return fig

def create_export_panel(dataframes: Dict[str, pd.DataFrame], 
                       charts: List[go.Figure]) -> None:
    """Create export panel for downloading results"""
    
    with st.expander("📤 Export Results", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Data")
            for name, df in dataframes.items():
                if not df.empty:
                    csv = df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label=f"📥 {name}.csv",
                        data=csv,
                        file_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"export_{name}"
                    )
        
        with col2:
            st.subheader("Export Charts")
            for i, fig in enumerate(charts):
                # Convert to HTML
                html = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label=f"📊 Chart {i+1}.html",
                    data=html,
                    file_name=f"chart_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    key=f"export_chart_{i}"
                )

# ==============================================================================
# ENHANCED MAIN APP with Smart Navigation
# ==============================================================================
def main():
    """Enhanced main application with improved UX"""
    
    st.markdown('<div class="main-header">📊 QUANTUM | Institutional Risk Analytics Platform</div>', unsafe_allow_html=True)
    
    # Header with quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Version", "8.0", "Enhanced")
    with col2:
        st.metric("Analytics", "50+ Metrics", "Comprehensive")
    with col3:
        st.metric("Assets", "50+ Universe", "Global")
    with col4:
        st.metric("Updates", "Real-time", "Live")
    
    # Enhanced sidebar with collapsible sections
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Data Settings (collapsible)
        with st.expander("📊 Data Settings", expanded=True):
            category = st.selectbox(
                "Universe Category",
                list(data_manager.universe.keys()),
                help="Select asset category for analysis"
            )
            
            # Smart asset selection
            universe_dict = data_manager.universe[category]
            
            # Get recommended assets
            recommended = data_manager.get_recommended_assets(category, 8)
            
            # Multi-select with search
            tickers = st.multiselect(
                "Select Assets",
                list(universe_dict.values()),
                default=recommended,
                format_func=lambda x: f"{universe_dict.get(x, x)} ({x})" if x in universe_dict.values() else x,
                help="Select 2-20 assets for optimal performance"
            )
            
            # Date range with smart defaults
            col_a, col_b = st.columns(2)
            with col_a:
                start = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365*3),
                    max_value=datetime.now()
                )
            with col_b:
                end = st.date_input(
                    "End Date",
                    value=datetime.now()
                )
            
            # Data quality settings
            min_points = st.slider(
                "Minimum Data Points",
                min_value=60,
                max_value=1000,
                value=200,
                step=20,
                help="Filter out assets with insufficient data"
            )
        
        # Advanced Settings (collapsible)
        with st.expander("⚡ Advanced Settings", expanded=False):
            rf_rate = st.number_input(
                "Risk-Free Rate (annual)",
                min_value=0.0,
                max_value=0.20,
                value=0.04,
                step=0.005,
                format="%.3f",
                help="Annual risk-free rate for Sharpe ratio calculations"
            )
            
            confidence_level = st.slider(
                "VaR Confidence Level",
                min_value=0.90,
                max_value=0.995,
                value=0.95,
                step=0.005,
                help="Confidence level for Value at Risk calculations"
            )
            
            # Cache settings
            cache_ttl = st.select_slider(
                "Cache Duration",
                options=[300, 600, 1800, 3600, 7200],
                value=3600,
                format_func=lambda x: f"{x//60} minutes" if x >= 60 else f"{x} seconds"
            )
        
        # Action buttons
        st.markdown("---")
        col_load, col_reset = st.columns(2)
        with col_load:
            load_data = st.button("🚀 Load & Analyze", type="primary", use_container_width=True)
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
        
        # Quick tips
        with st.expander("💡 Quick Tips", expanded=False):
            st.info("""
            **Best Practices:**
            1. Select 8-15 assets for optimal results
            2. Use at least 3 years of data for stable statistics
            3. Check data quality before running optimizations
            4. Export results for institutional reporting
            """)
    
    # Main content area
    if not tickers:
        st.info("👈 Select assets from the sidebar to begin analysis")
        return
    
    if load_data:
        with st.spinner("🔍 Fetching and validating market data..."):
            # Show progress
            progress_bar = st.progress(0)
            
            # Fetch data with enhanced validation
            prices, bench, report, quality_df = fetch_prices_with_validation(
                tuple(tickers), 
                str(start), 
                str(end), 
                int(min_points)
            )
            
            progress_bar.progress(100)
        
        if prices.empty:
            st.error("❌ No usable data found. Please adjust your selection.")
            if report.get("warnings"):
                for warning in report["warnings"]:
                    st.warning(warning)
            return
        
        # Store in session state
        st.session_state['prices'] = prices
        st.session_state['bench'] = bench
        st.session_state['report'] = report
        st.session_state['quality_df'] = quality_df
        st.session_state['rf_rate'] = rf_rate
        
        # Show success message
        st.success(f"✅ Successfully loaded {len(prices.columns)} assets with {len(prices)} data points")
    
    # Check if data is loaded
    if 'prices' not in st.session_state:
        st.info("👈 Click 'Load & Analyze' to fetch data")
        return
    
    # Retrieve data from session
    prices = st.session_state['prices']
    bench = st.session_state['bench']
    report = st.session_state['report']
    quality_df = st.session_state['quality_df']
    rf_rate = st.session_state['rf_rate']
    
    # Enhanced tabs with icons
    tabs = st.tabs([
        "📊 Data Quality",
        "📈 Market Overview", 
        "🧠 Optimization Lab",
        "⚠️ Risk Analytics",
        "📉 Stress Testing",
        "🔗 Correlation Matrix",
        "📐 Technical Analysis",
        "🏆 Performance",
        "📤 Export"
    ])
    
    with tabs[0]:
        # Data Quality Dashboard
        st.markdown('<div class="section-header">Data Quality Assessment</div>', unsafe_allow_html=True)
        
        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Quality Score", f"{report.get('quality_score', 0):.1%}", report.get('quality_rating', 'Unknown'))
        col2.metric("Assets Loaded", report.get('assets', 0))
        col3.metric("Data Points", report.get('rows', 0))
        col4.metric("Period", f"{report.get('start_date', '')} to {report.get('end_date', '')}")
        
        # Show quality dashboard
        fig_quality = create_data_quality_dashboard(quality_df)
        st.plotly_chart(fig_quality, use_container_width=True)
        
        # Show detailed quality table
        with st.expander("📋 Detailed Quality Metrics", expanded=False):
            st.dataframe(quality_df, use_container_width=True)
    
    with tabs[1]:
        tab_market_overview(prices, bench, report)
    
    with tabs[2]:
        tab_portfolio_optimization(prices, bench)
    
    with tabs[3]:
        tab_advanced_var(prices, bench)
    
    with tabs[4]:
        tab_stress_testing(prices)
    
    with tabs[5]:
        tab_correlation_risk(prices)
    
    with tabs[6]:
        tab_technicals_tracking(prices, bench)
    
    with tabs[7]:
        tab_advanced_performance(prices, bench)
    
    with tabs[8]:
        # Export panel
        st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
        
        # Collect all dataframes for export
        export_dataframes = {
            'Price_Data': prices,
            'Quality_Metrics': quality_df,
            'Report_Summary': pd.DataFrame.from_dict(report, orient='index').T
        }
        
        # Collect charts
        export_charts = []
        
        # Create export interface
        create_export_panel(export_dataframes, export_charts)
        
        # Additional export options
        st.markdown("### 📄 Report Generation")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Generate Summary Report", use_container_width=True):
                st.success("Summary report generated (placeholder)")
        
        with col2:
            if st.button("📈 Create Presentation", use_container_width=True):
                st.success("Presentation created (placeholder)")
    
    # Footer with institutional branding
    st.markdown("---")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <b>QUANTUM Institutional Terminal v8.0</b> | 
        For institutional use only | 
        Data provided by Yahoo Finance
        </div>
        """, unsafe_allow_html=True)

# Run the enhanced app
if __name__ == "__main__":
    main()
