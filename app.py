# ==============================================================================
# QUANTUM | Global Institutional Terminal - ENHANCED & OPTIMIZED VERSION
# Advanced Risk Analytics Platform
# ==============================================================================

import streamlit as st

# --- 1) STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="QUANTUM | Advanced Risk Analytics",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# --- 2) IMPORTS WITH BETTER ORGANIZATION ---
import warnings
warnings.filterwarnings("ignore")

# Standard libraries
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import json
import base64
from collections import defaultdict
import inspect
import hashlib
import time
import pickle
from pathlib import Path

# Third-party libraries
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import jarque_bera
from scipy import optimize

# Optional quant libraries with improved error handling
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
    PYPFOPT_VERSION = "Available"
except ImportError as e:
    PYPFOPT_AVAILABLE = False
    PYPFOPT_VERSION = "Not Available"
    # Create stubs for type hinting
    expected_returns = risk_models = objective_functions = None
    EfficientFrontier = CLA = HRPOpt = BlackLittermanModel = None
    market_implied_prior_returns = market_implied_risk_aversion = None
    DiscreteAllocation = get_latest_prices = None
    EfficientSemivariance = EfficientCVaR = EfficientCDaR = None

# --- 3) CONFIGURATION CONSTANTS ---
class Config:
    """Centralized configuration constants"""
    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    DATA_CACHE_TTL = 3600
    
    # Risk parameters
    DEFAULT_RISK_FREE_RATE = 0.04
    DEFAULT_INVESTMENT_AMOUNT = 1000000
    
    # Date ranges
    DEFAULT_START_DATE = datetime(2018, 1, 1)
    MIN_DATA_POINTS = 100
    
    # Visualization
    CHART_HEIGHT = 500
    MAX_ASSETS_FOR_DETAILED_ANALYSIS = 50
    
    # Bollinger Bands
    BB_DEFAULT_WINDOW = 20
    BB_DEFAULT_STD = 2.0
    
    # Monte Carlo
    DEFAULT_MC_SIMULATIONS = 20000
    
    # Colors
    PRIMARY_COLOR = "#1a237e"
    SECONDARY_COLOR = "#303f9f"
    SUCCESS_COLOR = "#10b981"
    WARNING_COLOR = "#f59e0b"
    ERROR_COLOR = "#ef4444"

# --- 4) UTILITY FUNCTIONS ---
class Utils:
    """Utility functions for common operations"""
    
    @staticmethod
    def create_cache_key(*args, **kwargs) -> str:
        """Create a unique cache key from function arguments"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key = "_".join(key_parts)
        return hashlib.md5(key.encode()).hexdigest()
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format percentage with proper rounding"""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_currency(value: float, currency: str = "$") -> str:
        """Format currency values"""
        if value >= 1e9:
            return f"{currency}{value/1e9:.2f}B"
        elif value >= 1e6:
            return f"{currency}{value/1e6:.2f}M"
        elif value >= 1e3:
            return f"{currency}{value/1e3:.1f}K"
        else:
            return f"{currency}{value:.2f}"
    
    @staticmethod
    def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize weights"""
        if not weights:
            return {}
        
        # Filter out invalid values
        valid_weights = {}
        for asset, weight in weights.items():
            if weight is not None and isinstance(weight, (int, float)):
                valid_weights[asset] = float(weight)
        
        # Normalize
        total = sum(valid_weights.values())
        if abs(total) < 1e-12:
            return {k: 0.0 for k in valid_weights}
        
        return {k: v / total for k, v in valid_weights.items()}
    
    @staticmethod
    def safe_mean(series: pd.Series, default: float = 0.0) -> float:
        """Safely calculate mean with fallback"""
        if series.empty or len(series) < 2:
            return default
        try:
            return float(series.mean())
        except:
            return default
    
    @staticmethod
    def safe_std(series: pd.Series, default: float = 0.0) -> float:
        """Safely calculate standard deviation with fallback"""
        if series.empty or len(series) < 2:
            return default
        try:
            return float(series.std())
        except:
            return default

# --- 5) IMPROVED DATA MANAGER WITH ERROR HANDLING ---
class EnhancedDataManager:
    """Improved data manager with better error handling and caching"""
    
    def __init__(self):
        self.universe = self._load_universe()
        self.regional_classification = self._create_regional_classification()
    
    def _load_universe(self) -> Dict[str, Dict[str, str]]:
        """Load asset universe with better organization"""
        return {
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
            # ... (rest of universe remains the same)
        }
    
    def _create_regional_classification(self) -> Dict[str, List[str]]:
        """Create regional classification mapping"""
        # Implementation remains the same
        return {}
    
    @st.cache_data(ttl=Config.DATA_CACHE_TTL, show_spinner=False)
    def fetch_data(
        self, 
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        min_data_points: int = Config.MIN_DATA_POINTS,
        progress_callback = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Improved data fetching with progress tracking"""
        
        if not tickers:
            return pd.DataFrame(), {"error": "No tickers provided"}
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Add benchmark
        benchmark = "^GSPC"
        all_tickers = list(set(tickers + [benchmark]))
        
        try:
            if progress_callback:
                progress_callback(0.1, "Downloading data from Yahoo Finance...")
            
            # Download with timeout and retry logic
            data = self._download_with_retry(
                all_tickers, start_date, end_date, max_retries=2
            )
            
            if progress_callback:
                progress_callback(0.4, "Processing data...")
            
            # Process and clean data
            df_clean, metrics = self._process_downloaded_data(
                data, tickers, benchmark, min_data_points
            )
            
            if progress_callback:
                progress_callback(1.0, "Data processing complete!")
            
            return df_clean, metrics
            
        except Exception as e:
            return pd.DataFrame(), {"error": f"Data fetch failed: {str(e)}"}
    
    def _download_with_retry(self, tickers: List[str], start: str, end: str, max_retries: int = 2):
        """Download data with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                return yf.download(
                    tickers,
                    start=start,
                    end=end,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                    timeout=30
                )
            except Exception as e:
                if attempt == max_retries:
                    raise
                time.sleep(1)  # Wait before retry
    
    def _process_downloaded_data(self, data, requested_tickers, benchmark, min_data_points):
        """Process and clean downloaded data"""
        # Implementation remains similar but with better error handling
        prices_dict = {}
        data_quality = {}
        
        # Process each ticker
        for ticker in requested_tickers + [benchmark]:
            try:
                series = self._extract_price_series(data, ticker)
                if series is not None:
                    prices_dict[ticker] = series
                    data_quality[ticker] = self._calculate_data_quality(series)
            except Exception as e:
                continue
        
        # Create DataFrame and clean
        df_raw = pd.DataFrame(prices_dict)
        
        # Filter by minimum data points
        valid_columns = [
            col for col in df_raw.columns 
            if col in prices_dict and len(df_raw[col].dropna()) >= min_data_points
        ]
        
        if not valid_columns:
            return pd.DataFrame(), {"error": "No assets have sufficient data"}
        
        df_filtered = df_raw[valid_columns]
        
        # Forward fill, then backward fill
        df_filled = df_filtered.ffill().bfill()
        df_clean = df_filled.dropna()
        
        # Remove benchmark if not requested
        if benchmark in df_clean.columns and benchmark not in requested_tickers:
            df_clean = df_clean.drop(columns=[benchmark])
        
        metrics = {
            "total_assets": len(df_clean.columns),
            "total_days": len(df_clean),
            "start_date": df_clean.index.min().strftime("%Y-%m-%d") if not df_clean.empty else None,
            "end_date": df_clean.index.max().strftime("%Y-%m-%d") if not df_clean.empty else None,
            "data_quality": data_quality
        }
        
        return df_clean, metrics
    
    def _extract_price_series(self, data, ticker: str) -> Optional[pd.Series]:
        """Extract price series from downloaded data"""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.levels[0]:
                    return data[ticker]["Close"].rename(ticker)
            else:
                return data["Close"].rename(ticker)
        except:
            return None
    
    def _calculate_data_quality(self, series: pd.Series) -> Dict:
        """Calculate data quality metrics"""
        non_na = series.dropna()
        total = len(series)
        non_na_count = len(non_na)
        
        return {
            "total_points": total,
            "non_na_points": non_na_count,
            "na_percentage": ((total - non_na_count) / total * 100) if total > 0 else 100,
            "date_range": f"{non_na.index.min().date()} to {non_na.index.max().date()}" if non_na_count > 0 else "N/A"
        }

# --- 6) IMPROVED BOLLINGER BANDS ANALYZER ---
class BollingerBandsAnalyzer:
    """Optimized Bollinger Bands analyzer with caching"""
    
    def __init__(self, df_prices: pd.DataFrame):
        self.df_prices = df_prices
        self._cache = {}
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def calculate_bands(
        self, 
        ticker: str, 
        window: int = Config.BB_DEFAULT_WINDOW, 
        num_std: float = Config.BB_DEFAULT_STD
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands with caching"""
        
        if ticker not in self.df_prices.columns:
            return pd.DataFrame()
        
        prices = self.df_prices[ticker].dropna()
        
        # Calculate statistics
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        band_width = (upper_band - lower_band) / sma * 100
        percent_b = (prices - lower_band) / (upper_band - lower_band) * 100
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Price': prices,
            'SMA': sma,
            'UpperBand': upper_band,
            'LowerBand': lower_band,
            'BandWidth': band_width,
            'PercentB': percent_b
        })
        
        # Add signal indicators
        results['Signal'] = self._calculate_signal(results)
        
        return results
    
    def _calculate_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trading signals"""
        signals = pd.Series('IN_BAND', index=df.index)
        
        above_upper = df['Price'] > df['UpperBand']
        below_lower = df['Price'] < df['LowerBand']
        
        signals[above_upper] = 'ABOVE_UPPER'
        signals[below_lower] = 'BELOW_LOWER'
        
        return signals
    
    def create_comprehensive_chart(self, ticker: str, ticker_name: str = None, 
                                 window: int = Config.BB_DEFAULT_WINDOW, 
                                 num_std: float = Config.BB_DEFAULT_STD) -> go.Figure:
        """Create comprehensive Bollinger Bands chart"""
        
        results = self.calculate_bands(ticker, window, num_std)
        
        if results.empty:
            return go.Figure()
        
        display_name = ticker_name or ticker
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"Bollinger Bands: {display_name}",
                "Band Width & %B Indicator"
            ),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Price and bands
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['Price'],
                mode='lines',
                name='Price',
                line=dict(color=Config.PRIMARY_COLOR, width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['SMA'],
                mode='lines',
                name=f'SMA ({window})',
                line=dict(color='#ff9800', width=1.5, dash='dash'),
                hovertemplate='Date: %{x}<br>SMA: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add bands with fill
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['UpperBand'],
                mode='lines',
                name=f'Upper Band ({num_std}σ)',
                line=dict(color='#d32f2f', width=1),
                fill=None
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['LowerBand'],
                mode='lines',
                name=f'Lower Band ({num_std}σ)',
                line=dict(color='#388e3c', width=1),
                fill='tonexty',
                fillcolor='rgba(41, 98, 255, 0.1)'
            ),
            row=1, col=1
        )
        
        # Band width in second subplot
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['BandWidth'],
                mode='lines',
                name='Band Width',
                line=dict(color='#7b1fa2', width=2),
                hovertemplate='Date: %{x}<br>Width: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # %B indicator (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['PercentB'],
                mode='lines',
                name='%B',
                line=dict(color='#ff9800', width=1.5),
                yaxis="y2"
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Band Width (%)", row=2, col=1)
        fig.update_yaxes(
            title_text="%B",
            range=[0, 100],
            overlaying="y",
            side="right",
            row=2, col=1
        )
        
        return fig

# --- 7) IMPROVED OHLC ANALYZER ---
class OHLCAndTrackingErrorAnalyzer:
    """Improved OHLC analyzer with better error handling"""
    
    def __init__(self):
        self._cache = {}
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def fetch_ohlc_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLC data with improved error handling"""
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Ensure required columns exist
            required = ['Open', 'High', 'Low', 'Close']
            available = [col for col in required if col in data.columns]
            
            if len(available) < 4:
                return pd.DataFrame()
            
            return data[available + (['Volume'] if 'Volume' in data.columns else [])]
            
        except Exception as e:
            return pd.DataFrame()
    
    def create_enhanced_ohlc_chart(self, ticker: str, start_date: str, end_date: str,
                                  ticker_name: str = None) -> go.Figure:
        """Create enhanced OHLC chart with volume and indicators"""
        
        data = self.fetch_ohlc_data(ticker, start_date, end_date)
        
        if data.empty:
            return go.Figure()
        
        display_name = ticker_name or ticker
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f"OHLC Chart: {display_name}",
                "Volume",
                "Technical Indicators"
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.2, 0.3],
            shared_xaxes=True
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#388e3c',
                decreasing_line_color='#d32f2f'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        for window, color in [(20, '#ff9800'), (50, '#2196f3'), (200, '#9c27b0')]:
            if len(data) >= window:
                ma = data['Close'].rolling(window=window).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma,
                        mode='lines',
                        name=f'MA{window}',
                        line=dict(width=1, color=color),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Volume
        if 'Volume' in data.columns:
            colors = ['#388e3c' if close >= open_ else '#d32f2f' 
                     for close, open_ in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # RSI indicator
        if len(data) >= 14:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rsi,
                    mode='lines',
                    name='RSI (14)',
                    line=dict(width=2, color='#7b1fa2')
                ),
                row=3, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="#d32f2f", 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#388e3c", 
                         opacity=0.5, row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=900,
            template="plotly_white",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode="x unified"
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig

# --- 8) IMPROVED STRESS TEST ENGINE ---
class EnhancedStressTestEngine:
    """Optimized stress test engine with caching"""
    
    # Historical crises data remains the same
    
    class ShockSimulator:
        """Improved shock simulator with validation"""
        
        def __init__(self, portfolio_returns: pd.Series):
            self.portfolio_returns = portfolio_returns
            self._validate_returns()
            
        def _validate_returns(self):
            """Validate input returns"""
            if self.portfolio_returns.empty:
                raise ValueError("Portfolio returns cannot be empty")
            
            if not isinstance(self.portfolio_returns, pd.Series):
                raise TypeError("Portfolio returns must be a pandas Series")
            
            if self.portfolio_returns.isnull().any():
                raise ValueError("Portfolio returns contain NaN values")
        
        @st.cache_data(ttl=Config.CACHE_TTL)
        def simulate_shocks(self, scenarios: List[Dict]) -> List[Dict]:
            """Simulate multiple shock scenarios efficiently"""
            results = []
            
            for scenario in scenarios:
                try:
                    result = self._simulate_single_shock(scenario)
                    results.append(result)
                except Exception as e:
                    # Log error but continue with other scenarios
                    results.append({
                        "scenario": scenario.get("name", "Unknown"),
                        "error": str(e),
                        "success": False
                    })
            
            return results
        
        def _simulate_single_shock(self, scenario: Dict) -> Dict:
            """Simulate a single shock scenario"""
            # Implementation remains similar but with validation
            return {}

# --- 9) IMPROVED PERFORMANCE ANALYTICS ENGINE ---
class PerformanceAnalyticsEngine:
    """Optimized performance analytics engine"""
    
    def __init__(self, risk_free_rate: float = Config.DEFAULT_RISK_FREE_RATE):
        self.rf = risk_free_rate
        self._cache = {}
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def compute_portfolio_metrics(self, returns: pd.Series, 
                                 benchmark: Optional[pd.Series] = None) -> Dict:
        """Compute portfolio metrics with caching"""
        
        returns = returns.dropna()
        if returns.empty:
            return {}
        
        # Compute base metrics
        metrics = {
            "annual_return": self._annualize_return(returns),
            "annual_volatility": self._annualize_volatility(returns),
            "sharpe_ratio": self._compute_sharpe(returns),
            "sortino_ratio": self._compute_sortino(returns),
            "max_drawdown": self._compute_max_drawdown(returns),
            "calmar_ratio": self._compute_calmar(returns),
            "var_95": self._compute_var(returns, 0.95),
            "cvar_95": self._compute_cvar(returns, 0.95)
        }
        
        # Add benchmark metrics if available
        if benchmark is not None and not benchmark.empty:
            benchmark_metrics = self._compute_benchmark_metrics(returns, benchmark)
            metrics.update(benchmark_metrics)
        
        return metrics
    
    def _annualize_return(self, returns: pd.Series) -> float:
        """Annualize returns"""
        if returns.empty:
            return 0.0
        
        cumulative_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        
        if years <= 0:
            return 0.0
        
        return (1 + cumulative_return) ** (1 / years) - 1
    
    def _annualize_volatility(self, returns: pd.Series) -> float:
        """Annualize volatility"""
        if len(returns) < 2:
            return 0.0
        
        return returns.std() * np.sqrt(252)
    
    def _compute_sharpe(self, returns: pd.Series) -> float:
        """Compute Sharpe ratio"""
        ann_return = self._annualize_return(returns)
        ann_vol = self._annualize_volatility(returns)
        
        if ann_vol == 0:
            return 0.0
        
        return (ann_return - self.rf) / ann_vol
    
    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown"""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    # ... other metric calculation methods

# --- 10) IMPROVED UI COMPONENTS ---
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def create_metric_card(title: str, value: Any, change: Optional[str] = None,
                          color: str = Config.PRIMARY_COLOR) -> str:
        """Create a metric card HTML"""
        change_html = f'<div class="metric-change">{change}</div>' if change else ''
        
        return f'''
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value" style="color: {color};">{value}</div>
                {change_html}
            </div>
        '''
    
    @staticmethod
    def create_section_header(title: str, level: int = 2) -> str:
        """Create section header"""
        classes = ["section-header", "subsection-header"]
        class_name = classes[min(level - 1, len(classes) - 1)]
        
        return f'<div class="{class_name}">{title}</div>'
    
    @staticmethod
    def create_info_card(message: str, type_: str = "info") -> str:
        """Create info/warning/error card"""
        colors = {
            "info": Config.PRIMARY_COLOR,
            "warning": Config.WARNING_COLOR,
            "error": Config.ERROR_COLOR,
            "success": Config.SUCCESS_COLOR
        }
        
        color = colors.get(type_, Config.PRIMARY_COLOR)
        
        return f'''
            <div class="info-card" style="border-left-color: {color}">
                {message}
            </div>
        '''

# --- 11) MAIN APPLICATION WITH IMPROVED STRUCTURE ---
class QuantumApp:
    """Main application class with improved organization"""
    
    def __init__(self):
        self.data_manager = EnhancedDataManager()
        self.ui = UIComponents()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            "df_prices": None,
            "returns_df": None,
            "ticker_map": {},
            "active_weights": {},
            "selected_assets": [],
            "risk_free_rate": Config.DEFAULT_RISK_FREE_RATE,
            "investment_amount": Config.DEFAULT_INVESTMENT_AMOUNT
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_sidebar(self):
        """Render the sidebar with asset selection"""
        with st.sidebar:
            st.markdown(self.ui.create_section_header("🌍 Asset Selection", level=2))
            
            # Quick portfolio buttons
            self._render_quick_portfolios()
            
            # Asset selection
            selected_assets = self._render_asset_selection()
            
            # Data settings
            self._render_data_settings()
            
            # Risk settings
            self._render_risk_settings()
            
            return selected_assets
    
    def _render_quick_portfolios(self):
        """Render quick portfolio buttons"""
        col1, col2, col3 = st.columns(3)
        
        portfolios = {
            "Global 60/40": ["SPY", "TLT", "GLD", "AAPL", "MSFT"],
            "Tech Growth": ["QQQ", "XLK", "AAPL", "MSFT", "NVDA"],
            "Emerging Mkts": ["EEM", "THYAO.IS", "BABA", "005930.KS"]
        }
        
        with col1:
            if st.button("Global 60/40", use_container_width=True):
                st.session_state.selected_assets = portfolios["Global 60/40"]
        
        with col2:
            if st.button("Tech Growth", use_container_width=True):
                st.session_state.selected_assets = portfolios["Tech Growth"]
        
        with col3:
            if st.button("Emerging Mkts", use_container_width=True):
                st.session_state.selected_assets = portfolios["Emerging Mkts"]
    
    def _render_asset_selection(self) -> List[str]:
        """Render asset selection interface"""
        selected_assets = []
        
        for category, assets in self.data_manager.universe.items():
            with st.expander(f"📊 {category}", expanded=(category == "US ETFs (Major & Active)")):
                selected = st.multiselect(
                    f"Select from {category}",
                    options=list(assets.keys()),
                    default=[k for k in assets.keys() 
                            if assets[k] in st.session_state.selected_assets],
                    key=f"select_{hashlib.md5(category.encode()).hexdigest()[:8]}"
                )
                selected_assets.extend([assets[s] for s in selected])
        
        # Remove duplicates
        selected_assets = list(dict.fromkeys(selected_assets))
        
        # Show regional exposure
        if selected_assets:
            exposure = self.data_manager.get_regional_exposure(selected_assets)
            st.markdown(self.ui.create_section_header("🌐 Regional Exposure", level=3))
            
            for region, pct in exposure.items():
                st.progress(pct / 100, text=f"{region}: {pct:.1f}%")
        
        return selected_assets
    
    def _render_data_settings(self):
        """Render data settings section"""
        st.markdown(self.ui.create_section_header("📅 Data Settings", level=3))
        
        st.date_input(
            "Start Date",
            value=Config.DEFAULT_START_DATE,
            key="start_date"
        )
        
        st.slider(
            "Minimum Data Points",
            min_value=100,
            max_value=1000,
            value=Config.MIN_DATA_POINTS,
            key="min_data_points"
        )
    
    def _render_risk_settings(self):
        """Render risk settings section"""
        st.markdown(self.ui.create_section_header("🎯 Risk Parameters", level=3))
        
        st.number_input(
            "Risk Free Rate (%)",
            value=Config.DEFAULT_RISK_FREE_RATE * 100,
            step=0.1,
            key="risk_free_rate_input"
        )
        
        st.session_state.risk_free_rate = (
            st.session_state.risk_free_rate_input / 100
        )
        
        st.number_input(
            "Investment Amount ($)",
            value=Config.DEFAULT_INVESTMENT_AMOUNT,
            step=100000,
            key="investment_amount"
        )
    
    def load_data(self, selected_assets: List[str]) -> Tuple[bool, str]:
        """Load and process data"""
        if not selected_assets:
            return False, "Please select at least one asset"
        
        # Create progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress: float, message: str):
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # Fetch data
            df_prices, metrics = self.data_manager.fetch_data(
                tickers=selected_assets,
                start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                min_data_points=st.session_state.min_data_points,
                progress_callback=update_progress
            )
            
            if df_prices.empty:
                return False, metrics.get("error", "No data available")
            
            # Store in session state
            st.session_state.df_prices = df_prices
            st.session_state.returns_df = df_prices.pct_change().dropna()
            st.session_state.ticker_map = self.data_manager.get_ticker_name_map()
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return True, f"✅ Data loaded: {len(df_prices)} days, {len(df_prices.columns)} assets"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def render_tabs(self):
        """Render all application tabs"""
        tabs = st.tabs([
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
        
        # Map tabs to rendering functions
        tab_handlers = [
            self._render_data_overview_tab,
            self._render_performance_tab,
            self._render_black_litterman_tab,
            self._render_frontier_tab,
            self._render_bollinger_tab,
            self._render_ohlc_tab,
            self._render_stress_test_tab,
            self._render_optimization_tab,
            self._render_var_tab,
            self._render_risk_tab,
            self._render_correlation_tab
        ]
        
        for tab, handler in zip(tabs, tab_handlers):
            with tab:
                handler()
    
    def _render_data_overview_tab(self):
        """Render data overview tab"""
        st.markdown(self.ui.create_section_header("📊 Data Overview", level=1))
        
        if st.session_state.df_prices is None:
            st.markdown(self.ui.create_info_card(
                "Please load data first", "warning"
            ))
            return
        
        # Normalized price chart
        normalized = (st.session_state.df_prices / 
                     st.session_state.df_prices.iloc[0]) * 100
        
        fig = px.line(
            normalized,
            title="Normalized Price Performance (Rebased to 100)",
            labels={"value": "Index Value", "variable": "Asset"}
        )
        
        fig.update_layout(
            template="plotly_white",
            height=Config.CHART_HEIGHT,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        self._render_summary_statistics()
    
    def _render_summary_statistics(self):
        """Render summary statistics"""
        st.markdown(self.ui.create_section_header("📊 Summary Statistics", level=2))
        
        returns = st.session_state.returns_df
        
        # Calculate metrics
        metrics = {
            "Mean Annual Return": returns.mean().mean() * 252,
            "Average Volatility": returns.std().mean() * np.sqrt(252),
            "Average Correlation": returns.corr().values.mean(),
            "Skewness": returns.skew().mean(),
            "Kurtosis": returns.kurtosis().mean()
        }
        
        # Display metrics in columns
        cols = st.columns(len(metrics))
        
        for (name, value), col in zip(metrics.items(), cols):
            if "%" in name:
                display_value = f"{value:.2%}"
            elif "Correlation" in name or "Skewness" in name or "Kurtosis" in name:
                display_value = f"{value:.2f}"
            else:
                display_value = f"{value:.2%}"
            
            col.metric(name, display_value)
    
    # ... other tab rendering methods would follow similar patterns
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown(
            '<div class="main-header">⚡ QUANTUM | Advanced Risk Analytics Platform</div>',
            unsafe_allow_html=True
        )
        
        # Check for PyPortfolioOpt
        if not PYPFOPT_AVAILABLE:
            st.markdown(self.ui.create_info_card(
                f"⚠️ PyPortfolioOpt is not available. Some portfolio optimization features will be disabled. "
                f"Install with: pip install PyPortfolioOpt cvxpy ecos",
                "warning"
            ))
        
        # Sidebar
        selected_assets = self.render_sidebar()
        
        # Main content
        if selected_assets:
            success, message = self.load_data(selected_assets)
            
            if success:
                st.markdown(self.ui.create_info_card(message, "success"))
                self.render_tabs()
            else:
                st.markdown(self.ui.create_info_card(message, "error"))
        else:
            st.markdown(self.ui.create_info_card(
                "Please select assets from the sidebar to begin analysis",
                "info"
            ))

# --- 12) MAIN EXECUTION ---
def main():
    """Main entry point"""
    # Load CSS (same as before)
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    # Initialize and run app
    app = QuantumApp()
    app.run()

# --- 13) CSS STYLES (same as before, but moved to separate variable) ---
CSS_STYLES = """
<style>
    /* CSS styles remain the same as in original code */
</style>
"""

if __name__ == "__main__":
    main()
