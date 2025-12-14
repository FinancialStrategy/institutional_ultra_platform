# ==============================================================================
# QUANTUM | Global Institutional Terminal - PROFESSIONAL EDITION
# ------------------------------------------------------------------------------
# Enhanced Features:
#  - Clean, minimalist institutional UI with reduced colors
#  - Advanced portfolio optimization with wheel/radial percentage charts
#  - Comprehensive VaR/CVaR/ES calculations across methodologies
#  - Relative VaR/CVaR/ES calculations vs benchmark
#  - Side-by-side comparative risk analytics
#  - Interactive strategy performance dashboards
# ==============================================================================

import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="QUANTUM | Institutional Risk Analytics",
    layout="wide",
    page_icon="📊",
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
import plotly.figure_factory as ff

import scipy.stats as stats
from scipy.optimize import minimize

# ==============================================================================
# Clean Professional Styling
# ==============================================================================
st.markdown("""
<style>
    .stApp { 
        background-color: #ffffff;
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .main-title { 
        font-size: 2.4rem; 
        font-weight: 700; 
        color: #1a1a1a; 
        text-align: center; 
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    .section-title { 
        font-size: 1.6rem; 
        font-weight: 600; 
        color: #2c3e50; 
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .subsection-title { 
        font-size: 1.2rem; 
        font-weight: 600; 
        color: #34495e; 
        margin: 1.2rem 0 0.8rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
    
    .risk-card-high {
        background: #fff5f5;
        border-left: 4px solid #e74c3c;
    }
    
    .risk-card-medium {
        background: #fff9e6;
        border-left: 4px solid #f39c12;
    }
    
    .risk-card-low {
        background: #f0f9ff;
        border-left: 4px solid #3498db;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #f8f9fa;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 10px 16px;
        font-weight: 500;
        background: white;
        color: #7f8c8d;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2c3e50;
        color: white;
        box-shadow: 0 2px 4px rgba(44, 62, 80, 0.1);
    }
    
    .stButton>button {
        background: #2c3e50;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: #34495e;
        box-shadow: 0 4px 8px rgba(44, 62, 80, 0.15);
    }
    
    .dataframe {
        font-size: 0.9rem;
    }
    
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Enhanced Data Manager
# ==============================================================================
class ProfessionalDataManager:
    def __init__(self):
        self.universe = {
            "Global Benchmarks": {
                "S&P 500": "^GSPC",
                "NASDAQ 100": "^NDX",
                "MSCI World": "URTH",
                "MSCI EM": "EEM",
            },
            "US Equities": {
                "Apple": "AAPL",
                "Microsoft": "MSFT",
                "Amazon": "AMZN",
                "Google": "GOOGL",
                "NVIDIA": "NVDA",
                "Tesla": "TSLA",
                "Meta": "META",
                "Berkshire": "BRK-B",
            },
            "Fixed Income": {
                "US 10Y Bond": "^TNX",
                "AGG": "AGG",
                "TLT": "TLT",
                "HYG": "HYG",
            },
            "Commodities": {
                "Gold": "GLD",
                "Silver": "SLV",
                "Oil": "USO",
            }
        }
    
    def get_ticker_name(self, ticker: str) -> str:
        for category in self.universe.values():
            for name, t in category.items():
                if t == ticker:
                    return name
        return ticker

data_manager = ProfessionalDataManager()

# ==============================================================================
# Advanced Portfolio Optimization Engine
# ==============================================================================
class InstitutionalPortfolioEngine:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
    def calculate_statistics(self):
        """Calculate comprehensive portfolio statistics"""
        annual_factor = 252
        stats_dict = {}
        
        for asset in self.assets:
            returns = self.returns[asset]
            stats_dict[asset] = {
                'Annual Return': returns.mean() * annual_factor,
                'Annual Volatility': returns.std() * np.sqrt(annual_factor),
                'Sharpe Ratio': (returns.mean() * annual_factor) / (returns.std() * np.sqrt(annual_factor)) if returns.std() > 0 else 0,
                'Max Drawdown': self._calculate_max_drawdown(returns),
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis(),
                'VaR_95': self._calculate_var(returns, 0.95),
                'CVaR_95': self._calculate_cvar(returns, 0.95)
            }
        
        return pd.DataFrame(stats_dict).T
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        var = self._calculate_var(returns, confidence)
        tail_losses = returns[returns <= -var]
        return -tail_losses.mean() if len(tail_losses) > 0 else 0
    
    def optimize_minimum_variance(self) -> Dict[str, float]:
        """Minimum variance portfolio"""
        cov_matrix = self.returns.cov().values * 252
        
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(self.n_assets)]
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(portfolio_variance, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {asset: float(w) for asset, w in zip(self.assets, result.x) if w > 0.001}
    
    def optimize_max_sharpe(self, risk_free_rate: float = 0.04) -> Dict[str, float]:
        """Maximum Sharpe ratio portfolio"""
        cov_matrix = self.returns.cov().values * 252
        expected_returns = self.returns.mean().values * 252
        
        def negative_sharpe(weights):
            port_return = weights.T @ expected_returns
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe = (port_return - risk_free_rate) / port_vol
            return -sharpe
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(self.n_assets)]
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(negative_sharpe, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {asset: float(w) for asset, w in zip(self.assets, result.x) if w > 0.001}
    
    def optimize_risk_parity(self) -> Dict[str, float]:
        """Risk parity portfolio"""
        cov_matrix = self.returns.cov().values * 252
        
        def risk_parity_objective(weights):
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            risk_contributions = (weights * (cov_matrix @ weights)) / port_vol
            target_rc = np.ones(self.n_assets) / self.n_assets * port_vol
            return np.sum((risk_contributions - target_rc) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0.01, 1) for _ in range(self.n_assets)]
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(risk_parity_objective, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {asset: float(w) for asset, w in zip(self.assets, result.x) if w > 0.001}
    
    def optimize_mean_variance(self, target_return: float = 0.10) -> Dict[str, float]:
        """Mean-variance optimization for target return"""
        cov_matrix = self.returns.cov().values * 252
        expected_returns = self.returns.mean().values * 252
        
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - target_return}
        ]
        bounds = [(0, 1) for _ in range(self.n_assets)]
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(portfolio_variance, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {asset: float(w) for asset, w in zip(self.assets, result.x) if w > 0.001}
    
    def calculate_portfolio_performance(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance metrics for given weights"""
        if not weights:
            return {}
        
        # Align weights with returns
        weight_series = pd.Series(weights).reindex(self.assets).fillna(0)
        portfolio_returns = (self.returns * weight_series).sum(axis=1)
        
        annual_factor = 252
        annual_return = portfolio_returns.mean() * annual_factor
        annual_vol = portfolio_returns.std() * np.sqrt(annual_factor)
        sharpe = (annual_return - 0.04) / annual_vol if annual_vol > 0 else 0
        max_dd = self._calculate_max_drawdown(portfolio_returns)
        var_95 = self._calculate_var(portfolio_returns, 0.95)
        cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        
        return {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Sortino Ratio': self._calculate_sortino(portfolio_returns),
            'Calmar Ratio': annual_return / abs(max_dd) if max_dd != 0 else 0
        }
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.04) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        return (returns.mean() * 252 - risk_free_rate) / downside_std if downside_std > 0 else 0

# ==============================================================================
# Advanced Risk Analytics Engine
# ==============================================================================
class InstitutionalRiskEngine:
    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()
        self.annual_factor = 252
        
    def calculate_absolute_risk_metrics(self, confidence_levels: List[float] = None):
        """Calculate comprehensive absolute risk metrics"""
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        metrics = {}
        
        # Basic statistics
        metrics['Annual Volatility'] = self.returns.std() * np.sqrt(self.annual_factor)
        metrics['Skewness'] = self.returns.skew()
        metrics['Kurtosis'] = self.returns.kurtosis()
        metrics['Max Drawdown'] = self._calculate_max_drawdown()
        
        # VaR/CVaR for each confidence level
        for cl in confidence_levels:
            historical_var = self._historical_var(cl)
            parametric_var = self._parametric_var(cl, 'normal')
            monte_carlo_var = self._monte_carlo_var(cl, 10000)
            
            metrics[f'Historical VaR ({int(cl*100)}%)'] = historical_var
            metrics[f'Parametric VaR ({int(cl*100)}%)'] = parametric_var
            metrics[f'Monte Carlo VaR ({int(cl*100)}%)'] = monte_carlo_var
            
            metrics[f'Historical CVaR ({int(cl*100)}%)'] = self._historical_cvar(cl)
            metrics[f'Parametric CVaR ({int(cl*100)}%)'] = self._parametric_cvar(cl)
        
        # Expected Shortfall at different levels
        for cl in [0.95, 0.99]:
            metrics[f'Expected Shortfall ({int(cl*100)}%)'] = self._expected_shortfall(cl)
        
        # Tail risk metrics
        metrics['Tail Ratio'] = self._calculate_tail_ratio()
        metrics['Gain-to-Pain Ratio'] = self._calculate_gain_to_pain()
        
        return pd.Series(metrics)
    
    def calculate_relative_risk_metrics(self, benchmark_returns: pd.Series, 
                                      confidence_levels: List[float] = None):
        """Calculate relative risk metrics vs benchmark"""
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        # Align returns
        aligned = pd.concat([self.returns, benchmark_returns], axis=1).dropna()
        portfolio_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]
        
        # Active returns
        active_returns = portfolio_ret - bench_ret
        
        metrics = {}
        
        # Relative metrics
        metrics['Tracking Error'] = active_returns.std() * np.sqrt(self.annual_factor)
        metrics['Information Ratio'] = (active_returns.mean() * self.annual_factor) / metrics['Tracking Error'] if metrics['Tracking Error'] > 0 else 0
        metrics['Beta'] = self._calculate_beta(portfolio_ret, bench_ret)
        metrics['Alpha'] = self._calculate_alpha(portfolio_ret, bench_ret)
        
        # Relative VaR/CVaR
        for cl in confidence_levels:
            rel_var = self._historical_var_relative(portfolio_ret, bench_ret, cl)
            rel_cvar = self._historical_cvar_relative(portfolio_ret, bench_ret, cl)
            
            metrics[f'Relative VaR ({int(cl*100)}%)'] = rel_var
            metrics[f'Relative CVaR ({int(cl*100)}%)'] = rel_cvar
        
        return pd.Series(metrics)
    
    def _calculate_max_drawdown(self) -> float:
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def _historical_var(self, confidence: float) -> float:
        return -np.percentile(self.returns, (1 - confidence) * 100)
    
    def _historical_cvar(self, confidence: float) -> float:
        var = self._historical_var(confidence)
        tail_losses = self.returns[self.returns <= -var]
        return -tail_losses.mean() if len(tail_losses) > 0 else 0
    
    def _historical_var_relative(self, portfolio_ret: pd.Series, bench_ret: pd.Series, confidence: float) -> float:
        active_returns = portfolio_ret - bench_ret
        return -np.percentile(active_returns, (1 - confidence) * 100)
    
    def _historical_cvar_relative(self, portfolio_ret: pd.Series, bench_ret: pd.Series, confidence: float) -> float:
        active_returns = portfolio_ret - bench_ret
        var = -np.percentile(active_returns, (1 - confidence) * 100)
        tail_losses = active_returns[active_returns <= -var]
        return -tail_losses.mean() if len(tail_losses) > 0 else 0
    
    def _parametric_var(self, confidence: float, distribution: str = 'normal') -> float:
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        if distribution == 'normal':
            z = stats.norm.ppf(confidence)
        elif distribution == 't':
            df = max(len(self.returns) - 1, 1)
            z = stats.t.ppf(confidence, df)
        else:
            z = stats.norm.ppf(confidence)
        
        return -(mu + z * sigma)
    
    def _parametric_cvar(self, confidence: float, distribution: str = 'normal') -> float:
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        if distribution == 'normal':
            z = stats.norm.ppf(confidence)
            cvar = mu - sigma * stats.norm.pdf(z) / (1 - confidence)
        else:
            # Simplified calculation for non-normal
            var = self._parametric_var(confidence, distribution)
            cvar = mu - sigma * 1.4  # Approximation
        return -cvar
    
    def _monte_carlo_var(self, confidence: float, n_simulations: int = 10000) -> float:
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        # Monte Carlo simulation
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
        return -np.percentile(simulated_returns, (1 - confidence) * 100)
    
    def _expected_shortfall(self, confidence: float) -> float:
        var = self._historical_var(confidence)
        tail_returns = self.returns[self.returns <= -var]
        return -tail_returns.mean() if len(tail_returns) > 0 else 0
    
    def _calculate_tail_ratio(self) -> float:
        """Ratio of 95th percentile to 5th percentile (positive to negative)"""
        if len(self.returns) < 100:
            return 0
        positive_tail = np.percentile(self.returns, 95)
        negative_tail = abs(np.percentile(self.returns, 5))
        return positive_tail / negative_tail if negative_tail > 0 else 0
    
    def _calculate_gain_to_pain(self) -> float:
        """Sum of positive returns divided by absolute sum of negative returns"""
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return gains / losses if losses > 0 else 0
    
    def _calculate_beta(self, portfolio_ret: pd.Series, bench_ret: pd.Series) -> float:
        covariance = np.cov(portfolio_ret, bench_ret)[0, 1]
        bench_variance = bench_ret.var()
        return covariance / bench_variance if bench_variance > 0 else 0
    
    def _calculate_alpha(self, portfolio_ret: pd.Series, bench_ret: pd.Series, 
                        risk_free_rate: float = 0.04) -> float:
        beta = self._calculate_beta(portfolio_ret, bench_ret)
        excess_port_return = portfolio_ret.mean() * self.annual_factor - risk_free_rate
        excess_bench_return = bench_ret.mean() * self.annual_factor - risk_free_rate
        return excess_port_return - beta * excess_bench_return

# ==============================================================================
# Visualization Functions
# ==============================================================================
class ProfessionalVisualizations:
    @staticmethod
    def create_portfolio_wheel_chart(weights: Dict[str, float], 
                                    asset_names: Dict[str, str]) -> go.Figure:
        """Create a professional wheel/radial chart for portfolio allocation"""
        # Prepare data
        labels = [asset_names.get(asset, asset) for asset in weights.keys()]
        values = list(weights.values())
        
        # Create radial chart
        fig = go.Figure()
        
        # Main pie chart
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=px.colors.sequential.Blues),
            textinfo='percent+label',
            textposition='outside',
            hoverinfo='label+percent+value'
        ))
        
        fig.update_layout(
            title="Portfolio Allocation",
            title_font=dict(size=16, color='#2c3e50'),
            showlegend=False,
            height=500,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
    
    @staticmethod
    def create_risk_comparison_chart(portfolio_metrics: pd.Series, 
                                    benchmark_metrics: pd.Series = None) -> go.Figure:
        """Create comparative risk metrics chart"""
        metrics_to_plot = ['Annual Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']
        
        fig = go.Figure()
        
        # Portfolio metrics
        fig.add_trace(go.Bar(
            name='Portfolio',
            x=metrics_to_plot,
            y=[abs(portfolio_metrics.get(m, 0)) for m in metrics_to_plot],
            marker_color='#3498db'
        ))
        
        # Benchmark metrics (if provided)
        if benchmark_metrics is not None:
            fig.add_trace(go.Bar(
                name='Benchmark',
                x=metrics_to_plot,
                y=[abs(benchmark_metrics.get(m, 0)) for m in metrics_to_plot],
                marker_color='#95a5a6'
            ))
        
        fig.update_layout(
            title="Risk Metrics Comparison",
            barmode='group',
            yaxis_title="Value",
            height=400,
            template='plotly_white',
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
    
    @staticmethod
    def create_var_methodology_comparison(var_results: Dict[str, float]) -> go.Figure:
        """Compare VaR across different methodologies"""
        fig = go.Figure()
        
        methods = list(var_results.keys())
        values = list(var_results.values())
        
        fig.add_trace(go.Bar(
            x=methods,
            y=values,
            marker_color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
            text=[f'{v:.3%}' for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="VaR Comparison Across Methodologies",
            yaxis_title="VaR (95%)",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_return_distribution_chart(returns: pd.Series, 
                                        var_95: float = None) -> go.Figure:
        """Create histogram with VaR overlay"""
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='#3498db',
            opacity=0.7
        ))
        
        # VaR line
        if var_95 is not None:
            fig.add_vline(
                x=-var_95 * 100,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"VaR 95%: {-var_95:.2%}",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title="Return Distribution with VaR",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=400,
            template='plotly_white'
        )
        
        return fig

# ==============================================================================
# Data Fetching
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch and align market data"""
    try:
        # Download data
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker'
        )
        
        # Extract adjusted close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = pd.DataFrame()
            for ticker in tickers:
                if ticker in data.columns.levels[0]:
                    prices[ticker] = data[(ticker, 'Adj Close')]
        else:
            prices = data['Adj Close'].rename(tickers[0])
        
        # Use S&P 500 as default benchmark
        benchmark = yf.download('^GSPC', start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Align dates
        aligned = pd.concat([prices, benchmark], axis=1).dropna()
        portfolio_prices = aligned.iloc[:, :-1]
        benchmark_prices = aligned.iloc[:, -1]
        
        return portfolio_prices, benchmark_prices
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.Series()

# ==============================================================================
# Main Application
# ==============================================================================
def main():
    st.markdown('<div class="main-title">QUANTUM | Institutional Portfolio & Risk Analytics</div>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### Configuration")
        
        # Asset Selection
        selected_category = st.selectbox(
            "Asset Category",
            list(data_manager.universe.keys())
        )
        
        available_assets = data_manager.universe[selected_category]
        selected_assets = st.multiselect(
            "Select Assets",
            list(available_assets.values()),
            default=list(available_assets.values())[:4],
            format_func=lambda x: data_manager.get_ticker_name(x)
        )
        
        # Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*3))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Risk Parameters
        st.markdown("### Risk Parameters")
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        risk_free_rate = st.number_input("Risk-Free Rate", 0.0, 0.10, 0.04, 0.005)
        
        # Optimization Parameters
        st.markdown("### Optimization")
        target_return = st.slider("Target Return (Annual)", 0.05, 0.20, 0.10, 0.01)
    
    # Fetch Data
    if not selected_assets:
        st.warning("Please select at least one asset")
        return
    
    with st.spinner("Fetching market data..."):
        prices, benchmark_prices = fetch_market_data(
            selected_assets, 
            str(start_date), 
            str(end_date)
        )
    
    if prices.empty:
        st.error("No data available for selected assets and date range")
        return
    
    # Calculate Returns
    returns = prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    
    # Initialize Engines
    portfolio_engine = InstitutionalPortfolioEngine(returns)
    risk_engine = InstitutionalRiskEngine(returns.mean(axis=1))  # Equal weight portfolio for risk analysis
    viz = ProfessionalVisualizations()
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Portfolio Optimization", "⚖️ Risk Analytics", "📈 Performance Dashboard"])
    
    # Tab 1: Portfolio Optimization
    with tab1:
        st.markdown('<div class="section-title">Portfolio Optimization Strategies</div>', unsafe_allow_html=True)
        
        # Calculate All Strategies
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="subsection-title">Optimization Strategies</div>', unsafe_allow_html=True)
            
            # Equal Weight
            equal_weights = {asset: 1/len(selected_assets) for asset in selected_assets}
            equal_perf = portfolio_engine.calculate_portfolio_performance(equal_weights)
            
            # Minimum Variance
            min_var_weights = portfolio_engine.optimize_minimum_variance()
            min_var_perf = portfolio_engine.calculate_portfolio_performance(min_var_weights)
            
            # Maximum Sharpe
            max_sharpe_weights = portfolio_engine.optimize_max_sharpe(risk_free_rate)
            max_sharpe_perf = portfolio_engine.calculate_portfolio_performance(max_sharpe_weights)
            
            # Risk Parity
            risk_parity_weights = portfolio_engine.optimize_risk_parity()
            risk_parity_perf = portfolio_engine.calculate_portfolio_performance(risk_parity_weights)
            
            # Mean-Variance
            mean_var_weights = portfolio_engine.optimize_mean_variance(target_return)
            mean_var_perf = portfolio_engine.calculate_portfolio_performance(mean_var_weights)
            
            # Create comparison table
            strategies = {
                'Equal Weight': {'weights': equal_weights, 'performance': equal_perf},
                'Minimum Variance': {'weights': min_var_weights, 'performance': min_var_perf},
                'Maximum Sharpe': {'weights': max_sharpe_weights, 'performance': max_sharpe_perf},
                'Risk Parity': {'weights': risk_parity_weights, 'performance': risk_parity_perf},
                'Mean-Variance': {'weights': mean_var_weights, 'performance': mean_var_perf}
            }
            
            # Display strategy comparison
            perf_data = []
            for strategy, data in strategies.items():
                perf = data['performance']
                perf_data.append({
                    'Strategy': strategy,
                    'Return': f"{perf.get('Annual Return', 0):.2%}",
                    'Volatility': f"{perf.get('Annual Volatility', 0):.2%}",
                    'Sharpe': f"{perf.get('Sharpe Ratio', 0):.3f}",
                    'Max DD': f"{perf.get('Max Drawdown', 0):.2%}",
                    'VaR (95%)': f"{perf.get('VaR (95%)', 0):.2%}",
                    'CVaR (95%)': f"{perf.get('CVaR (95%)', 0):.2%}"
                })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True, height=300)
        
        with col2:
            st.markdown('<div class="subsection-title">Selected Strategy Allocation</div>', unsafe_allow_html=True)
            
            # Strategy selector
            selected_strategy = st.selectbox(
                "View Strategy",
                list(strategies.keys()),
                index=2
            )
            
            # Display wheel chart for selected strategy
            weights = strategies[selected_strategy]['weights']
            asset_names = {asset: data_manager.get_ticker_name(asset) for asset in weights.keys()}
            
            fig = viz.create_portfolio_wheel_chart(weights, asset_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display weights table
            weights_df = pd.DataFrame({
                'Asset': [asset_names.get(a, a) for a in weights.keys()],
                'Weight': [f"{w:.2%}" for w in weights.values()]
            })
            st.dataframe(weights_df, use_container_width=True, height=200)
        
        # Strategy Performance Comparison Chart
        st.markdown('<div class="subsection-title">Strategy Performance Comparison</div>', unsafe_allow_html=True)
        
        fig_comparison = go.Figure()
        
        for strategy, data in strategies.items():
            perf = data['performance']
            fig_comparison.add_trace(go.Bar(
                name=strategy,
                x=['Return', 'Volatility', 'Sharpe', 'Max DD'],
                y=[
                    perf.get('Annual Return', 0),
                    perf.get('Annual Volatility', 0),
                    perf.get('Sharpe Ratio', 0),
                    abs(perf.get('Max Drawdown', 0))
                ]
            ))
        
        fig_comparison.update_layout(
            barmode='group',
            title="Strategy Performance Metrics",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Tab 2: Risk Analytics
    with tab2:
        st.markdown('<div class="section-title">Comprehensive Risk Analytics</div>', unsafe_allow_html=True)
        
        # Calculate absolute risk metrics
        absolute_metrics = risk_engine.calculate_absolute_risk_metrics(
            confidence_levels=[0.90, 0.95, 0.99]
        )
        
        # Calculate relative risk metrics
        relative_metrics = risk_engine.calculate_relative_risk_metrics(
            benchmark_returns,
            confidence_levels=[0.90, 0.95, 0.99]
        )
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-title">Absolute Risk Metrics</div>', unsafe_allow_html=True)
            
            # Key metrics cards
            metrics_to_display = [
                ('Annual Volatility', absolute_metrics.get('Annual Volatility', 0)),
                ('Max Drawdown', absolute_metrics.get('Max Drawdown', 0)),
                ('VaR (95%)', absolute_metrics.get('Historical VaR (95%)', 0)),
                ('CVaR (95%)', absolute_metrics.get('Historical CVaR (95%)', 0)),
                ('Skewness', absolute_metrics.get('Skewness', 0)),
                ('Kurtosis', absolute_metrics.get('Kurtosis', 0))
            ]
            
            for metric_name, value in metrics_to_display:
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #7f8c8d;">{metric_name}</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">
                            {abs(value):.2% if 'VaR' not in metric_name and 'CVaR' not in metric_name else -value:.2%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="subsection-title">Relative Risk Metrics (vs S&P 500)</div>', unsafe_allow_html=True)
            
            rel_metrics_to_display = [
                ('Tracking Error', relative_metrics.get('Tracking Error', 0)),
                ('Information Ratio', relative_metrics.get('Information Ratio', 0)),
                ('Beta', relative_metrics.get('Beta', 0)),
                ('Alpha', relative_metrics.get('Alpha', 0)),
                ('Relative VaR (95%)', relative_metrics.get('Relative VaR (95%)', 0)),
                ('Relative CVaR (95%)', relative_metrics.get('Relative CVaR (95%)', 0))
            ]
            
            for metric_name, value in rel_metrics_to_display:
                risk_class = "risk-card-high" if metric_name in ['Tracking Error', 'Relative VaR (95%)', 'Relative CVaR (95%)'] and abs(value) > 0.05 else \
                            "risk-card-medium" if abs(value) > 0.02 else "risk-card-low"
                
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <div style="font-size: 0.9rem; color: #7f8c8d;">{metric_name}</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">
                            {value:.2% if metric_name != 'Beta' and metric_name != 'Information Ratio' else value:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # VaR Methodology Comparison
        st.markdown('<div class="subsection-title">VaR Methodologies Comparison</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate VaR using different methods
            var_methods = {
                'Historical VaR': risk_engine._historical_var(confidence_level),
                'Parametric VaR (Normal)': risk_engine._parametric_var(confidence_level, 'normal'),
                'Monte Carlo VaR': risk_engine._monte_carlo_var(confidence_level, 5000),
                'Parametric VaR (t-dist)': risk_engine._parametric_var(confidence_level, 't')
            }
            
            fig_var = viz.create_var_methodology_comparison(var_methods)
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            # Return distribution with VaR
            var_95 = risk_engine._historical_var(0.95)
            fig_dist = viz.create_return_distribution_chart(risk_engine.returns, var_95)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Risk Metrics Comparison
        st.markdown('<div class="subsection-title">Absolute vs Relative Risk Comparison</div>', unsafe_allow_html=True)
        
        # Create comparison DataFrame
        comparison_data = {
            'Metric': [
                'VaR (95%)', 'CVaR (95%)', 
                'Relative VaR (95%)', 'Relative CVaR (95%)'
            ],
            'Value': [
                -absolute_metrics.get('Historical VaR (95%)', 0),
                -absolute_metrics.get('Historical CVaR (95%)', 0),
                -relative_metrics.get('Relative VaR (95%)', 0),
                -relative_metrics.get('Relative CVaR (95%)', 0)
            ],
            'Type': ['Absolute', 'Absolute', 'Relative', 'Relative']
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        fig_comparison = px.bar(
            df_comparison,
            x='Metric',
            y='Value',
            color='Type',
            barmode='group',
            color_discrete_map={'Absolute': '#3498db', 'Relative': '#2ecc71'},
            title="Absolute vs Relative Risk Measures"
        )
        
        fig_comparison.update_layout(
            height=400,
            template='plotly_white',
            yaxis_tickformat='.2%'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed Risk Metrics Table
        st.markdown('<div class="subsection-title">Detailed Risk Metrics</div>', unsafe_allow_html=True)
        
        detailed_metrics = pd.DataFrame({
            'Absolute Metrics': absolute_metrics,
            'Relative Metrics': relative_metrics.reindex(absolute_metrics.index)
        }).dropna(how='all')
        
        st.dataframe(detailed_metrics.style.format({
            'Absolute Metrics': '{:.4%}',
            'Relative Metrics': '{:.4%}'
        }), use_container_width=True, height=400)
    
    # Tab 3: Performance Dashboard
    with tab3:
        st.markdown('<div class="section-title">Performance Analytics Dashboard</div>', unsafe_allow_html=True)
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        cum_benchmark = (1 + benchmark_returns).cumprod()
        
        # Create performance chart
        fig_performance = go.Figure()
        
        # Portfolio (equal weight for illustration)
        portfolio_cum = cum_returns.mean(axis=1)
        fig_performance.add_trace(go.Scatter(
            x=portfolio_cum.index,
            y=portfolio_cum,
            mode='lines',
            name='Portfolio',
            line=dict(color='#3498db', width=2)
        ))
        
        # Benchmark
        fig_performance.add_trace(go.Scatter(
            x=cum_benchmark.index,
            y=cum_benchmark,
            mode='lines',
            name='S&P 500',
            line=dict(color='#95a5a6', width=2)
        ))
        
        fig_performance.update_layout(
            title="Cumulative Performance",
            yaxis_title="Growth of $1",
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Rolling metrics
        st.markdown('<div class="subsection-title">Rolling Risk Metrics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            window = st.slider("Rolling Window (days)", 20, 252, 60)
            
            # Calculate rolling volatility
            rolling_vol = portfolio_cum.pct_change().rolling(window).std() * np.sqrt(252)
            
            fig_rolling_vol = go.Figure()
            fig_rolling_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name=f'{window}-day Rolling Vol',
                line=dict(color='#e74c3c', width=1.5)
            ))
            
            fig_rolling_vol.update_layout(
                title=f"Rolling Annualized Volatility ({window}-day)",
                yaxis_title="Volatility",
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_rolling_vol, use_container_width=True)
        
        with col2:
            # Calculate rolling Sharpe
            rolling_sharpe = portfolio_cum.pct_change().rolling(window).mean() * 252 / \
                           (portfolio_cum.pct_change().rolling(window).std() * np.sqrt(252))
            
            fig_rolling_sharpe = go.Figure()
            fig_rolling_sharpe.add_trace(go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name=f'{window}-day Rolling Sharpe',
                line=dict(color='#2ecc71', width=1.5)
            ))
            
            fig_rolling_sharpe.update_layout(
                title=f"Rolling Sharpe Ratio ({window}-day)",
                yaxis_title="Sharpe Ratio",
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_rolling_sharpe, use_container_width=True)
        
        # Drawdown analysis
        st.markdown('<div class="subsection-title">Drawdown Analysis</div>', unsafe_allow_html=True)
        
        # Calculate drawdown
        peak = portfolio_cum.expanding(min_periods=1).max()
        drawdown = (portfolio_cum - peak) / peak
        
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#e74c3c', width=1),
            fillcolor='rgba(231, 76, 60, 0.3)'
        ))
        
        fig_drawdown.update_layout(
            title="Portfolio Drawdown",
            yaxis_title="Drawdown (%)",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_drawdown, use_container_width=True)
        
        # Performance statistics table
        st.markdown('<div class="subsection-title">Performance Statistics</div>', unsafe_allow_html=True)
        
        # Calculate statistics for each asset
        stats_df = portfolio_engine.calculate_statistics()
        
        # Format for display
        display_stats = stats_df.copy()
        display_stats = display_stats.applymap(lambda x: f"{x:.2%}" if isinstance(x, float) and abs(x) < 1 else f"{x:.2f}")
        
        st.dataframe(display_stats, use_container_width=True, height=400)

if __name__ == "__main__":
    main()
