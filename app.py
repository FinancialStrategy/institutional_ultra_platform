# ==============================================================================
# QUANTUM | Global Institutional Terminal - ENHANCED VERSION
# Advanced Performance Metrics + Black-Litterman + Advanced Efficient Frontier
# ==============================================================================

import streamlit as st

# --- 1) STREAMLIT PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="QUANTUM | Advanced Portfolio Analytics",
    layout="wide",
    page_icon="📈",
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
from scipy import optimize
import seaborn as sns
import matplotlib.pyplot as plt


# ==============================================================================
# 3) ADVANCED PERFORMANCE METRICS ENGINE
# ==============================================================================

class AdvancedPerformanceMetrics:
    """Comprehensive performance metrics calculation engine"""
    
    @staticmethod
    def calculate_all_metrics(returns: pd.Series, 
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: float = 0.04,
                            periods_per_year: int = 252) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Ensure returns are clean
        returns_clean = returns.dropna()
        if len(returns_clean) < 30:
            return {"error": "Insufficient data points"}
        
        metrics = {}
        
        # 1. Basic Statistics
        metrics["Total Return"] = float((1 + returns_clean).prod() - 1)
        metrics["Annualized Return"] = float((1 + returns_clean.mean()) ** periods_per_year - 1)
        metrics["Annualized Volatility"] = float(returns_clean.std() * np.sqrt(periods_per_year))
        metrics["Skewness"] = float(returns_clean.skew())
        metrics["Kurtosis"] = float(returns_clean.kurtosis())
        metrics["Max Return"] = float(returns_clean.max())
        metrics["Min Return"] = float(returns_clean.min())
        
        # 2. Risk-Adjusted Ratios
        ann_vol = metrics["Annualized Volatility"]
        ann_ret = metrics["Annualized Return"]
        
        # Sharpe Ratio
        metrics["Sharpe Ratio"] = float((ann_ret - risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        metrics["Sortino Ratio"] = float((ann_ret - risk_free_rate) / downside_std) if downside_std > 0 else 0.0
        
        # Calmar Ratio
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        metrics["Max Drawdown"] = float(max_dd)
        metrics["Calmar Ratio"] = float(ann_ret / max_dd) if max_dd > 0 else 0.0
        
        # Omega Ratio
        threshold = risk_free_rate / periods_per_year
        excess_returns = returns_clean - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        metrics["Omega Ratio"] = float(gains / losses) if losses > 0 else float("inf")
        
        # 3. Advanced Risk Metrics
        # Value at Risk (Historical)
        metrics["VaR 95%"] = float(-np.percentile(returns_clean, 5))
        metrics["CVaR 95%"] = float(-returns_clean[returns_clean <= np.percentile(returns_clean, 5)].mean())
        
        # Tail Ratio
        metrics["Tail Ratio"] = float(abs(np.percentile(returns_clean, 95)) / abs(np.percentile(returns_clean, 5)))
        
        # Gain to Pain Ratio
        metrics["Gain to Pain Ratio"] = float(returns_clean.sum() / abs(returns_clean[returns_clean < 0].sum()))
        
        # 4. Distribution Metrics
        # Jarque-Bera test for normality
        from scipy.stats import jarque_bera
        jb_stat, jb_pvalue = jarque_bera(returns_clean)
        metrics["Jarque-Bera Stat"] = float(jb_stat)
        metrics["Jarque-Bera p-value"] = float(jb_pvalue)
        metrics["Is Normal"] = jb_pvalue > 0.05
        
        # 5. Benchmark-Adjusted Metrics (if benchmark provided)
        if benchmark_returns is not None and not benchmark_returns.empty:
            # Align returns with benchmark
            aligned_data = pd.concat([returns_clean, benchmark_returns], axis=1).dropna()
            if len(aligned_data) > 10:
                port_returns = aligned_data.iloc[:, 0]
                bench_returns = aligned_data.iloc[:, 1]
                
                # Beta and Alpha
                cov_matrix = np.cov(port_returns, bench_returns)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
                metrics["Beta"] = float(beta)
                
                bench_ann_ret = (1 + bench_returns.mean()) ** periods_per_year - 1
                metrics["Alpha"] = float(ann_ret - risk_free_rate - beta * (bench_ann_ret - risk_free_rate))
                
                # Information Ratio
                active_returns = port_returns - bench_returns
                tracking_error = active_returns.std() * np.sqrt(periods_per_year)
                metrics["Information Ratio"] = float(active_returns.mean() * periods_per_year / tracking_error) if tracking_error > 0 else 0.0
                
                # R-squared
                correlation = port_returns.corr(bench_returns)
                metrics["R-squared"] = float(correlation ** 2)
                metrics["Tracking Error"] = float(tracking_error)
        
        # 6. Time-Based Metrics
        # Win Rate
        metrics["Win Rate"] = float(len(returns_clean[returns_clean > 0]) / len(returns_clean))
        metrics["Average Win"] = float(returns_clean[returns_clean > 0].mean())
        metrics["Average Loss"] = float(returns_clean[returns_clean < 0].mean())
        metrics["Profit Factor"] = float(abs(returns_clean[returns_clean > 0].sum() / returns_clean[returns_clean < 0].sum()))
        
        # 7. Advanced Drawdown Analysis
        drawdown_series = drawdown
        underwater_duration = (drawdown_series < 0).astype(int)
        underwater_periods = []
        current_period = 0
        
        for val in underwater_duration:
            if val == 1:
                current_period += 1
            else:
                if current_period > 0:
                    underwater_periods.append(current_period)
                current_period = 0
        
        if underwater_periods:
            metrics["Avg Underwater Duration"] = float(np.mean(underwater_periods))
            metrics["Max Underwater Duration"] = float(np.max(underwater_periods))
        else:
            metrics["Avg Underwater Duration"] = 0.0
            metrics["Max Underwater Duration"] = 0.0
        
        # 8. Risk Decomposition
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        metrics["Upside Volatility"] = float(positive_returns.std() * np.sqrt(periods_per_year)) if len(positive_returns) > 0 else 0.0
        metrics["Downside Volatility"] = float(negative_returns.std() * np.sqrt(periods_per_year)) if len(negative_returns) > 0 else 0.0
        
        # 9. Risk/Return Efficiency
        metrics["Return per Unit Risk"] = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
        metrics["Volatility Skew"] = float(metrics["Upside Volatility"] / metrics["Downside Volatility"]) if metrics["Downside Volatility"] > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def create_performance_radar_chart(metrics: Dict) -> go.Figure:
        """Create radar chart for key performance metrics"""
        
        # Select key metrics for radar chart
        radar_categories = [
            'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
            'Omega Ratio', 'Win Rate', 'Profit Factor'
        ]
        
        radar_values = []
        for cat in radar_categories:
            value = metrics.get(cat, 0)
            # Normalize values for radar chart
            if cat == 'Sharpe Ratio':
                norm_value = min(value / 3, 1.0)  # Cap at 3
            elif cat == 'Sortino Ratio':
                norm_value = min(value / 4, 1.0)  # Cap at 4
            elif cat == 'Calmar Ratio':
                norm_value = min(value / 2, 1.0)  # Cap at 2
            elif cat == 'Omega Ratio':
                norm_value = min(value / 5, 1.0)  # Cap at 5
            elif cat == 'Win Rate':
                norm_value = value  # Already 0-1
            elif cat == 'Profit Factor':
                norm_value = min(value / 5, 1.0)  # Cap at 5
            else:
                norm_value = min(value, 1.0)
            radar_values.append(max(0, norm_value))
        
        fig = go.Figure(data=go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill='toself',
            fillcolor='rgba(26, 35, 126, 0.3)',
            line=dict(color='#1a237e', width=2),
            marker=dict(size=8, color='#283593')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(color='#424242'),
                    gridcolor='#e0e0e0'
                ),
                angularaxis=dict(
                    tickfont=dict(color='#424242'),
                    gridcolor='#e0e0e0'
                ),
                bgcolor='#f8f9fa'
            ),
            title="Performance Radar Chart",
            title_font=dict(color='#1a237e', size=16),
            showlegend=False,
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_rolling_metrics_chart(returns: pd.Series, 
                                   window: int = 63) -> go.Figure:
        """Create chart with rolling performance metrics"""
        
        # Calculate rolling metrics
        rolling_returns = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns * 252 / window) / rolling_vol
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Rolling Returns (63-day)", 
                          "Rolling Volatility (63-day)",
                          "Rolling Sharpe Ratio (63-day)"),
            vertical_spacing=0.1,
            row_heights=[0.33, 0.33, 0.33]
        )
        
        # Rolling Returns
        fig.add_trace(
            go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns * 100,
                mode='lines',
                name='Rolling Returns',
                line=dict(color='#1a237e', width=2),
                fill='tozeroy',
                fillcolor='rgba(26, 35, 126, 0.1)'
            ),
            row=1, col=1
        )
        
        # Rolling Volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol * 100,
                mode='lines',
                name='Rolling Vol',
                line=dict(color='#d32f2f', width=2),
                fill='tozeroy',
                fillcolor='rgba(211, 47, 47, 0.1)'
            ),
            row=2, col=1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='#388e3c', width=2),
                fill='tozeroy',
                fillcolor='rgba(56, 142, 60, 0.1)'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            template="plotly_white",
            title="Rolling Performance Metrics Analysis",
            title_font=dict(color='#1a237e', size=16),
            font=dict(color='#424242')
        )
        
        fig.update_yaxes(title_text="Return (%)", row=1, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1, title_font_color="#424242")
        
        return fig


# ==============================================================================
# 4) ADVANCED BLACK-LITTERMAN OPTIMIZATION ENGINE
# ==============================================================================

class AdvancedBlackLittermanOptimizer:
    """Advanced Black-Litterman portfolio optimization with multiple view types"""
    
    def __init__(self, df_prices: pd.DataFrame, risk_free_rate: float = 0.04):
        self.df_prices = df_prices
        self.returns = df_prices.pct_change().dropna()
        self.cov_matrix = self.returns.cov()
        self.num_assets = len(df_prices.columns)
        self.risk_free_rate = risk_free_rate
        
        # Initialize PyPortfolioOpt objects if available
        try:
            from pypfopt import risk_models, expected_returns
            self.mu = expected_returns.mean_historical_return(df_prices)
            self.S = risk_models.sample_cov(df_prices)
            self.OPTIMIZATION_AVAILABLE = True
        except:
            self.OPTIMIZATION_AVAILABLE = False
    
    def create_equilibrium_returns(self, method: str = "market_cap") -> pd.Series:
        """Calculate equilibrium returns using different methods"""
        
        if method == "market_cap":
            # Simplified market cap weighted returns
            weights = np.ones(self.num_assets) / self.num_assets
            equilibrium_returns = self.returns.mean() * 252
            
        elif method == "reverse_optimization":
            # Reverse optimization to get implied returns
            from scipy.optimize import minimize
            
            # Market portfolio (equally weighted for simplicity)
            market_weights = np.ones(self.num_assets) / self.num_assets
            market_return = self.returns.dot(market_weights).mean() * 252
            
            # Calculate implied returns
            implied_returns = market_return * self.cov_matrix.dot(market_weights) / \
                            (market_weights.T @ self.cov_matrix @ market_weights)
            equilibrium_returns = pd.Series(implied_returns, index=self.returns.columns)
            
        else:  # mean_historical
            equilibrium_returns = self.returns.mean() * 252
        
        return equilibrium_returns
    
    def create_view_matrix(self, views: Dict[str, float], 
                          view_types: Dict[str, str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create view matrix P and view vector Q from user views"""
        
        assets = list(self.df_prices.columns)
        num_views = len(views)
        
        P = np.zeros((num_views, self.num_assets))
        Q = np.zeros(num_views)
        view_types_dict = view_types or {}
        
        for i, (asset, view) in enumerate(views.items()):
            if asset in assets:
                asset_idx = assets.index(asset)
                view_type = view_types_dict.get(asset, "absolute")
                
                if view_type == "absolute":
                    # Absolute view: asset will return X%
                    P[i, asset_idx] = 1
                    Q[i] = view
                elif view_type == "relative":
                    # Relative view: asset A will outperform asset B by X%
                    # For simplicity, using first asset in relative view
                    P[i, asset_idx] = 1
                    Q[i] = view
                elif view_type == "ranking":
                    # Ranking view: asset rank position
                    P[i, asset_idx] = 1
                    Q[i] = view
        
        return P, Q
    
    def calculate_confidence_matrix(self, views: Dict[str, float], 
                                  confidence_method: str = "idzorek",
                                  confidences: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Calculate confidence/uncertainty matrix Omega"""
        
        num_views = len(views)
        
        if confidence_method == "idzorek":
            # Idzorek method: confidence levels mapped to uncertainty
            if confidences is None:
                confidences = {asset: 0.5 for asset in views.keys()}
            
            omega = np.zeros((num_views, num_views))
            for i, (asset, conf) in enumerate(confidences.items()):
                if asset in views:
                    # Convert confidence (0-1) to uncertainty
                    uncertainty = (1 - conf) * 0.1  # Scale factor
                    omega[i, i] = uncertainty
        
        elif confidence_method == "proportional":
            # Proportional to variance
            omega = np.eye(num_views) * 0.05  # 5% uncertainty
        
        else:  # scaled
            omega = np.eye(num_views) * 0.1
        
        return omega
    
    def black_litterman_returns(self, 
                               views: Dict[str, float],
                               view_types: Dict[str, str] = None,
                               confidences: Optional[Dict[str, float]] = None,
                               tau: float = 0.025) -> pd.Series:
        """Calculate Black-Litterman expected returns"""
        
        # Equilibrium returns
        pi = self.create_equilibrium_returns("market_cap")
        
        # Create view matrices
        P, Q = self.create_view_matrix(views, view_types)
        
        # Create confidence matrix
        omega = self.calculate_confidence_matrix(views, "idzorek", confidences)
        
        # Black-Litterman formula
        # E(R) = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1Π + P'Ω^-1Q]
        
        tau_sigma_inv = np.linalg.inv(tau * self.cov_matrix.values)
        omega_inv = np.linalg.inv(omega)
        
        term1 = tau_sigma_inv + P.T @ omega_inv @ P
        term2 = tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q
        
        bl_returns = np.linalg.inv(term1) @ term2
        
        return pd.Series(bl_returns, index=self.df_prices.columns)
    
    def optimize_portfolio(self, 
                          views: Dict[str, float],
                          view_types: Dict[str, str] = None,
                          confidences: Optional[Dict[str, float]] = None,
                          objective: str = "max_sharpe",
                          constraints: Dict = None) -> Dict:
        """Optimize portfolio using Black-Litterman model"""
        
        try:
            if not self.OPTIMIZATION_AVAILABLE:
                raise ImportError("PyPortfolioOpt not available")
            
            from pypfopt.black_litterman import BlackLittermanModel
            from pypfopt.efficient_frontier import EfficientFrontier
            
            # Create Black-Litterman model
            bl = BlackLittermanModel(
                cov_matrix=self.S,
                pi="equal",  # Equal prior
                absolute_views=views,
                omega="idzorek",
                view_confidences=list(confidences.values()) if confidences else None
            )
            
            # Get BL returns
            bl_returns = bl.bl_returns()
            
            # Optimize portfolio
            ef = EfficientFrontier(bl_returns, self.S)
            
            if objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            elif objective == "min_volatility":
                ef.min_volatility()
            elif objective == "max_quadratic_utility":
                ef.max_quadratic_utility()
            
            # Apply constraints if any
            if constraints:
                if "min_weight" in constraints:
                    ef.add_constraint(lambda w: w >= constraints["min_weight"])
                if "max_weight" in constraints:
                    ef.add_constraint(lambda w: w <= constraints["max_weight"])
            
            # Get optimized weights
            weights = ef.clean_weights()
            
            # Calculate performance
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
            
            return {
                "weights": weights,
                "expected_return": perf[0],
                "volatility": perf[1],
                "sharpe_ratio": perf[2],
                "bl_returns": bl_returns
            }
            
        except Exception as e:
            # Fallback to simplified optimization
            st.warning(f"Advanced BL optimization failed: {e}. Using simplified method.")
            return self._simplified_optimization(views, confidences)
    
    def _simplified_optimization(self, views: Dict[str, float], 
                                confidences: Optional[Dict[str, float]] = None) -> Dict:
        """Simplified optimization when PyPortfolioOpt is not available"""
        
        # Calculate BL returns
        bl_returns = self.black_litterman_returns(views, confidences=confidences)
        
        # Simple mean-variance optimization
        from scipy.optimize import minimize
        
        def portfolio_variance(weights):
            return weights.T @ self.cov_matrix.values @ weights
        
        def negative_sharpe(weights):
            port_return = weights @ bl_returns.values
            port_vol = np.sqrt(portfolio_variance(weights))
            return -(port_return - self.risk_free_rate) / port_vol
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # Non-negative
        ]
        
        # Initial guess (equal weights)
        init_weights = np.ones(self.num_assets) / self.num_assets
        
        # Optimize
        result = minimize(negative_sharpe, init_weights, 
                         constraints=constraints, 
                         bounds=[(0, 1) for _ in range(self.num_assets)])
        
        if result.success:
            weights = result.x
            weights = weights / weights.sum()  # Normalize
            weights_dict = {asset: float(w) for asset, w in zip(self.df_prices.columns, weights)}
            
            port_return = weights @ bl_returns.values
            port_vol = np.sqrt(portfolio_variance(weights))
            sharpe = (port_return - self.risk_free_rate) / port_vol
            
            return {
                "weights": weights_dict,
                "expected_return": float(port_return),
                "volatility": float(port_vol),
                "sharpe_ratio": float(sharpe),
                "bl_returns": bl_returns
            }
        else:
            # Return equal weights as fallback
            equal_weight = 1.0 / self.num_assets
            weights_dict = {asset: equal_weight for asset in self.df_prices.columns}
            
            port_return = equal_weight * bl_returns.mean() * self.num_assets
            port_vol = np.sqrt(np.diag(self.cov_matrix).mean())
            sharpe = (port_return - self.risk_free_rate) / port_vol
            
            return {
                "weights": weights_dict,
                "expected_return": float(port_return),
                "volatility": float(port_vol),
                "sharpe_ratio": float(sharpe),
                "bl_returns": bl_returns
            }
    
    def create_view_analysis_chart(self, 
                                 views: Dict[str, float],
                                 confidences: Optional[Dict[str, float]] = None,
                                 bl_returns: Optional[pd.Series] = None) -> go.Figure:
        """Create chart comparing equilibrium vs BL returns"""
        
        # Get equilibrium returns
        equilibrium_returns = self.create_equilibrium_returns("market_cap")
        
        # Get BL returns if not provided
        if bl_returns is None:
            bl_returns = self.black_litterman_returns(views, confidences=confidences)
        
        # Prepare data
        assets = list(self.df_prices.columns)
        eq_returns = equilibrium_returns.values * 100
        bl_returns_scaled = bl_returns.values * 100
        
        # User views for highlighting
        view_assets = list(views.keys())
        view_indices = [assets.index(asset) for asset in view_assets if asset in assets]
        
        fig = go.Figure()
        
        # Equilibrium returns
        fig.add_trace(go.Bar(
            x=assets,
            y=eq_returns,
            name='Equilibrium Returns',
            marker_color='rgba(158, 202, 225, 0.7)',
            text=[f"{r:.1f}%" for r in eq_returns],
            textposition='auto'
        ))
        
        # Black-Litterman returns
        fig.add_trace(go.Bar(
            x=assets,
            y=bl_returns_scaled,
            name='BL Adjusted Returns',
            marker_color='rgba(26, 35, 126, 0.7)',
            text=[f"{r:.1f}%" for r in bl_returns_scaled],
            textposition='auto'
        ))
        
        # Highlight assets with views
        if view_indices:
            highlighted_assets = [assets[i] for i in view_indices]
            highlighted_returns = [bl_returns_scaled[i] for i in view_indices]
            
            fig.add_trace(go.Scatter(
                x=highlighted_assets,
                y=highlighted_returns,
                mode='markers+text',
                name='Your Views',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='#ff9800',
                    line=dict(width=2, color='#ff5722')
                ),
                text=[f"View: {views[asset]:.1%}" for asset in highlighted_assets],
                textposition="top center"
            ))
        
        fig.update_layout(
            title="Black-Litterman: Equilibrium vs Adjusted Returns",
            title_font=dict(color='#1a237e', size=16),
            xaxis_title="Assets",
            yaxis_title="Expected Return (%)",
            barmode='group',
            template="plotly_white",
            height=500,
            font=dict(color='#424242'),
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
# 5) ADVANCED EFFICIENT FRONTIER CHART ENGINE
# ==============================================================================

class AdvancedEfficientFrontierChart:
    """Advanced efficient frontier visualization with multiple features"""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.04):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
    
    def generate_frontier_points(self, num_portfolios: int = 5000) -> Dict:
        """Generate random portfolios for efficient frontier"""
        
        num_assets = len(self.mean_returns)
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights = weights / weights.sum()
            
            # Calculate portfolio statistics
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = portfolio_sharpe
            weights_record.append(weights)
        
        return {
            "returns": results[0],
            "volatilities": results[1],
            "sharpes": results[2],
            "weights": weights_record
        }
    
    def calculate_optimal_portfolios(self) -> Dict:
        """Calculate key optimal portfolios"""
        
        portfolios = {}
        
        # 1. Minimum Variance Portfolio
        from scipy.optimize import minimize
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = tuple((0, 1) for _ in range(len(self.mean_returns)))
        
        init_weights = np.ones(len(self.mean_returns)) / len(self.mean_returns)
        
        # Min Volatility
        result = minimize(portfolio_volatility, init_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            min_var_weights = result.x
            portfolios["min_volatility"] = {
                "weights": min_var_weights,
                "return": float(np.dot(min_var_weights, self.mean_returns)),
                "volatility": float(portfolio_volatility(min_var_weights)),
                "sharpe": float((np.dot(min_var_weights, self.mean_returns) - self.risk_free_rate) / 
                               portfolio_volatility(min_var_weights))
            }
        
        # 2. Maximum Sharpe Ratio Portfolio
        def negative_sharpe(weights):
            port_return = np.dot(weights, self.mean_returns)
            port_vol = portfolio_volatility(weights)
            return -(port_return - self.risk_free_rate) / port_vol
        
        result = minimize(negative_sharpe, init_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            max_sharpe_weights = result.x
            portfolios["max_sharpe"] = {
                "weights": max_sharpe_weights,
                "return": float(np.dot(max_sharpe_weights, self.mean_returns)),
                "volatility": float(portfolio_volatility(max_sharpe_weights)),
                "sharpe": float((np.dot(max_sharpe_weights, self.mean_returns) - self.risk_free_rate) / 
                               portfolio_volatility(max_sharpe_weights))
            }
        
        # 3. Equal Weight Portfolio
        equal_weights = np.ones(len(self.mean_returns)) / len(self.mean_returns)
        portfolios["equal_weight"] = {
            "weights": equal_weights,
            "return": float(np.dot(equal_weights, self.mean_returns)),
            "volatility": float(portfolio_volatility(equal_weights)),
            "sharpe": float((np.dot(equal_weights, self.mean_returns) - self.risk_free_rate) / 
                           portfolio_volatility(equal_weights))
        }
        
        return portfolios
    
    def create_advanced_frontier_chart(self, 
                                     optimal_portfolios: Dict = None,
                                     user_portfolio: Dict = None,
                                     show_capital_market_line: bool = True) -> go.Figure:
        """Create advanced efficient frontier visualization"""
        
        # Generate frontier points
        frontier_data = self.generate_frontier_points(3000)
        
        # Calculate optimal portfolios if not provided
        if optimal_portfolios is None:
            optimal_portfolios = self.calculate_optimal_portfolios()
        
        # Create figure with multiple subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Efficient Frontier", 
                          "Sharpe Ratio Distribution",
                          "Risk Contribution",
                          "Portfolio Comparison"),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Efficient Frontier (top-left)
        colors = frontier_data["sharpes"]
        colors_normalized = (colors - colors.min()) / (colors.max() - colors.min())
        
        fig.add_trace(
            go.Scatter(
                x=frontier_data["volatilities"] * 100,
                y=frontier_data["returns"] * 100,
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors_normalized,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio", x=0.45, y=0.5)
                ),
                name='Feasible Portfolios',
                hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add optimal portfolios
        optimal_colors = {
            'min_volatility': '#d32f2f',
            'max_sharpe': '#388e3c',
            'equal_weight': '#ff9800'
        }
        
        optimal_symbols = {
            'min_volatility': 'circle',
            'max_sharpe': 'star',
            'equal_weight': 'square'
        }
        
        for port_name, port_data in optimal_portfolios.items():
            fig.add_trace(
                go.Scatter(
                    x=[port_data["volatility"] * 100],
                    y=[port_data["return"] * 100],
                    mode='markers+text',
                    marker=dict(
                        symbol=optimal_symbols.get(port_name, 'circle'),
                        size=15,
                        color=optimal_colors.get(port_name, '#757575'),
                        line=dict(width=2, color='white')
                    ),
                    name=port_name.replace('_', ' ').title(),
                    text=[port_name.replace('_', ' ').title()],
                    textposition="top center",
                    hovertemplate=f"{port_name.replace('_', ' ').title()}<br>"
                                f"Return: {port_data['return']*100:.2f}%<br>"
                                f"Vol: {port_data['volatility']*100:.2f}%<br>"
                                f"Sharpe: {port_data['sharpe']:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Add user portfolio if provided
        if user_portfolio:
            fig.add_trace(
                go.Scatter(
                    x=[user_portfolio.get("volatility", 0) * 100],
                    y=[user_portfolio.get("return", 0) * 100],
                    mode='markers+text',
                    marker=dict(
                        symbol='pentagon',
                        size=20,
                        color='#1a237e',
                        line=dict(width=2, color='white')
                    ),
                    name='Your Portfolio',
                    text=['Your Portfolio'],
                    textposition="top center",
                    hovertemplate="Your Portfolio<br>"
                                f"Return: {user_portfolio.get('return', 0)*100:.2f}%<br>"
                                f"Vol: {user_portfolio.get('volatility', 0)*100:.2f}%<br>"
                                f"Sharpe: {user_portfolio.get('sharpe', 0):.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Add Capital Market Line
        if show_capital_market_line and "max_sharpe" in optimal_portfolios:
            max_sharpe_port = optimal_portfolios["max_sharpe"]
            x_range = np.linspace(0, max(frontier_data["volatilities"]) * 100 * 1.2, 100)
            cml_y = self.risk_free_rate * 100 + (max_sharpe_port["sharpe"] * x_range)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=cml_y,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color='#ff5722', width=2, dash='dash'),
                    hovertemplate="CML: y = Rf + Sharpe × σ<extra></extra>"
                ),
                row=1, col=1
            )
        
        # 2. Sharpe Ratio Distribution (top-right)
        fig.add_trace(
            go.Histogram(
                x=frontier_data["sharpes"],
                nbinsx=50,
                name='Sharpe Distribution',
                marker_color='rgba(56, 142, 60, 0.7)',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add vertical line for max sharpe
        if "max_sharpe" in optimal_portfolios:
            fig.add_vline(
                x=optimal_portfolios["max_sharpe"]["sharpe"],
                line_dash="dash",
                line_color="#d32f2f",
                annotation_text=f"Max Sharpe: {optimal_portfolios['max_sharpe']['sharpe']:.2f}",
                row=1, col=2
            )
        
        # 3. Risk Contribution (bottom-left)
        if "max_sharpe" in optimal_portfolios:
            weights = optimal_portfolios["max_sharpe"]["weights"]
            mctr = weights * (self.cov_matrix @ weights)  # Marginal Contribution to Risk
            risk_contrib = mctr / np.sqrt(weights.T @ self.cov_matrix @ weights)
            
            assets = list(self.returns.columns)
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=risk_contrib * 100,
                    name='Risk Contribution (%)',
                    marker_color='rgba(26, 35, 126, 0.7)',
                    text=[f"{w*100:.1f}%" for w in weights],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # 4. Portfolio Comparison (bottom-right)
        if optimal_portfolios and user_portfolio:
            portfolios_to_compare = {**optimal_portfolios, "user_portfolio": user_portfolio}
            
            x_positions = []
            returns_data = []
            volatilities_data = []
            sharpe_data = []
            names = []
            
            for i, (name, data) in enumerate(portfolios_to_compare.items()):
                x_positions.append(i)
                returns_data.append(data.get("return", 0) * 100)
                volatilities_data.append(data.get("volatility", 0) * 100)
                sharpe_data.append(data.get("sharpe", 0))
                names.append(name.replace('_', ' ').title())
            
            # Returns comparison
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=returns_data,
                    name='Returns (%)',
                    marker_color='rgba(26, 35, 126, 0.6)',
                    text=[f"{r:.1f}%" for r in returns_data],
                    textposition='auto'
                ),
                row=2, col=2
            )
            
            # Volatilities comparison (as line)
            fig.add_trace(
                go.Scatter(
                    x=names,
                    y=volatilities_data,
                    mode='lines+markers',
                    name='Volatility (%)',
                    line=dict(color='#d32f2f', width=3),
                    marker=dict(size=10, symbol='diamond'),
                    yaxis='y2'
                ),
                row=2, col=2
            )
            
            # Add secondary y-axis for volatility
            fig.update_layout(
                yaxis2=dict(
                    title="Volatility (%)",
                    titlefont=dict(color="#d32f2f"),
                    tickfont=dict(color="#d32f2f"),
                    anchor="x",
                    overlaying="y",
                    side="right"
                )
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            template="plotly_white",
            title="Advanced Efficient Frontier Analysis",
            title_font=dict(color='#1a237e', size=18),
            font=dict(color='#424242'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=1, title_font_color="#424242")
        fig.update_yaxes(title_text="Return (%)", row=1, col=1, title_font_color="#424242")
        fig.update_xaxes(title_text="Sharpe Ratio", row=1, col=2, title_font_color="#424242")
        fig.update_yaxes(title_text="Frequency", row=1, col=2, title_font_color="#424242")
        fig.update_xaxes(title_text="Assets", row=2, col=1, title_font_color="#424242", tickangle=45)
        fig.update_yaxes(title_text="Risk Contribution (%)", row=2, col=1, title_font_color="#424242")
        fig.update_xaxes(title_text="Portfolio", row=2, col=2, title_font_color="#424242")
        fig.update_yaxes(title_text="Return (%)", row=2, col=2, title_font_color="#424242")
        
        return fig
    
    def create_3d_frontier_chart(self) -> go.Figure:
        """Create 3D efficient frontier visualization"""
        
        # Generate more points for smooth 3D surface
        frontier_data = self.generate_frontier_points(10000)
        
        # Calculate efficient frontier boundary
        from scipy.spatial import ConvexHull
        
        # Find convex hull of efficient portfolios
        points = np.column_stack([frontier_data["volatilities"], frontier_data["returns"]])
        hull = ConvexHull(points)
        
        # Get efficient frontier points
        frontier_points = points[hull.vertices]
        frontier_points = frontier_points[frontier_points[:, 0].argsort()]  # Sort by volatility
        
        # Create 3D surface
        fig = go.Figure(data=[
            go.Scatter3d(
                x=frontier_data["volatilities"] * 100,
                y=frontier_data["returns"] * 100,
                z=frontier_data["sharpes"],
                mode='markers',
                marker=dict(
                    size=3,
                    color=frontier_data["sharpes"],
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=True
                ),
                name='Feasible Portfolios'
            )
        ])
        
        # Add efficient frontier line
        if len(frontier_points) > 1:
            fig.add_trace(go.Scatter3d(
                x=frontier_points[:, 0] * 100,
                y=frontier_points[:, 1] * 100,
                z=(frontier_points[:, 1] - self.risk_free_rate) / frontier_points[:, 0],
                mode='lines',
                line=dict(color='#ff5722', width=4),
                name='Efficient Frontier'
            ))
        
        fig.update_layout(
            title="3D Efficient Frontier: Return vs Volatility vs Sharpe Ratio",
            title_font=dict(color='#1a237e', size=16),
            scene=dict(
                xaxis_title='Volatility (%)',
                yaxis_title='Return (%)',
                zaxis_title='Sharpe Ratio',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1)
                ),
                xaxis=dict(color='#424242'),
                yaxis=dict(color='#424242'),
                zaxis=dict(color='#424242')
            ),
            height=700,
            showlegend=True
        )
        
        return fig


# ==============================================================================
# 6) INTEGRATION WITH MAIN APP - NEW TABS AND FEATURES
# ==============================================================================

def create_advanced_performance_tab(df_prices: pd.DataFrame, 
                                  portfolio_returns: Optional[pd.Series] = None,
                                  benchmark_data: Optional[pd.Series] = None):
    """Create advanced performance metrics tab"""
    
    st.markdown('<div class="section-header">📊 Advanced Performance & Risk Metrics</div>', unsafe_allow_html=True)
    
    # Configuration
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        risk_free_rate = st.number_input(
            "Risk Free Rate (%)",
            value=4.5,
            step=0.1,
            key="perf_rf_rate"
        ) / 100
    
    with col_config2:
        if portfolio_returns is None:
            # Calculate simple equal weight portfolio returns
            returns = df_prices.pct_change().dropna()
            equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
            portfolio_returns = returns.dot(equal_weights)
            st.info("Using equal weight portfolio for analysis")
        else:
            st.success("Using optimized portfolio returns")
    
    with col_config3:
        # Calculate benchmark returns if available
        benchmark_returns = None
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Calculate metrics
    with st.spinner("🔬 Calculating comprehensive performance metrics..."):
        metrics_engine = AdvancedPerformanceMetrics()
        metrics = metrics_engine.calculate_all_metrics(
            portfolio_returns, 
            benchmark_returns,
            risk_free_rate
        )
    
    if "error" in metrics:
        st.error(metrics["error"])
        return
    
    # Display key metrics in a dashboard
    st.markdown('<div class="subsection-header">📈 Key Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Top row - Core metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annualized Return", f"{metrics['Annualized Return']*100:.2f}%")
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.3f}")
    
    with col2:
        st.metric("Annualized Volatility", f"{metrics['Annualized Volatility']*100:.2f}%")
        st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.3f}")
    
    with col3:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")
        st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.3f}")
    
    with col4:
        st.metric("Omega Ratio", f"{metrics['Omega Ratio']:.3f}")
        st.metric("Win Rate", f"{metrics['Win Rate']*100:.1f}%")
    
    # Detailed metrics in expandable sections
    with st.expander("📊 Detailed Risk Metrics", expanded=False):
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.metric("VaR 95% (1-day)", f"{metrics['VaR 95%']*100:.2f}%")
            st.metric("CVaR 95% (1-day)", f"{metrics['CVar 95%']*100:.2f}%")
            st.metric("Tail Ratio", f"{metrics['Tail Ratio']:.3f}")
        
        with col_risk2:
            st.metric("Skewness", f"{metrics['Skewness']:.3f}")
            st.metric("Kurtosis", f"{metrics['Kurtosis']:.3f}")
            st.metric("Jarque-Bera p-value", f"{metrics['Jarque-Bera p-value']:.4f}")
        
        with col_risk3:
            st.metric("Upside Volatility", f"{metrics['Upside Volatility']*100:.2f}%")
            st.metric("Downside Volatility", f"{metrics['Downside Volatility']*100:.2f}%")
            st.metric("Volatility Skew", f"{metrics['Volatility Skew']:.3f}")
    
    with st.expander("🏆 Performance Statistics", expanded=False):
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        
        with col_perf1:
            st.metric("Total Return", f"{metrics['Total Return']*100:.2f}%")
            st.metric("Profit Factor", f"{metrics['Profit Factor']:.3f}")
            st.metric("Gain to Pain Ratio", f"{metrics['Gain to Pain Ratio']:.3f}")
        
        with col_perf2:
            st.metric("Average Win", f"{metrics['Average Win']*100:.2f}%")
            st.metric("Average Loss", f"{metrics['Average Loss']*100:.2f}%")
            st.metric("Best Day", f"{metrics['Max Return']*100:.2f}%")
        
        with col_perf3:
            st.metric("Worst Day", f"{metrics['Min Return']*100:.2f}%")
            st.metric("Avg Underwater Duration", f"{metrics['Avg Underwater Duration']:.0f} days")
            st.metric("Max Underwater Duration", f"{metrics['Max Underwater Duration']:.0f} days")
    
    # Benchmark metrics if available
    if benchmark_returns is not None:
        with st.expander("📈 Benchmark Comparison", expanded=False):
            col_bench1, col_bench2, col_bench3 = st.columns(3)
            
            with col_bench1:
                st.metric("Alpha", f"{metrics.get('Alpha', 0)*100:.2f}%")
                st.metric("Beta", f"{metrics.get('Beta', 0):.3f}")
            
            with col_bench2:
                st.metric("Information Ratio", f"{metrics.get('Information Ratio', 0):.3f}")
                st.metric("R-squared", f"{metrics.get('R-squared', 0):.3f}")
            
            with col_bench3:
                st.metric("Tracking Error", f"{metrics.get('Tracking Error', 0)*100:.2f}%")
    
    # Visualizations
    st.markdown('<div class="subsection-header">📊 Performance Visualizations</div>', unsafe_allow_html=True)
    
    tab_viz1, tab_viz2 = st.tabs(["Performance Radar", "Rolling Metrics"])
    
    with tab_viz1:
        radar_chart = metrics_engine.create_performance_radar_chart(metrics)
        st.plotly_chart(radar_chart, use_container_width=True)
    
    with tab_viz2:
        rolling_chart = metrics_engine.create_rolling_metrics_chart(portfolio_returns)
        st.plotly_chart(rolling_chart, use_container_width=True)
    
    # Export metrics
    st.markdown('<div class="subsection-header">💾 Export Metrics</div>', unsafe_allow_html=True)
    
    if st.button("📥 Export Metrics to CSV", use_container_width=True):
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        csv = df_metrics.to_csv()
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def create_black_litterman_tab(df_prices: pd.DataFrame):
    """Create Black-Litterman optimization tab"""
    
    st.markdown('<div class="section-header">🎯 Black-Litterman Portfolio Optimization</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    💡 <b>Black-Litterman Model:</b> Combines market equilibrium with your views to create customized expected returns.
    Enter your market views below to adjust the portfolio optimization.
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        risk_free_rate = st.number_input(
            "Risk Free Rate (%)",
            value=4.5,
            step=0.1,
            key="bl_rf_rate"
        ) / 100
        
        tau_value = st.slider(
            "Uncertainty Parameter (τ)",
            min_value=0.001,
            max_value=0.05,
            value=0.025,
            step=0.001,
            help="Controls uncertainty in equilibrium returns"
        )
    
    with col_config2:
        confidence_level = st.select_slider(
            "Default Confidence Level",
            options=[0.1, 0.3, 0.5, 0.7, 0.9],
            value=0.7,
            help="Confidence in your views (higher = more weight on views)"
        )
        
        optimization_objective = st.selectbox(
            "Optimization Objective",
            ["max_sharpe", "min_volatility", "max_quadratic_utility"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    # Initialize optimizer
    optimizer = AdvancedBlackLittermanOptimizer(df_prices, risk_free_rate)
    
    # Asset selection for views
    st.markdown('<div class="subsection-header">📝 Enter Your Market Views</div>', unsafe_allow_html=True)
    
    # Quick view templates
    st.markdown("### ⚡ Quick View Templates")
    col_temp1, col_temp2, col_temp3 = st.columns(3)
    
    with col_temp1:
        if st.button("Bullish on Tech", use_container_width=True):
            st.session_state.bl_views = {
                "AAPL": 0.15,
                "MSFT": 0.14,
                "NVDA": 0.25,
                "GOOGL": 0.12
            }
            st.session_state.bl_conf = {k: 0.8 for k in st.session_state.bl_views.keys()}
    
    with col_temp2:
        if st.button("Bearish on Bonds", use_container_width=True):
            st.session_state.bl_views = {
                "TLT": 0.02,
                "BND": 0.03,
                "AGG": 0.035
            }
            st.session_state.bl_conf = {k: 0.7 for k in st.session_state.bl_views.keys()}
    
    with col_temp3:
        if st.button("Neutral Market", use_container_width=True):
            st.session_state.bl_views = {}
            st.session_state.bl_conf = {}
    
    # Initialize session state for views
    if "bl_views" not in st.session_state:
        st.session_state.bl_views = {}
    if "bl_conf" not in st.session_state:
        st.session_state.bl_conf = {}
    
    # View creation interface
    col_view1, col_view2 = st.columns(2)
    
    with col_view1:
        st.markdown("#### Add New View")
        
        selected_asset = st.selectbox(
            "Select Asset",
            options=df_prices.columns.tolist(),
            key="bl_new_asset"
        )
        
        expected_return = st.number_input(
            "Expected Annual Return (%)",
            min_value=-50.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            key="bl_new_return"
        ) / 100
        
        confidence = st.slider(
            "Confidence Level",
            min_value=0.1,
            max_value=1.0,
            value=confidence_level,
            step=0.1,
            key="bl_new_conf"
        )
        
        view_type = st.selectbox(
            "View Type",
            options=["absolute", "relative", "ranking"],
            format_func=lambda x: x.title(),
            key="bl_new_type"
        )
        
        col_add1, col_add2 = st.columns(2)
        with col_add1:
            if st.button("➕ Add View", use_container_width=True):
                st.session_state.bl_views[selected_asset] = expected_return
                st.session_state.bl_conf[selected_asset] = confidence
        
        with col_add2:
            if st.button("🗑️ Clear All Views", use_container_width=True):
                st.session_state.bl_views = {}
                st.session_state.bl_conf = {}
    
    with col_view2:
        st.markdown("#### Current Views")
        
        if st.session_state.bl_views:
            views_df = pd.DataFrame({
                "Asset": list(st.session_state.bl_views.keys()),
                "Expected Return": [f"{v*100:.1f}%" for v in st.session_state.bl_views.values()],
                "Confidence": [f"{st.session_state.bl_conf.get(k, 0.5)*100:.0f}%" 
                             for k in st.session_state.bl_views.keys()]
            })
            st.dataframe(views_df, use_container_width=True, height=200)
        else:
            st.info("No views added yet. Add views above or use quick templates.")
    
    # Run optimization
    st.markdown('<div class="subsection-header">🚀 Run Black-Litterman Optimization</div>', unsafe_allow_html=True)
    
    if st.button("⚡ Run Black-Litterman Optimization", type="primary", use_container_width=True):
        if not st.session_state.bl_views:
            st.warning("Please add at least one view to use Black-Litterman optimization.")
            return
        
        with st.spinner("Running Black-Litterman optimization..."):
            try:
                # Run optimization
                result = optimizer.optimize_portfolio(
                    views=st.session_state.bl_views,
                    confidences=st.session_state.bl_conf,
                    objective=optimization_objective
                )
                
                if result:
                    st.success("✅ Black-Litterman optimization completed!")
                    
                    # Display results
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.markdown("#### 📊 Optimized Portfolio")
                        weights_df = pd.DataFrame.from_dict(
                            result["weights"], 
                            orient='index', 
                            columns=['Weight']
                        ).sort_values('Weight', ascending=False)
                        
                        # Format for display
                        display_weights = weights_df[weights_df['Weight'] > 0.001].copy()
                        display_weights['Weight'] = display_weights['Weight'].apply(
                            lambda x: f"{x:.2%}"
                        )
                        display_weights['Amount ($1M)'] = (weights_df['Weight'] * 1000000).apply(
                            lambda x: f"${x:,.0f}"
                        )
                        
                        st.dataframe(
                            display_weights,
                            use_container_width=True,
                            height=400
                        )
                    
                    with col_res2:
                        st.markdown("#### 📈 Performance Summary")
                        col_perf1, col_perf2 = st.columns(2)
                        
                        with col_perf1:
                            st.metric(
                                "Expected Return",
                                f"{result['expected_return']*100:.2f}%"
                            )
                            st.metric(
                                "Volatility",
                                f"{result['volatility']*100:.2f}%"
                            )
                        
                        with col_perf2:
                            st.metric(
                                "Sharpe Ratio",
                                f"{result['sharpe_ratio']:.3f}"
                            )
                    
                    # Store results for other tabs
                    st.session_state.bl_results = result
                    
                    # Create visualization
                    st.markdown("#### 📊 Black-Litterman Analysis")
                    
                    view_chart = optimizer.create_view_analysis_chart(
                        views=st.session_state.bl_views,
                        confidences=st.session_state.bl_conf,
                        bl_returns=result.get("bl_returns")
                    )
                    st.plotly_chart(view_chart, use_container_width=True)
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")


def create_advanced_frontier_tab(df_prices: pd.DataFrame):
    """Create advanced efficient frontier visualization tab"""
    
    st.markdown('<div class="section-header">📈 Advanced Efficient Frontier Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    📊 <b>Efficient Frontier:</b> Visualizes the optimal portfolios that offer the highest expected return for a given level of risk.
    Explore the trade-off between risk and return with advanced visualizations.
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        risk_free_rate = st.number_input(
            "Risk Free Rate (%)",
            value=4.5,
            step=0.1,
            key="frontier_rf_rate"
        ) / 100
        
        num_portfolios = st.slider(
            "Number of Random Portfolios",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=1000,
            help="More portfolios = smoother frontier but slower computation"
        )
    
    with col_config2:
        show_cml = st.checkbox(
            "Show Capital Market Line",
            value=True,
            help="Show line connecting risk-free asset to optimal portfolio"
        )
        
        show_user_portfolio = st.checkbox(
            "Include Current Portfolio",
            value=True,
            help="Include current portfolio in comparison"
        )
    
    # Get returns data
    returns = df_prices.pct_change().dropna()
    
    # Initialize frontier engine
    frontier_engine = AdvancedEfficientFrontierChart(returns, risk_free_rate)
    
    # Calculate user portfolio if exists
    user_portfolio = None
    if show_user_portfolio and "portfolio_returns" in st.session_state:
        # Extract from session state
        portfolio_returns = st.session_state.get("portfolio_returns")
        if portfolio_returns is not None and len(portfolio_returns) > 0:
            user_return = portfolio_returns.mean() * 252
            user_vol = portfolio_returns.std() * np.sqrt(252)
            user_sharpe = (user_return - risk_free_rate) / user_vol
            
            user_portfolio = {
                "return": user_return,
                "volatility": user_vol,
                "sharpe": user_sharpe
            }
    
    # Generate frontier visualization
    st.markdown('<div class="subsection-header">📊 Efficient Frontier Visualization</div>', unsafe_allow_html=True)
    
    tab_frontier1, tab_frontier2 = st.tabs(["2D Analysis", "3D Visualization"])
    
    with tab_frontier1:
        with st.spinner("🔄 Generating efficient frontier..."):
            # Calculate optimal portfolios
            optimal_portfolios = frontier_engine.calculate_optimal_portfolios()
            
            # Create advanced chart
            frontier_chart = frontier_engine.create_advanced_frontier_chart(
                optimal_portfolios=optimal_portfolios,
                user_portfolio=user_portfolio,
                show_capital_market_line=show_cml
            )
            
            st.plotly_chart(frontier_chart, use_container_width=True)
        
        # Display optimal portfolios comparison
        if optimal_portfolios:
            st.markdown("#### 🏆 Optimal Portfolios Comparison")
            
            comparison_data = []
            for port_name, port_data in optimal_portfolios.items():
                comparison_data.append({
                    "Portfolio": port_name.replace('_', ' ').title(),
                    "Return": f"{port_data['return']*100:.2f}%",
                    "Volatility": f"{port_data['volatility']*100:.2f}%",
                    "Sharpe Ratio": f"{port_data['sharpe']:.3f}"
                })
            
            if user_portfolio:
                comparison_data.append({
                    "Portfolio": "Your Portfolio",
                    "Return": f"{user_portfolio['return']*100:.2f}%",
                    "Volatility": f"{user_portfolio['volatility']*100:.2f}%",
                    "Sharpe Ratio": f"{user_portfolio['sharpe']:.3f}"
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(
                df_comparison.style.highlight_max(subset=['Sharpe Ratio'], color='#c8e6c9')
                                 .highlight_max(subset=['Return'], color='#bbdefb')
                                 .highlight_min(subset=['Volatility'], color='#ffccbc'),
                use_container_width=True
            )
    
    with tab_frontier2:
        with st.spinner("🔄 Generating 3D visualization..."):
            frontier_3d = frontier_engine.create_3d_frontier_chart()
            st.plotly_chart(frontier_3d, use_container_width=True)
    
    # Portfolio simulation
    st.markdown('<div class="subsection-header">🎯 Portfolio Simulation</div>', unsafe_allow_html=True)
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        target_return = st.number_input(
            "Target Annual Return (%)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=1.0
        ) / 100
    
    with col_sim2:
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Very Low", "Low", "Medium", "High", "Very High"],
            value="Medium"
        )
    
    with col_sim3:
        max_weight = st.slider(
            "Maximum Asset Weight",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Maximum allocation to any single asset"
        )
    
    if st.button("🎯 Find Optimal Portfolio for Target", use_container_width=True):
        with st.spinner("Finding optimal portfolio..."):
            try:
                from scipy.optimize import minimize
                
                def portfolio_variance(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(frontier_engine.cov_matrix, weights)))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.dot(x, frontier_engine.mean_returns) - target_return}
                ]
                
                bounds = [(0, max_weight) for _ in range(len(frontier_engine.mean_returns))]
                init_weights = np.ones(len(frontier_engine.mean_returns)) / len(frontier_engine.mean_returns)
                
                result = minimize(portfolio_variance, init_weights,
                                 method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    optimal_weights = result.x
                    optimal_vol = portfolio_variance(optimal_weights)
                    optimal_sharpe = (target_return - risk_free_rate) / optimal_vol
                    
                    st.success(f"✅ Found optimal portfolio for {target_return*100:.1f}% target return")
                    
                    # Display weights
                    weights_dict = {asset: float(w) for asset, w in 
                                  zip(df_prices.columns, optimal_weights) if w > 0.001}
                    
                    col_disp1, col_disp2 = st.columns(2)
                    
                    with col_disp1:
                        st.markdown("#### 📊 Optimal Allocation")
                        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=['Weight'])
                        weights_df = weights_df.sort_values('Weight', ascending=False)
                        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(weights_df, use_container_width=True)
                    
                    with col_disp2:
                        st.markdown("#### 📈 Portfolio Characteristics")
                        st.metric("Expected Return", f"{target_return*100:.2f}%")
                        st.metric("Expected Volatility", f"{optimal_vol*100:.2f}%")
                        st.metric("Expected Sharpe Ratio", f"{optimal_sharpe:.3f}")
                
                else:
                    st.warning("Could not find portfolio meeting all constraints. Try adjusting parameters.")
            
            except Exception as e:
                st.error(f"Portfolio simulation failed: {str(e)}")


# ==============================================================================
# 7) UPDATED MAIN FUNCTION WITH NEW TABS
# ==============================================================================

def main():
    st.markdown('<div class="main-header">⚡ QUANTUM | Advanced Portfolio Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Import necessary original classes
    from your_original_code import EnhancedDataManager, _fetch_and_align_data_cached
    
    # Initialize data manager
    dm = EnhancedDataManager()
    
    # Session state initialization for new features
    if "bl_views" not in st.session_state:
        st.session_state.bl_views = {}
    if "bl_conf" not in st.session_state:
        st.session_state.bl_conf = {}
    if "bl_results" not in st.session_state:
        st.session_state.bl_results = None
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
            if st.button("Tech Focus", use_container_width=True):
                st.session_state.selected_assets_preset = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
        
        with col_preset3:
            if st.button("Diversified", use_container_width=True):
                st.session_state.selected_assets_preset = ["SPY", "EEM", "TLT", "GLD", "BABA"]
        
        st.divider()
        
        # Asset selection
        selected_assets: List[str] = []
        default_assets = ["SPY", "TLT", "GLD", "AAPL", "MSFT", "NVDA"]
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
        selected_assets = list(dict.fromkeys(selected_assets))
        
        st.divider()
        
        # Data settings
        st.markdown('<div class="subsection-header" style="font-size: 1.1rem;">Data Settings</div>', unsafe_allow_html=True)
        start_date = st.date_input("Start Date", value=datetime(2018, 1, 1))
        min_data_length = st.slider("Minimum Data Points", 100, 1000, 252)
        
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not selected_assets:
        st.markdown('<div class="warning-card">⚠️ Please select at least one asset from the sidebar.</div>', unsafe_allow_html=True)
        return
    
    # Fetch and align data
    with st.spinner("🔄 Fetching and aligning data..."):
        df_prices, benchmark_data, data_report = _fetch_and_align_data_cached(
            selected_tickers=tuple(selected_assets),
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            min_data_length=min_data_length
        )
    
    if df_prices is None or df_prices.empty:
        st.markdown('<div class="warning-card">❌ No valid data available after alignment.</div>', unsafe_allow_html=True)
        return
    
    st.markdown(f'<div class="success-card">✅ Data ready for analysis: {len(df_prices)} data points, {len(df_prices.columns)} assets</div>', unsafe_allow_html=True)
    
    # Create enhanced tabs - REORDERED with new tabs first
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Advanced Performance",
        "🎯 Black-Litterman",
        "📈 Advanced Frontier",
        "📈 Data Overview",
        "⚖️ Portfolio Optimization",
        "🎲 Advanced VaR/ES",
        "⚠️ Stress Testing Lab"
    ])
    
    # --- NEW TAB 1: ADVANCED PERFORMANCE METRICS ---
    with tab1:
        create_advanced_performance_tab(df_prices, None, benchmark_data)
    
    # --- NEW TAB 2: BLACK-LITTERMAN OPTIMIZATION ---
    with tab2:
        create_black_litterman_tab(df_prices)
    
    # --- NEW TAB 3: ADVANCED EFFICIENT FRONTIER ---
    with tab3:
        create_advanced_frontier_tab(df_prices)
    
    # --- ORIGINAL TABS (renumbered) ---
    with tab4:
        # Original Data Overview tab
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
    
    with tab5:
        # Original Portfolio Optimization tab
        # ... [Keep your original optimization tab code here] ...
        pass
    
    with tab6:
        # Original VaR/ES tab
        # ... [Keep your original VaR tab code here] ...
        pass
    
    with tab7:
        # Original Stress Testing tab
        # ... [Keep your original stress testing tab code here] ...
        pass


# ==============================================================================
# 8) MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
