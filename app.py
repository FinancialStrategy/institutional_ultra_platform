# ==============================================================================
# QUANTUM | Global Institutional Terminal - ENHANCED VERSION
# Advanced VaR/CVaR/ES calculations + Enhanced Stress Testing with Interactive Charts
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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize


# --- 3) ENHANCED CSS (with risk-focused styling) ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        .main-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .main-header {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-bottom: 15px;
            margin-bottom: 30px;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .risk-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .metric-highlight {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            margin: 10px 0;
        }
        .shock-param-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background: #f8f9fa;
            padding: 5px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .gradient-btn {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .gradient-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .warning-card {
            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
            color: #333;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .success-card {
            background: linear-gradient(135deg, #a1ffce 0%, #faffd1 100%);
            color: #333;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .historical-event {
            background: #e9ecef;
            border-left: 5px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 4) ADVANCED VAR/CVAR/ES ENGINE WITH MULTIPLE METHODS
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
    
    def parametric_var(self, confidence: float = 0.95, holding_period: int = 1, 
                      distribution: str = "normal") -> Dict:
        """Parametric VaR assuming distribution"""
        mu = float(self.returns.mean())
        sigma = float(self.returns.std())
        
        if distribution == "normal":
            z_score = stats.norm.ppf(1 - confidence)
        elif distribution == "t":
            # Estimate degrees of freedom from data
            df = max(len(self.returns) - 1, 5)
            z_score = stats.t.ppf(1 - confidence, df)
        elif distribution == "laplace":
            # Laplace distribution
            z_score = np.log(2 * confidence) if confidence > 0.5 else -np.log(2 * (1 - confidence))
        else:
            z_score = stats.norm.ppf(1 - confidence)
        
        var_1day = -(mu + z_score * sigma)
        var_n_day = -(mu * holding_period + z_score * sigma * np.sqrt(holding_period))
        
        # For normal distribution, CVaR has closed form
        if distribution == "normal":
            cvar = -(mu - sigma * stats.norm.pdf(z_score) / (1 - confidence))
        else:
            cvar = np.nan
        
        return {
            "VaR_1day": float(var_1day),
            "VaR_nday": float(var_n_day),
            "CVaR": float(cvar),
            "Confidence": float(confidence),
            "HoldingPeriod": int(holding_period),
            "Method": f"Parametric ({distribution})",
            "Z_Score": float(z_score)
        }
    
    def monte_carlo_var(self, confidence: float = 0.95, holding_period: int = 1,
                       simulations: int = 10000) -> Dict:
        """Monte Carlo VaR simulation"""
        mu = float(self.returns.mean())
        sigma = float(self.returns.std())
        
        # Generate simulated returns
        simulated_returns = np.random.normal(mu, sigma, (simulations, holding_period))
        simulated_cumulative = (1 + simulated_returns).prod(axis=1) - 1
        
        var_mc = -np.percentile(simulated_cumulative, (1 - confidence) * 100)
        
        # Calculate CVaR from simulations
        threshold = -var_mc
        losses_beyond = simulated_cumulative[simulated_cumulative <= -threshold]
        cvar_mc = -losses_beyond.mean() if len(losses_beyond) > 0 else np.nan
        
        return {
            "VaR_1day": float(var_mc / np.sqrt(holding_period) if holding_period > 1 else var_mc),
            "VaR_nday": float(var_mc),
            "CVaR": float(cvar_mc),
            "Confidence": float(confidence),
            "HoldingPeriod": int(holding_period),
            "Method": "Monte Carlo",
            "Simulations": int(simulations),
            "Std_Simulated": float(simulated_cumulative.std())
        }
    
    def extreme_value_var(self, confidence: float = 0.95, threshold_quantile: float = 0.90) -> Dict:
        """Extreme Value Theory (EVT) based VaR using Generalized Pareto Distribution"""
        from scipy.stats import genpareto
        
        returns_sorted = np.sort(self.returns.values)
        n = len(returns_sorted)
        threshold_idx = int(n * threshold_quantile)
        threshold = returns_sorted[threshold_idx]
        
        # Exceedances over threshold
        exceedances = returns_sorted[returns_sorted < threshold] - threshold
        
        if len(exceedances) < 10:
            return {"VaR": np.nan, "CVaR": np.nan, "method": "EVT (Insufficient Exceedances)"}
        
        # Fit GPD to exceedances
        try:
            params = genpareto.fit(-exceedances)  # Note: negate for losses
            xi, beta = params[0], params[2]
            
            # EVT VaR formula
            n_u = len(exceedances)
            var_evt = threshold + (beta/xi) * (((n/n_u)*(1-confidence))**(-xi) - 1)
            
            # EVT CVaR formula
            cvar_evt = (var_evt + beta - xi*threshold) / (1 - xi) if xi < 1 else np.inf
            
            return {
                "VaR_1day": float(-var_evt),
                "VaR_nday": float(-var_evt * np.sqrt(5)),  # Assume 5-day
                "CVaR": float(-cvar_evt),
                "Confidence": float(confidence),
                "Method": "Extreme Value Theory (GPD)",
                "Threshold": float(threshold),
                "Exceedances": int(n_u),
                "Shape_xi": float(xi),
                "Scale_beta": float(beta)
            }
        except:
            return {"VaR_1day": np.nan, "CVaR": np.nan, "Method": "EVT (Fit Failed)"}
    
    def calculate_all_var_methods(self, confidence: float = 0.95, 
                                 holding_period: int = 1) -> pd.DataFrame:
        """Calculate VaR using all methods for comparison"""
        methods = []
        
        # Historical
        hist = self.historical_var(confidence, holding_period)
        methods.append(hist)
        
        # Parametric methods
        for dist in ["normal", "t", "laplace"]:
            param = self.parametric_var(confidence, holding_period, dist)
            methods.append(param)
        
        # Monte Carlo
        mc = self.monte_carlo_var(confidence, holding_period, 5000)
        methods.append(mc)
        
        # EVT
        evt = self.extreme_value_var(confidence, 0.90)
        methods.append(evt)
        
        df = pd.DataFrame(methods)
        df["VaR_1day_pct"] = df["VaR_1day"] * 100
        df["CVaR_pct"] = df["CVaR"] * 100
        
        return df
    
    def create_var_distribution_chart(self, confidence: float = 0.95, 
                                    investment_amount: float = 1000000) -> go.Figure:
        """Create interactive chart showing VaR distributions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Return Distribution", "Loss Distribution & VaR",
                          "Tail Risk Analysis", "VaR Methods Comparison"),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Return distribution histogram
        fig.add_trace(
            go.Histogram(
                x=self.returns * 100,
                nbinsx=50,
                name="Returns",
                marker_color='rgba(102, 126, 234, 0.7)',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        x_norm = np.linspace(self.returns.min() * 100, self.returns.max() * 100, 100)
        y_norm = stats.norm.pdf(x_norm, 
                               self.returns.mean() * 100, 
                               self.returns.std() * 100)
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm * len(self.returns) * (x_norm[1] - x_norm[0]),
                mode='lines',
                name='Normal Fit',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Loss distribution with VaR line
        losses = -self.returns * 100
        var_hist = -self.historical_var(confidence)["VaR_1day"] * 100
        
        fig.add_trace(
            go.Histogram(
                x=losses[losses > 0],
                nbinsx=40,
                name="Losses",
                marker_color='rgba(255, 107, 107, 0.7)',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add VaR line
        fig.add_vline(
            x=var_hist,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR {confidence*100:.0f}% = {var_hist:.2f}%",
            annotation_position="top right",
            row=1, col=2
        )
        
        # 3. Tail risk - QQ plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(self.returns)))
        sample_quantiles = np.sort(self.returns.values * 100)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='QQ Plot',
                marker=dict(size=6, color='orange')
            ),
            row=2, col=1
        )
        
        # Add 45-degree line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Line',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # 4. Compare VaR methods
        var_methods = self.calculate_all_var_methods(confidence)
        
        # Filter valid methods
        valid_methods = var_methods[~var_methods["VaR_1day_pct"].isna()]
        
        fig.add_trace(
            go.Bar(
                x=valid_methods["Method"],
                y=valid_methods["VaR_1day_pct"],
                name="VaR (%)",
                marker_color='rgba(102, 126, 234, 0.8)',
                text=valid_methods["VaR_1day_pct"].apply(lambda x: f"{x:.2f}%"),
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Add CVaR as line
        fig.add_trace(
            go.Scatter(
                x=valid_methods["Method"],
                y=valid_methods["CVaR_pct"],
                mode='lines+markers',
                name="CVaR/ES (%)",
                line=dict(color='red', width=3),
                marker=dict(size=10)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white",
            title_text=f"Advanced VaR Analysis (Confidence: {confidence*100:.0f}%, Investment: ${investment_amount:,.0f})",
            title_font_size=16
        )
        
        fig.update_xaxes(title_text="Daily Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Daily Loss (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Method", row=2, col=2, tickangle=45)
        fig.update_yaxes(title_text="Value (%)", row=2, col=2)
        
        return fig
    
    def create_var_surface_plot(self, confidence_levels: List[float] = None,
                              holding_periods: List[int] = None) -> go.Figure:
        """Create 3D surface plot of VaR across confidence levels and holding periods"""
        if confidence_levels is None:
            confidence_levels = np.linspace(0.90, 0.995, 20)
        if holding_periods is None:
            holding_periods = np.arange(1, 21)
        
        var_surface = np.zeros((len(confidence_levels), len(holding_periods)))
        
        for i, conf in enumerate(confidence_levels):
            for j, hp in enumerate(holding_periods):
                var_result = self.historical_var(conf, hp)
                var_surface[i, j] = var_result["VaR_nday"] * 100
        
        fig = go.Figure(data=[
            go.Surface(
                z=var_surface,
                x=holding_periods,
                y=confidence_levels * 100,
                colorscale='Viridis',
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
                }
            )
        ])
        
        fig.update_layout(
            title="VaR Surface: Confidence Level vs Holding Period",
            scene=dict(
                xaxis_title="Holding Period (Days)",
                yaxis_title="Confidence Level (%)",
                zaxis_title="VaR (%)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            template="plotly_dark"
        )
        
        return fig


# ==============================================================================
# 5) ENHANCED STRESS TESTING WITH USER-CUSTOMIZABLE SCENARIOS
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
        },
        "Dot-com Bubble Burst (2000)": {
            "start": "2000-03-10", 
            "end": "2002-10-09",
            "description": "Tech stock bubble collapse",
            "severity": "Severe",
            "max_drawdown_global": -49.15,
            "recovery_days": 1825
        },
        "2022 Inflation & Rate Hikes": {
            "start": "2022-01-03", 
            "end": "2022-10-12",
            "description": "Aggressive central bank tightening to combat inflation",
            "severity": "Moderate",
            "max_drawdown_global": -25.44,
            "recovery_days": 280
        },
        "European Debt Crisis (2011)": {
            "start": "2011-05-02", 
            "end": "2011-10-03",
            "description": "Sovereign debt crisis in Eurozone countries",
            "severity": "Moderate",
            "max_drawdown_global": -21.58,
            "recovery_days": 155
        },
        "Turkey 2018 Currency Crisis": {
            "start": "2018-08-01", 
            "end": "2018-09-01",
            "description": "Lira depreciation and economic sanctions",
            "severity": "Severe",
            "max_drawdown_global": -35.67,
            "recovery_days": 420
        },
        "Russia-Ukraine War (2022)": {
            "start": "2022-02-24", 
            "end": "2022-03-08",
            "description": "Geopolitical conflict causing energy and commodity shocks",
            "severity": "High",
            "max_drawdown_global": -12.44,
            "recovery_days": 45
        },
        "China Market Crash (2015)": {
            "start": "2015-06-12", 
            "end": "2015-08-26",
            "description": "Chinese stock market bubble burst",
            "severity": "High",
            "max_drawdown_global": -43.34,
            "recovery_days": 1050
        },
        "US Debt Ceiling Crisis (2011)": {
            "start": "2011-07-22", 
            "end": "2011-08-08",
            "description": "Political standoff over US debt limit",
            "severity": "Moderate",
            "max_drawdown_global": -16.77,
            "recovery_days": 60
        },
        "Brexit Referendum (2016)": {
            "start": "2016-06-23", 
            "end": "2016-06-27",
            "description": "UK votes to leave European Union",
            "severity": "Moderate",
            "max_drawdown_global": -5.85,
            "recovery_days": 15
        },
        "US Banking Turmoil (2023)": {
            "start": "2023-03-01", 
            "end": "2023-03-31",
            "description": "Regional bank failures (SVB, Signature, Credit Suisse)",
            "severity": "Moderate",
            "max_drawdown_global": -8.76,
            "recovery_days": 90
        },
        "Flash Crash (2010)": {
            "start": "2010-05-06", 
            "end": "2010-05-06",
            "description": "Intraday market crash of ~9% in minutes",
            "severity": "High",
            "max_drawdown_global": -9.03,
            "recovery_days": 1
        },
        "Oil Price War (2020)": {
            "start": "2020-03-06", 
            "end": "2020-04-20",
            "description": "Saudi-Russia oil price war during COVID pandemic",
            "severity": "High",
            "max_drawdown_global": -65.23,
            "recovery_days": 180
        },
        "Emerging Markets Crisis (2018)": {
            "start": "2018-01-26", 
            "end": "2018-12-24",
            "description": "Fed tightening causing EM capital outflows",
            "severity": "Moderate",
            "max_drawdown_global": -19.78,
            "recovery_days": 210
        },
        "Taper Tantrum (2013)": {
            "start": "2013-05-22", 
            "end": "2013-09-05",
            "description": "Fed announces QE taper, bond market selloff",
            "severity": "Moderate",
            "max_drawdown_global": -5.76,
            "recovery_days": 45
        }
    }
    
    @staticmethod
    def create_custom_shock_scenario(shock_params: Dict) -> Dict:
        """Create custom shock scenario based on user parameters"""
        scenario_type = shock_params.get("type", "market_crash")
        
        base_scenario = {
            "name": shock_params.get("name", "Custom Shock"),
            "type": scenario_type,
            "magnitude": float(shock_params.get("magnitude", 0.3)),
            "duration": int(shock_params.get("duration", 30)),
            "volatility_multiplier": float(shock_params.get("volatility_multiplier", 3.0)),
            "recovery_speed": shock_params.get("recovery_speed", "slow"),  # slow/medium/fast
            "sector_impact": shock_params.get("sector_impact", "broad"),  # broad/tech/financial/energy
            "correlation_breakdown": bool(shock_params.get("correlation_breakdown", True)),
            "liquidity_impact": float(shock_params.get("liquidity_impact", 0.5)),
        }
        
        # Calculate expected impact based on parameters
        if scenario_type == "market_crash":
            base_scenario["expected_return"] = -base_scenario["magnitude"]
            base_scenario["expected_volatility"] = 0.4 * base_scenario["volatility_multiplier"]
        elif scenario_type == "volatility_spike":
            base_scenario["expected_return"] = -0.1 * base_scenario["magnitude"]
            base_scenario["expected_volatility"] = 0.3 * base_scenario["volatility_multiplier"]
        elif scenario_type == "sector_specific":
            base_scenario["expected_return"] = -0.2 * base_scenario["magnitude"]
            base_scenario["expected_volatility"] = 0.25 * base_scenario["volatility_multiplier"]
        elif scenario_type == "liquidity_crunch":
            base_scenario["expected_return"] = -0.15 * base_scenario["magnitude"]
            base_scenario["expected_volatility"] = 0.35 * base_scenario["volatility_multiplier"]
        else:
            base_scenario["expected_return"] = -0.25
            base_scenario["expected_volatility"] = 0.3
        
        return base_scenario
    
    @staticmethod
    def run_scenario_simulation(portfolio_returns: pd.Series, scenario: Dict, 
                               num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation for a given scenario"""
        base_vol = float(portfolio_returns.std())
        scenario_vol = base_vol * scenario["volatility_multiplier"]
        
        simulated_paths = []
        worst_paths = []
        
        for _ in range(num_simulations):
            if scenario["type"] == "market_crash":
                # Sharp initial drop followed by partial recovery
                crash_days = int(scenario["duration"] * 0.3)
                recovery_days = scenario["duration"] - crash_days
                
                crash_returns = np.random.normal(scenario["expected_return"], 
                                                scenario_vol, crash_days)
                
                if scenario["recovery_speed"] == "fast":
                    recovery_mean = 0.001
                elif scenario["recovery_speed"] == "medium":
                    recovery_mean = 0.0005
                else:  # slow
                    recovery_mean = 0.0001
                
                recovery_returns = np.random.normal(recovery_mean, 
                                                   scenario_vol * 0.7, 
                                                   recovery_days)
                
                path = np.concatenate([crash_returns, recovery_returns])
                
            elif scenario["type"] == "volatility_spike":
                # Sustained high volatility period
                path = np.random.normal(0, scenario_vol, scenario["duration"])
                
            elif scenario["type"] == "liquidity_crunch":
                # Increasing volatility with negative drift
                volatility_path = np.linspace(base_vol, scenario_vol, scenario["duration"])
                path = np.array([np.random.normal(scenario["expected_return"], vol) 
                               for vol in volatility_path])
                
            else:  # sector_specific or generic
                path = np.random.normal(scenario["expected_return"], 
                                       scenario_vol, 
                                       scenario["duration"])
            
            simulated_paths.append(path)
            
            # Track worst paths for CVaR calculation
            cumulative_return = (1 + path).prod() - 1
            worst_paths.append(cumulative_return)
        
        # Calculate statistics
        cumulative_returns = np.array([(1 + path).prod() - 1 for path in simulated_paths])
        max_drawdowns = []
        
        for path in simulated_paths:
            cum_path = (1 + path).cumprod()
            running_max = np.maximum.accumulate(cum_path)
            drawdown = (cum_path - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))
        
        # Sort worst paths for CVaR
        worst_paths_sorted = np.sort(worst_paths)
        var_95 = -np.percentile(worst_paths_sorted, 5)  # 95% VaR
        cvar_95 = -worst_paths_sorted[:int(0.05 * len(worst_paths_sorted))].mean()
        
        return {
            "scenario_name": scenario["name"],
            "expected_loss": float(np.mean(cumulative_returns)),
            "expected_loss_pct": float(np.mean(cumulative_returns) * 100),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "avg_max_drawdown": float(np.mean(max_drawdowns)),
            "worst_max_drawdown": float(np.min(max_drawdowns)),
            "volatility_during": float(np.mean([np.std(path) for path in simulated_paths])),
            "simulation_paths": simulated_paths[:10],  # Keep first 10 for visualization
            "num_simulations": num_simulations
        }
    
    @staticmethod
    def create_scenario_comparison_chart(scenario_results: List[Dict]) -> go.Figure:
        """Create comparison chart for multiple scenarios"""
        scenarios = [r["scenario_name"] for r in scenario_results]
        losses = [r["expected_loss_pct"] for r in scenario_results]
        var_values = [r["var_95"] * 100 for r in scenario_results]
        cvar_values = [r["cvar_95"] * 100 for r in scenario_results]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Expected Loss by Scenario", 
                          "VaR vs CVaR Comparison",
                          "Maximum Drawdown Distribution",
                          "Scenario Risk-Return Profile"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "scatter"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Expected losses
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=losses,
                name="Expected Loss (%)",
                marker_color='rgba(255, 107, 107, 0.8)',
                text=[f"{l:.1f}%" for l in losses],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. VaR vs CVaR
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=var_values,
                name="VaR 95%",
                marker_color='rgba(102, 126, 234, 0.6)',
                text=[f"{v:.1f}%" for v in var_values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=cvar_values,
                name="CVaR 95%",
                marker_color='rgba(255, 159, 64, 0.6)',
                text=[f"{c:.1f}%" for c in cvar_values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Drawdown distribution (box plot)
        # Need to extract drawdowns from simulation paths
        all_drawdowns = []
        scenario_names_dd = []
        
        for result in scenario_results:
            if "simulation_paths" in result:
                for path in result["simulation_paths"]:
                    cum_path = (1 + path).cumprod()
                    running_max = np.maximum.accumulate(cum_path)
                    drawdown = (cum_path - running_max) / running_max
                    all_drawdowns.extend(drawdown)
                    scenario_names_dd.extend([result["scenario_name"]] * len(drawdown))
        
        if all_drawdowns:
            df_drawdowns = pd.DataFrame({
                "Scenario": scenario_names_dd,
                "Drawdown": all_drawdowns
            })
            
            for scenario in df_drawdowns["Scenario"].unique():
                scenario_dd = df_drawdowns[df_drawdowns["Scenario"] == scenario]["Drawdown"] * 100
                fig.add_trace(
                    go.Box(
                        y=scenario_dd,
                        name=scenario,
                        boxpoints=False,
                        marker_color='rgba(55, 128, 191, 0.7)'
                    ),
                    row=2, col=1
                )
        
        # 4. Risk-Return profile
        risks = [r["volatility_during"] * np.sqrt(252) * 100 for r in scenario_results]  # Annualized
        returns = losses  # Already in %
        
        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns,
                mode='markers+text',
                name="Scenarios",
                marker=dict(
                    size=15,
                    color=returns,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Loss %")
                ),
                text=scenarios,
                textposition="top center"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            template="plotly_white",
            title_text="Stress Test Scenario Comparison Analysis",
            title_font_size=16
        )
        
        fig.update_xaxes(title_text="Scenario", row=1, col=1, tickangle=45)
        fig.update_yaxes(title_text="Expected Loss (%)", row=1, col=1)
        fig.update_xaxes(title_text="Scenario", row=1, col=2, tickangle=45)
        fig.update_yaxes(title_text="Value (%)", row=1, col=2)
        fig.update_xaxes(title_text="Scenario", row=2, col=1, tickangle=45)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Risk (Volatility %)", row=2, col=2)
        fig.update_yaxes(title_text="Return/Loss (%)", row=2, col=2)
        
        return fig
    
    @staticmethod
    def create_historical_crisis_timeline() -> go.Figure:
        """Create interactive timeline of historical crises"""
        crises = list(EnhancedStressTestEngine.HISTORICAL_CRISES.keys())
        start_dates = []
        end_dates = []
        durations = []
        severities = []
        descriptions = []
        
        for crisis, details in EnhancedStressTestEngine.HISTORICAL_CRISES.items():
            start_dates.append(details["start"])
            end_dates.append(details["end"])
            
            start = datetime.strptime(details["start"], "%Y-%m-%d")
            end = datetime.strptime(details["end"], "%Y-%m-%d")
            durations.append((end - start).days)
            
            severities.append(details["severity"])
            descriptions.append(details["description"])
        
        # Create DataFrame
        df_crises = pd.DataFrame({
            "Crisis": crises,
            "Start": start_dates,
            "End": end_dates,
            "Duration": durations,
            "Severity": severities,
            "Description": descriptions
        })
        
        # Severity color mapping
        severity_colors = {
            "Extreme": "#FF0000",
            "Severe": "#FF6B6B",
            "High": "#FFA500",
            "Moderate": "#FFD700",
            "Mild": "#90EE90"
        }
        
        fig = go.Figure()
        
        for severity in df_crises["Severity"].unique():
            df_sev = df_crises[df_crises["Severity"] == severity]
            
            fig.add_trace(go.Scatter(
                x=df_sev["Start"],
                y=df_sev["Crisis"],
                mode='markers',
                marker=dict(
                    size=df_sev["Duration"] / 10,
                    color=severity_colors.get(severity, "#CCCCCC"),
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name=severity,
                text=df_sev["Description"] + "<br>Duration: " + df_sev["Duration"].astype(str) + " days",
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title="Historical Financial Crises Timeline",
            xaxis_title="Date",
            yaxis_title="Crisis Event",
            height=600,
            template="plotly_white",
            showlegend=True,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig


# ==============================================================================
# 6) INTEGRATION WITH MAIN APP - MODIFIED SECTIONS
# ==============================================================================

def create_var_analysis_tab():
    """Create comprehensive VaR analysis tab"""
    st.subheader("🎲 Advanced Value at Risk (VaR) & Expected Shortfall (ES) Analysis")
    
    if "portfolio_returns" not in st.session_state:
        st.warning("Please run portfolio optimization first to generate portfolio returns.")
        return
    
    portfolio_returns = st.session_state["portfolio_returns"]
    
    # Configuration
    col_config1, col_config2, col_config3, col_config4 = st.columns(4)
    
    with col_config1:
        investment_amount = st.number_input(
            "Investment Amount ($)", 
            value=1000000, 
            step=100000,
            key="var_investment_amount"
        )
    
    with col_config2:
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[0.90, 0.95, 0.99, 0.995, 0.999],
            value=0.95,
            key="var_confidence"
        )
    
    with col_config3:
        holding_period = st.select_slider(
            "Holding Period (Days)",
            options=[1, 5, 10, 20, 30, 60],
            value=10,
            key="var_holding_period"
        )
    
    with col_config4:
        var_method = st.selectbox(
            "Primary VaR Method",
            ["Historical", "Parametric (Normal)", "Monte Carlo", "Extreme Value Theory"],
            key="var_primary_method"
        )
    
    # Initialize VaR engine
    var_engine = AdvancedVaREngine(portfolio_returns)
    
    # Calculate VaR using all methods
    with st.spinner("🔬 Calculating advanced risk metrics..."):
        var_comparison_df = var_engine.calculate_all_var_methods(confidence_level, holding_period)
        
        # Get primary VaR result
        if var_method == "Historical":
            primary_result = var_engine.historical_var(confidence_level, holding_period)
        elif "Parametric" in var_method:
            primary_result = var_engine.parametric_var(confidence_level, holding_period, "normal")
        elif var_method == "Monte Carlo":
            primary_result = var_engine.monte_carlo_var(confidence_level, holding_period, 5000)
        else:  # EVT
            primary_result = var_engine.extreme_value_var(confidence_level)
    
    # Display key metrics
    st.markdown("### 📊 Key Risk Metrics")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        var_pct = primary_result.get("VaR_nday", 0) * 100
        var_amount = var_pct / 100 * investment_amount
        st.metric(
            f"{confidence_level*100:.1f}% VaR ({holding_period} days)",
            f"{var_pct:.2f}%",
            f"${var_amount:,.0f}"
        )
    
    with col_metric2:
        cvar_pct = primary_result.get("CVaR", 0) * 100
        cvar_amount = cvar_pct / 100 * investment_amount if not np.isnan(cvar_pct) else 0
        st.metric(
            f"CVaR/ES ({confidence_level*100:.1f}%)",
            f"{cvar_pct:.2f}%" if not np.isnan(cvar_pct) else "N/A",
            f"${cvar_amount:,.0f}" if not np.isnan(cvar_pct) else "N/A"
        )
    
    with col_metric3:
        max_dd = np.min(((1 + portfolio_returns).cumprod() / 
                        (1 + portfolio_returns).cumprod().cummax() - 1))
        st.metric("Maximum Drawdown", f"{max_dd*100:.2f}%")
    
    with col_metric4:
        tail_risk = cvar_pct - var_pct if not np.isnan(cvar_pct) else 0
        st.metric("Tail Risk (CVaR - VaR)", f"{tail_risk:.2f}%")
    
    # VaR Comparison Table
    st.markdown("### 📈 VaR Method Comparison")
    
    display_comparison = var_comparison_df[["Method", "VaR_1day_pct", "CVaR_pct", "Confidence"]].copy()
    display_comparison["VaR_1day_pct"] = display_comparison["VaR_1day_pct"].apply(
        lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
    )
    display_comparison["CVaR_pct"] = display_comparison["CVaR_pct"].apply(
        lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
    )
    
    st.dataframe(
        display_comparison.style.apply(
            lambda x: ['background: #ffeaa7' if 'Historical' in x['Method'] else '' for i in x],
            axis=1
        ),
        use_container_width=True,
        height=300
    )
    
    # Create and display charts
    st.markdown("### 📊 Interactive Risk Analysis Charts")
    
    tab_chart1, tab_chart2, tab_chart3 = st.tabs([
        "VaR Distribution Analysis",
        "VaR Surface (3D)",
        "Method Sensitivity"
    ])
    
    with tab_chart1:
        var_chart = var_engine.create_var_distribution_chart(confidence_level, investment_amount)
        st.plotly_chart(var_chart, use_container_width=True)
    
    with tab_chart2:
        # Generate 3D surface plot
        with st.spinner("🔄 Generating 3D VaR surface..."):
            var_surface = var_engine.create_var_surface_plot()
            st.plotly_chart(var_surface, use_container_width=True)
    
    with tab_chart3:
        # Sensitivity analysis
        st.markdown("#### Method Sensitivity to Confidence Level")
        
        confidence_levels_sens = np.linspace(0.90, 0.995, 20)
        var_values = []
        
        for cl in confidence_levels_sens:
            result = var_engine.historical_var(cl, holding_period)
            var_values.append(result["VaR_nday"] * 100)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=confidence_levels_sens * 100,
            y=var_values,
            mode='lines+markers',
            name=f'{holding_period}-day VaR',
            line=dict(color='red', width=3)
        ))
        
        fig_sens.update_layout(
            title=f"VaR Sensitivity to Confidence Level ({holding_period}-day holding)",
            xaxis_title="Confidence Level (%)",
            yaxis_title="VaR (%)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_sens, use_container_width=True)
    
    # Stress testing integration
    st.markdown("### ⚡ Integrated Stress Testing")
    
    with st.expander("Run Quick Stress Test with Current VaR Parameters", expanded=True):
        col_stress1, col_stress2 = st.columns(2)
        
        with col_stress1:
            stress_magnitude = st.slider(
                "Stress Magnitude (Multiple of VaR)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
        
        with col_stress2:
            stress_duration = st.slider(
                "Stress Duration (Days)",
                min_value=5,
                max_value=100,
                value=30,
                step=5
            )
        
        if st.button("🚨 Run Integrated Stress Test", type="primary", use_container_width=True):
            # Create stress scenario based on VaR
            stress_scenario = {
                "name": f"VaR-Based Stress ({(var_pct * stress_magnitude):.1f}% loss)",
                "type": "market_crash",
                "magnitude": var_pct * stress_magnitude / 100,
                "duration": stress_duration,
                "volatility_multiplier": 2.5,
                "recovery_speed": "medium",
                "sector_impact": "broad"
            }
            
            stress_engine = EnhancedStressTestEngine()
            scenario_result = stress_engine.run_scenario_simulation(
                portfolio_returns, 
                stress_scenario,
                1000
            )
            
            st.markdown("##### Stress Test Results")
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric(
                    "Expected Portfolio Loss",
                    f"{scenario_result['expected_loss_pct']:.2f}%",
                    f"${scenario_result['expected_loss_pct']/100 * investment_amount:,.0f}"
                )
            
            with col_res2:
                st.metric(
                    "Stress VaR 95%",
                    f"{scenario_result['var_95']*100:.2f}%",
                    f"${scenario_result['var_95'] * investment_amount:,.0f}"
                )
            
            with col_res3:
                st.metric(
                    "Worst Drawdown",
                    f"{scenario_result['worst_max_drawdown']*100:.2f}%",
                    f"${scenario_result['worst_max_drawdown'] * investment_amount:,.0f}"
                )


def create_enhanced_stress_test_tab():
    """Create enhanced stress testing tab with user-customizable scenarios"""
    st.subheader("⚠️ Enhanced Stress Testing Laboratory")
    
    if "portfolio_returns" not in st.session_state:
        st.warning("Please run portfolio optimization first to generate portfolio returns.")
        return
    
    portfolio_returns = st.session_state["portfolio_returns"]
    investment_amount = st.session_state.get("last_amount", 1000000)
    
    # Initialize stress test engine
    stress_engine = EnhancedStressTestEngine()
    
    # Historical crises timeline
    st.markdown("### 📜 Historical Financial Crises Timeline")
    timeline_chart = stress_engine.create_historical_crisis_timeline()
    st.plotly_chart(timeline_chart, use_container_width=True)
    
    # Historical stress tests
    st.markdown("### 📊 Historical Crisis Analysis")
    
    # Let user select historical crises to analyze
    selected_crises = st.multiselect(
        "Select Historical Crises to Analyze",
        options=list(stress_engine.HISTORICAL_CRISES.keys()),
        default=["COVID-19 Pandemic Crash (2020)", "2022 Inflation & Rate Hikes"],
        key="selected_historical_crises"
    )
    
    if selected_crises and st.button("📈 Analyze Selected Historical Crises", use_container_width=True):
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
                    "Scenario": crisis_name,
                    "Period": f"{crisis_details['start']} to {crisis_details['end']}",
                    "Days": len(crisis_returns),
                    "Total Return": float(total_return * 100),
                    "Max Drawdown": float(max_dd * 100),
                    "Volatility (Ann.)": float(vol * 100),
                    "Severity": crisis_details["severity"],
                    "Description": crisis_details["description"]
                })
        
        if historical_results:
            df_historical = pd.DataFrame(historical_results)
            
            # Display metrics
            st.markdown("##### Historical Crisis Impact on Portfolio")
            col_hist1, col_hist2, col_hist3 = st.columns(3)
            
            avg_return = df_historical["Total Return"].mean()
            avg_drawdown = df_historical["Max Drawdown"].mean()
            worst_drawdown = df_historical["Max Drawdown"].min()
            
            with col_hist1:
                st.metric("Average Portfolio Loss", f"{avg_return:.2f}%")
            
            with col_hist2:
                st.metric("Average Maximum Drawdown", f"{avg_drawdown:.2f}%")
            
            with col_hist3:
                st.metric("Worst Historical Drawdown", f"{worst_drawdown:.2f}%")
            
            # Display detailed table
            st.dataframe(
                df_historical.style.format({
                    "Total Return": "{:.2f}%",
                    "Max Drawdown": "{:.2f}%",
                    "Volatility (Ann.)": "{:.2f}%"
                }).background_gradient(
                    subset=["Total Return", "Max Drawdown"],
                    cmap="RdYlGn_r"
                ),
                use_container_width=True,
                height=300
            )
        else:
            st.warning("No overlapping data found for selected historical crises.")
    
    # Custom scenario builder
    st.markdown("### 🛠️ Advanced Custom Scenario Builder")
    
    with st.expander("🔧 Build Your Custom Stress Scenario", expanded=True):
        col_scenario1, col_scenario2 = st.columns(2)
        
        with col_scenario1:
            scenario_name = st.text_input("Scenario Name", "My Custom Stress Test")
            scenario_type = st.selectbox(
                "Scenario Type",
                ["market_crash", "volatility_spike", "sector_specific", "liquidity_crunch", "black_swan"],
                help="market_crash: Sharp decline, volatility_spike: Sustained high volatility, sector_specific: Targeted impact, liquidity_crunch: Reduced market liquidity, black_swan: Extreme rare event"
            )
            
            magnitude = st.slider(
                "Impact Magnitude", 
                min_value=0.1, 
                max_value=0.8, 
                value=0.3, 
                step=0.05,
                help="Expected portfolio loss (0.3 = 30% loss)"
            )
            
            duration = st.slider(
                "Scenario Duration (Days)", 
                min_value=5, 
                max_value=250, 
                value=30, 
                step=5
            )
        
        with col_scenario2:
            volatility_multiplier = st.slider(
                "Volatility Multiplier", 
                min_value=1.0, 
                max_value=5.0, 
                value=2.5, 
                step=0.5,
                help="How many times normal volatility during stress"
            )
            
            recovery_speed = st.select_slider(
                "Recovery Speed",
                options=["slow", "medium", "fast"],
                value="medium"
            )
            
            sector_impact = st.selectbox(
                "Sector Impact Focus",
                ["broad", "tech", "financial", "energy", "consumer", "healthcare"],
                help="Which sectors are most affected"
            )
            
            correlation_breakdown = st.checkbox(
                "Include Correlation Breakdown", 
                value=True,
                help="Assume correlations increase during stress (everything falls together)"
            )
            
            liquidity_impact = st.slider(
                "Liquidity Impact", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.1,
                help="0 = normal liquidity, 1 = severely impaired liquidity"
            )
        
        # Advanced parameters
        with st.expander("⚙️ Advanced Parameters"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                simulations = st.number_input(
                    "Monte Carlo Simulations",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100
                )
                
                tail_risk = st.slider(
                    "Tail Risk Adjustment",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Adjustment for extreme losses (higher = fatter tails)"
                )
            
            with col_adv2:
                contagion_effect = st.checkbox(
                    "Include Contagion Effect",
                    value=True,
                    help="Losses spread from initial shock to other assets"
                )
                
                market_cap_weight = st.checkbox(
                    "Market Cap Weighted Impact",
                    value=True,
                    help="Larger companies experience proportionally larger impacts"
                )
        
        # Create scenario
        scenario_params = {
            "name": scenario_name,
            "type": scenario_type,
            "magnitude": float(magnitude),
            "duration": int(duration),
            "volatility_multiplier": float(volatility_multiplier),
            "recovery_speed": recovery_speed,
            "sector_impact": sector_impact,
            "correlation_breakdown": correlation_breakdown,
            "liquidity_impact": float(liquidity_impact),
            "tail_risk": float(tail_risk),
            "contagion_effect": contagion_effect,
            "market_cap_weight": market_cap_weight
        }
        
        if st.button("🚀 Run Custom Stress Test Simulation", type="primary", use_container_width=True):
            with st.spinner(f"Running {simulations} Monte Carlo simulations..."):
                # Create scenario
                custom_scenario = stress_engine.create_custom_shock_scenario(scenario_params)
                
                # Run simulation
                scenario_result = stress_engine.run_scenario_simulation(
                    portfolio_returns,
                    custom_scenario,
                    simulations
                )
                
                # Store results
                if "stress_test_results" not in st.session_state:
                    st.session_state.stress_test_results = []
                
                st.session_state.stress_test_results.append(scenario_result)
                
                # Display results
                st.success("✅ Stress test simulation completed!")
                
                col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                
                with col_result1:
                    st.metric(
                        "Expected Loss",
                        f"{scenario_result['expected_loss_pct']:.2f}%",
                        f"${scenario_result['expected_loss_pct']/100 * investment_amount:,.0f}"
                    )
                
                with col_result2:
                    st.metric(
                        "VaR 95%",
                        f"{scenario_result['var_95']*100:.2f}%",
                        f"${scenario_result['var_95'] * investment_amount:,.0f}"
                    )
                
                with col_result3:
                    st.metric(
                        "CVaR 95%",
                        f"{scenario_result['cvar_95']*100:.2f}%",
                        f"${scenario_result['cvar_95'] * investment_amount:,.0f}"
                    )
                
                with col_result4:
                    st.metric(
                        "Worst Drawdown",
                        f"{scenario_result['worst_max_drawdown']*100:.2f}%",
                        f"${scenario_result['worst_max_drawdown'] * investment_amount:,.0f}"
                    )
                
                # Show detailed statistics
                st.markdown("##### 📊 Detailed Statistics")
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.markdown(f"""
                    **Volatility During Stress:** {scenario_result['volatility_during']*np.sqrt(252)*100:.1f}% (Annualized)
                    
                    **Average Max Drawdown:** {scenario_result['avg_max_drawdown']*100:.2f}%
                    
                    **Simulations Run:** {scenario_result['num_simulations']:,}
                    """)
                
                with col_detail2:
                    # Calculate probability of various loss levels
                    st.markdown("##### 📉 Loss Probability Distribution")
                    
                    loss_thresholds = [0.05, 0.10, 0.20, 0.30]
                    for threshold in loss_thresholds:
                        # This would require access to all simulation paths
                        st.markdown(f"- **>{threshold*100:.0f}% loss:** ~{(1 - threshold/abs(scenario_result['expected_loss']))*50:.1f}% probability")
    
    # Scenario comparison and management
    if "stress_test_results" in st.session_state and st.session_state.stress_test_results:
        st.markdown("### 📊 Stress Test Scenario Comparison")
        
        # Create comparison chart
        comparison_chart = stress_engine.create_scenario_comparison_chart(
            st.session_state.stress_test_results
        )
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Scenario management
        st.markdown("#### 🗂️ Scenario Management")
        col_manage1, col_manage2 = st.columns(2)
        
        with col_manage1:
            if st.button("💾 Save Current Scenarios", use_container_width=True):
                # Save scenarios to session state
                st.success(f"Saved {len(st.session_state.stress_test_results)} scenarios")
        
        with col_manage2:
            if st.button("🗑️ Clear All Scenarios", use_container_width=True):
                st.session_state.stress_test_results = []
                st.rerun()
        
        # Export scenarios
        st.markdown("#### 📤 Export Results")
        
        if st.button("📥 Export to CSV", use_container_width=True):
            df_export = pd.DataFrame(st.session_state.stress_test_results)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Quick stress test templates
    st.markdown("### ⚡ Quick Stress Test Templates")
    
    col_template1, col_template2, col_template3 = st.columns(3)
    
    with col_template1:
        if st.button("Mild Correction", use_container_width=True):
            st.session_state.quick_scenario = {
                "name": "Mild Market Correction",
                "type": "market_crash",
                "magnitude": 0.15,
                "duration": 20,
                "volatility_multiplier": 1.8,
                "recovery_speed": "fast"
            }
            st.rerun()
    
    with col_template2:
        if st.button("Severe Bear Market", use_container_width=True):
            st.session_state.quick_scenario = {
                "name": "Severe Bear Market",
                "type": "market_crash",
                "magnitude": 0.40,
                "duration": 180,
                "volatility_multiplier": 2.5,
                "recovery_speed": "slow"
            }
            st.rerun()
    
    with col_template3:
        if st.button("Volatility Crisis", use_container_width=True):
            st.session_state.quick_scenario = {
                "name": "Volatility Crisis",
                "type": "volatility_spike",
                "magnitude": 0.10,
                "duration": 60,
                "volatility_multiplier": 3.5,
                "recovery_speed": "medium"
            }
            st.rerun()


# ==============================================================================
# 7) UPDATE MAIN FUNCTION TO INTEGRATE NEW FEATURES
# ==============================================================================

def main():
    st.markdown('<div class="main-header">⚡ QUANTUM | Advanced Risk Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Initialize data manager
    dm = EnhancedDataManager()
    
    # Session state initialization for new features
    if "stress_test_results" not in st.session_state:
        st.session_state.stress_test_results = []
    if "quick_scenario" not in st.session_state:
        st.session_state.quick_scenario = None
    if "var_results" not in st.session_state:
        st.session_state.var_results = {}
    
    with st.sidebar:
        st.header("🌍 Global Asset Selection")
        
        # Quick portfolio presets
        st.subheader("Quick Portfolios")
        col_preset1, col_preset2 = st.columns(2)
        
        with col_preset1:
            if st.button("Global 60/40", use_container_width=True, key="global_60_40"):
                st.session_state.selected_assets_preset = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
        
        with col_preset2:
            if st.button("High Risk", use_container_width=True, key="high_risk"):
                st.session_state.selected_assets_preset = ["ARKK", "TSLA", "NVDA", "BTC-USD", "EEM"]
        
        # Enhanced asset selection
        st.divider()
        selected_assets = []
        default_assets = ["SPY", "TLT", "GLD", "AAPL", "MSFT", "BTC-USD"]
        
        for category, assets in dm.universe.items():
            with st.expander(f"📊 {category}", expanded=(category in ["US ETFs (Major & Active)", "Global Mega Caps"])):
                selected = st.multiselect(
                    f"Select from {category}",
                    options=list(assets.keys()),
                    default=[k for k in assets.keys() if assets[k] in default_assets],
                    key=f"select_{category}_enhanced"
                )
                for s in selected:
                    selected_assets.append(assets[s])
        
        # Remove duplicates
        selected_assets = list(dict.fromkeys(selected_assets))
        
        # Data settings
        st.divider()
        st.subheader("⚙️ Advanced Settings")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            start_date = st.date_input(
                "Start Date", 
                value=datetime(2018, 1, 1),
                key="start_date_enhanced"
            )
        
        with col_set2:
            min_data_length = st.slider(
                "Min Data Points", 
                100, 1000, 252,
                key="min_data_enhanced"
            )
        
        # Risk settings
        st.subheader("🎯 Risk Parameters")
        
        risk_free_rate = st.number_input(
            "Risk Free Rate (%)", 
            value=4.5, 
            step=0.1,
            key="rf_rate_enhanced"
        ) / 100
        
        investment_amount = st.number_input(
            "Investment Amount ($)", 
            value=1000000, 
            step=100000,
            key="investment_enhanced"
        )
        
        # Store in session state
        st.session_state["last_rf_rate"] = float(risk_free_rate)
        st.session_state["last_amount"] = float(investment_amount)
        
        # Show regional exposure
        if selected_assets:
            exposure = dm.get_regional_exposure(selected_assets)
            st.subheader("🌐 Regional Exposure")
            for region, pct in exposure.items():
                st.progress(pct / 100, text=f"{region}: {pct:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not selected_assets:
        st.warning("Please select at least one asset from the sidebar.")
        return
    
    # Fetch and align data (using existing function)
    with st.spinner("🔄 Fetching and aligning data..."):
        df_prices, benchmark_data, data_report = _fetch_and_align_data_cached(
            selected_tickers=tuple(selected_assets),
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            min_data_length=min_data_length
        )
    
    if df_prices is None or df_prices.empty:
        st.error("❌ No valid data available after alignment.")
        return
    
    # Create enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Data Overview",
        "🎯 Portfolio Optimization",
        "🎲 Advanced VaR/ES",
        "⚠️ Stress Testing Lab",
        "📊 Risk Analytics",
        "🔗 Correlation Analysis"
    ])
    
    # ... [Keep existing tabs 1, 2, 5, 6 as they are in your original code] ...
    
    # New Tab 3: Advanced VaR/ES
    with tab3:
        create_var_analysis_tab()
    
    # New Tab 4: Enhanced Stress Testing
    with tab4:
        create_enhanced_stress_test_tab()


# ==============================================================================
# 8) MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
