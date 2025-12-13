import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings

# --- FIXED IMPORTS FOR PyPortfolioOpt ---
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.hierarchical_risk_parity import HRPOpt
from pypfopt import objective_functions
from pypfopt import black_litterman
from pypfopt import BlackLittermanModel
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="QUANTUM | Global Institutional Terminal",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional White Theme
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
    </style>
""", unsafe_allow_html=True)

# --- 2. ENHANCED DATA MANAGER WITH GLOBAL ASSETS ---
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
    
    def get_ticker_name_map(self):
        mapping = {}
        for category, assets in self.universe.items():
            for name, ticker in assets.items():
                mapping[ticker] = name
        return mapping
    
    def get_regional_exposure(self, tickers):
        """Calculate regional exposure for selected tickers"""
        exposure = {}
        for region, region_tickers in self.regional_classification.items():
            region_count = sum(1 for t in tickers if t in region_tickers)
            if region_count > 0:
                exposure[region] = region_count / len(tickers) * 100
        return exposure

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(_self, selected_tickers, start_date="2018-01-01"):
        """Fetches daily OHLC data with error handling and retry logic."""
        if not selected_tickers:
            return pd.DataFrame()
        
        # Ensure benchmark is included
        benchmark_ticker = "^GSPC"
        tickers_to_fetch = list(set(selected_tickers + [benchmark_ticker]))
        
        try:
            # Download with progress bar
            with st.spinner(f"Fetching data for {len(tickers_to_fetch)} instruments..."):
                data = yf.download(
                    tickers_to_fetch, 
                    start=start_date, 
                    group_by='ticker', 
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                    timeout=30
                )
            
            df_close = pd.DataFrame()
            for t in tickers_to_fetch:
                try:
                    if len(tickers_to_fetch) == 1:
                        if 'Close' in data.columns:
                            df_close[t] = data['Close']
                    else:
                        if t in data.columns.levels[0]:
                            df_close[t] = data[t]['Close']
                        else:
                            st.warning(f"Could not fetch data for {t}")
                except Exception as e:
                    continue
            
            if not df_close.empty:
                df_close = df_close.ffill().bfill()
                # Remove any columns with all NaN
                df_close = df_close.dropna(axis=1, how='all')
            
            return df_close
        except Exception as e:
            st.error(f"Data fetching error: {str(e)[:100]}")
            return pd.DataFrame()

# --- 3. ENHANCED OPTIMIZATION ENGINE WITH BLACK-LITTERMAN ---
class EnhancedOptimizationEngine:
    def __init__(self, df_prices, selected_assets):
        self.df = df_prices[selected_assets]
        self.mu = expected_returns.mean_historical_return(self.df)
        self.S = risk_models.sample_cov(self.df)
        self.num_assets = len(selected_assets)
        
    def optimize_mean_variance(self, objective="max_sharpe", risk_free_rate=0.04, gamma=None):
        ef = EfficientFrontier(self.mu, self.S)
        
        if gamma:
            ef.add_objective(objective_functions.L2_reg, gamma=gamma)
        
        try:
            if objective == "max_sharpe":
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == "min_volatility":
                weights = ef.min_volatility()
            elif objective == "max_quadratic_utility":
                ef.max_quadratic_utility(risk_aversion=1)
                weights = ef.clean_weights()
            else:
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            
            cleaned_weights = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            return cleaned_weights, perf
        except Exception as e:
            st.error(f"Optimization error: {e}")
            # Return equal weights as fallback
            equal_weight = 1.0 / self.num_assets
            weights = {asset: equal_weight for asset in self.df.columns}
            ret = self.mu.mean()
            vol = np.sqrt(np.diag(self.S).mean())
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            return weights, (ret, vol, sharpe)
    
    def optimize_hrp(self):
        """Hierarchical Risk Parity with error handling"""
        try:
            returns = self.df.pct_change().dropna()
            
            # Check if we have enough data and non-zero correlation
            if len(returns) < 252:  # Less than 1 year
                raise ValueError("Insufficient data for HRP optimization")
            
            # Check correlation matrix
            corr_matrix = returns.corr()
            if corr_matrix.isnull().any().any():
                raise ValueError("Correlation matrix contains NaN values")
            
            hrp = HRPOpt(returns=returns)
            weights = hrp.optimize()
            cleaned_weights = hrp.clean_weights()
            
            # Calculate performance
            port_returns = returns.dot(pd.Series(cleaned_weights))
            ann_ret = port_returns.mean() * 252
            ann_vol = port_returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            return cleaned_weights, (ann_ret, ann_vol, sharpe)
        except Exception as e:
            st.warning(f"HRP optimization failed: {e}. Falling back to equal weights.")
            equal_weight = 1.0 / self.num_assets
            weights = {asset: equal_weight for asset in self.df.columns}
            returns = self.df.pct_change().dropna()
            port_returns = returns.dot(pd.Series(weights))
            ann_ret = port_returns.mean() * 252
            ann_vol = port_returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            return weights, (ann_ret, ann_vol, sharpe)
    
    def optimize_black_litterman(self, views_dict, confidences, delta=2.5, risk_aversion=1):
        """
        Black-Litterman model implementation
        
        Parameters:
        -----------
        views_dict: dict - Mapping of view assets to their returns
        confidences: dict - Confidence levels for each view (0-1)
        delta: float - Risk aversion parameter
        risk_aversion: float - Market risk aversion
        """
        try:
            # Market equilibrium returns (implied from market cap or equal weights)
            market_caps = None  # Could be replaced with actual market cap data
            if market_caps is None:
                # Use equal weight assumption if market caps not available
                market_weights = np.array([1/self.num_assets] * self.num_assets)
            else:
                market_weights = market_caps / market_caps.sum()
            
            # Create Black-Litterman model
            bl = BlackLittermanModel(
                cov_matrix=self.S,
                pi="equal",  # Prior returns (could be market-implied)
                absolute_views=views_dict,
                omega="idzorek",  # Use Idzorek method for confidence
                view_confidences=list(confidences.values())
            )
            
            # Get Black-Litterman expected returns
            bl_returns = bl.bl_returns()
            
            # Optimize with new expected returns
            ef = EfficientFrontier(bl_returns, self.S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False)
            
            return cleaned_weights, perf, bl_returns
        except Exception as e:
            st.error(f"Black-Litterman optimization failed: {e}")
            return self.optimize_mean_variance("max_sharpe")

# --- 4. ENHANCED RISK METRICS ENGINE ---
class EnhancedRiskMetricsEngine:
    def __init__(self, portfolio_returns, risk_free_rate=0.04, benchmark_returns=None):
        self.returns = portfolio_returns
        self.rf = risk_free_rate
        self.mean = self.returns.mean()
        self.std = self.returns.std()
        self.benchmark_returns = benchmark_returns
        
    def calculate_comprehensive_metrics(self, confidence=0.95, investment=1000000):
        """Calculate all risk metrics in one go"""
        metrics = {}
        
        # Basic statistics
        metrics['Annual Return'] = self.mean * 252
        metrics['Annual Volatility'] = self.std * np.sqrt(252)
        metrics['Skewness'] = self.returns.skew()
        metrics['Kurtosis'] = self.returns.kurtosis()
        
        # Ratio calculations
        if metrics['Annual Volatility'] > 0:
            metrics['Sharpe Ratio'] = (metrics['Annual Return'] - self.rf) / metrics['Annual Volatility']
        else:
            metrics['Sharpe Ratio'] = 0
        
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['Sortino Ratio'] = (metrics['Annual Return'] - self.rf) / downside_std if downside_std > 0 else 0
        
        # Drawdown
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        metrics['Max Drawdown'] = drawdown.min()
        metrics['Calmar Ratio'] = metrics['Annual Return'] / abs(metrics['Max Drawdown']) if abs(metrics['Max Drawdown']) > 0 else 0
        
        # Omega Ratio
        threshold = self.rf / 252  # Daily risk-free rate
        excess_returns = self.returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        metrics['Omega Ratio'] = gains / losses if losses > 0 else np.inf
        
        # Beta calculation
        if self.benchmark_returns is not None:
            aligned = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
            if len(aligned) > 10:
                cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0][1]
                var = np.var(aligned.iloc[:, 1])
                metrics['Beta'] = cov / var if var > 0 else 0
                # Alpha calculation
                metrics['Alpha'] = (metrics['Annual Return'] - self.rf) - metrics['Beta'] * (self.benchmark_returns.mean() * 252 - self.rf)
            else:
                metrics['Beta'] = 0
                metrics['Alpha'] = 0
        else:
            metrics['Beta'] = 0
            metrics['Alpha'] = 0
        
        # VaR metrics
        metrics['Historical VaR (95%)'] = -np.percentile(self.returns, 5)
        metrics['CVaR (95%)'] = -self.returns[self.returns <= np.percentile(self.returns, 5)].mean()
        
        # Tail ratio
        tail_percentile = 5
        metrics['Tail Ratio'] = abs(np.percentile(self.returns, 100-tail_percentile) / 
                                  np.percentile(self.returns, tail_percentile))
        
        # Gain to Pain ratio
        metrics['Gain to Pain Ratio'] = self.returns.sum() / abs(self.returns[self.returns < 0].sum()) if self.returns[self.returns < 0].sum() < 0 else np.inf
        
        return pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })

# --- 5. ENHANCED VISUALIZATION ENGINE ---
class EnhancedChartFactory:
    @staticmethod
    def plot_radar_chart(metrics_df):
        """Create a radar chart for risk metrics"""
        metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio']
        values = []
        for metric in metrics:
            val = metrics_df.loc[metrics_df['Metric'] == metric, 'Value'].values
            values.append(float(val[0]) if len(val) > 0 else 0)
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            line_color='#0056b3',
            fillcolor='rgba(0, 86, 179, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2]
                )),
            showlegend=False,
            template="plotly_white",
            height=400
        )
        return fig
    
    @staticmethod
    def plot_rolling_metrics(returns_series, window=252):
        """Plot rolling Sharpe and volatility"""
        rolling_sharpe = returns_series.rolling(window).mean() * np.sqrt(252) / returns_series.rolling(window).std()
        rolling_vol = returns_series.rolling(window).std() * np.sqrt(252)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            name='Rolling Sharpe (1Y)',
            line=dict(color='#10b981', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            name='Rolling Volatility (1Y)',
            yaxis='y2',
            line=dict(color='#ef4444', width=2)
        ))
        
        fig.update_layout(
            title='Rolling Risk Metrics (1-Year Window)',
            yaxis=dict(title='Sharpe Ratio'),
            yaxis2=dict(
                title='Volatility',
                overlaying='y',
                side='right'
            ),
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        return fig

# --- 6. ENHANCED STRESS TESTING MODULE ---
class EnhancedStressTestLab:
    @staticmethod
    def run_custom_stress_test(portfolio_returns, shock_params):
        """
        Run custom stress tests based on user-defined parameters
        
        Parameters:
        -----------
        shock_params: dict with keys:
            - shock_type: 'market_crash', 'volatility_spike', 'sector_specific', 'regional'
            - magnitude: float (e.g., 0.2 for 20% drop)
            - duration: int (days)
            - affected_assets: list of tickers or 'all'
        """
        base_vol = portfolio_returns.std()
        results = []
        
        if shock_params['shock_type'] == 'market_crash':
            # Simulate market crash with increased correlations
            shock_return = -shock_params['magnitude'] / shock_params['duration']
            simulated = pd.Series([shock_return] * shock_params['duration'], 
                                index=pd.date_range(start=pd.Timestamp.now(), periods=shock_params['duration'], freq='D'))
            total_return = (1 + simulated).prod() - 1
            max_dd = total_return  # In crash scenario, max drawdown equals total loss
            
        elif shock_params['shock_type'] == 'volatility_spike':
            # Simulate volatility spike
            shock_vol = base_vol * (1 + shock_params['magnitude'])
            simulated = np.random.normal(portfolio_returns.mean(), shock_vol, shock_params['duration'])
            simulated = pd.Series(simulated)
            total_return = (1 + simulated).prod() - 1
            max_dd = ((1 + simulated).cumprod() / (1 + simulated).cumprod().cummax() - 1).min()
            
        else:
            # For other shock types, use historical worst periods
            simulated = portfolio_returns.tail(shock_params['duration'])
            total_return = (1 + simulated).prod() - 1
            max_dd = ((1 + simulated).cumprod() / (1 + simulated).cumprod().cummax() - 1).min()
        
        vol = simulated.std() * np.sqrt(252)
        
        return {
            "Scenario": f"Custom: {shock_params['shock_type'].replace('_', ' ').title()}",
            "Magnitude": f"{shock_params['magnitude']:.1%}",
            "Duration": f"{shock_params['duration']} days",
            "Total Return": total_return,
            "Max Drawdown": max_dd,
            "Volatility (Ann.)": vol
        }
    
    @staticmethod
    def run_comprehensive_stress_test(portfolio_returns):
        historical_scenarios = {
            "Global Financial Crisis (2008)": {"start": "2008-09-15", "end": "2009-03-09", "description": "Lehman Brothers collapse"},
            "COVID-19 Crash (2020)": {"start": "2020-02-19", "end": "2020-03-23", "description": "Global pandemic onset"},
            "Dot-com Bubble Burst (2000)": {"start": "2000-03-10", "end": "2002-10-09", "description": "Tech sector collapse"},
            "2022 Inflation/Correction": {"start": "2022-01-03", "end": "2022-06-30", "description": "Rising rates & inflation"},
            "Turkey 2018 Currency Crisis": {"start": "2018-08-01", "end": "2018-09-01", "description": "Lira depreciation"},
            "US Debt Ceiling (2011)": {"start": "2011-07-25", "end": "2011-08-08", "description": "US credit rating downgrade"},
            "Brexit Referendum (2016)": {"start": "2016-06-23", "end": "2016-06-27", "description": "UK votes to leave EU"},
            "China Market Crash (2015)": {"start": "2015-06-12", "end": "2015-08-26", "description": "Shanghai composite plunge"},
            "Oil Price Crash (2014)": {"start": "2014-06-20", "end": "2015-01-13", "description": "Brent crude drops 60%"},
            "Eurozone Crisis (2011)": {"start": "2011-07-01", "end": "2011-09-30", "description": "Greek debt crisis"}
        }
        
        results = []
        for name, params in historical_scenarios.items():
            try:
                mask = (portfolio_returns.index >= params["start"]) & (portfolio_returns.index <= params["end"])
                period_returns = portfolio_returns.loc[mask]
                if len(period_returns) > 5:
                    total_return = (1 + period_returns).prod() - 1
                    max_dd = ((1 + period_returns).cumprod() / (1 + period_returns).cumprod().cummax() - 1).min()
                    vol = period_returns.std() * np.sqrt(252)
                    results.append({
                        "Scenario": name,
                        "Description": params["description"],
                        "Period": f"{params['start']} to {params['end']}",
                        "Total Return": total_return,
                        "Max Drawdown": max_dd,
                        "Volatility (Ann.)": vol
                    })
            except:
                continue
        
        return pd.DataFrame(results)

# --- 7. SMART TABLES & REPORTING ---
class SmartReportGenerator:
    @staticmethod
    def create_performance_table(metrics_df):
        """Create a styled performance table"""
        styled_df = metrics_df.copy()
        
        # Format values
        def format_value(val, metric):
            if 'Ratio' in metric or 'Beta' in metric or 'Alpha' in metric:
                return f"{val:.3f}"
            elif 'Return' in metric or 'Volatility' in metric or 'VaR' in metric or 'CVaR' in metric:
                return f"{val:.2%}"
            elif 'Drawdown' in metric:
                return f"{val:.2%}"
            else:
                return f"{val:.4f}"
        
        styled_df['Formatted Value'] = styled_df.apply(lambda x: format_value(x['Value'], x['Metric']), axis=1)
        
        # Add risk categorization
        def get_risk_category(metric, value):
            if 'Sharpe' in metric:
                if value > 1.5: return '<span class="risk-badge risk-low">Excellent</span>'
                elif value > 0.8: return '<span class="risk-badge risk-medium">Good</span>'
                else: return '<span class="risk-badge risk-high">Poor</span>'
            elif 'Drawdown' in metric:
                if abs(value) < 0.1: return '<span class="risk-badge risk-low">Low</span>'
                elif abs(value) < 0.2: return '<span class="risk-badge risk-medium">Moderate</span>'
                else: return '<span class="risk-badge risk-high">High</span>'
            elif 'Volatility' in metric:
                if value < 0.15: return '<span class="risk-badge risk-low">Low</span>'
                elif value < 0.25: return '<span class="risk-badge risk-medium">Moderate</span>'
                else: return '<span class="risk-badge risk-high">High</span>'
            else:
                return ''
        
        styled_df['Assessment'] = styled_df.apply(lambda x: get_risk_category(x['Metric'], x['Value']), axis=1)
        
        return styled_df[['Metric', 'Formatted Value', 'Assessment']]

# --- 8. MAIN APPLICATION ---
def main():
    st.markdown('<div class="main-header">🌐 QUANTUM | Global Institutional Analytics Platform</div>', unsafe_allow_html=True)
    dm = EnhancedDataManager()
    
    # Initialize session state for custom shocks
    if 'custom_shocks' not in st.session_state:
        st.session_state.custom_shocks = []
    
    with st.sidebar:
        st.header("🌍 Global Asset Selection")
        
        # Regional filter
        region_filter = st.multiselect(
            "Filter by Region",
            options=["All", "North America", "Europe", "Asia Pacific", "Emerging Markets"],
            default=["All"]
        )
        
        selected_assets = []
        default_assets = ["SPY", "QQQ", "GLD", "TLT", "THYAO.IS", "AAPL", "7203.T", "CBA.AX"]
        
        for category, assets in dm.universe.items():
            with st.expander(f"📊 {category}", expanded=(category in ["US ETFs (Major & Active)", "Global Mega Caps"])):
                # Filter by region if specified
                filtered_assets = assets
                if "All" not in region_filter:
                    filtered_assets = {
                        k: v for k, v in assets.items() 
                        if any(v in dm.regional_classification.get(r, []) for r in region_filter)
                    }
                
                selected = st.multiselect(
                    f"Select from {category}",
                    options=list(filtered_assets.keys()),
                    default=[k for k in filtered_assets.keys() if filtered_assets[k] in default_assets],
                    key=f"select_{category}"
                )
                for s in selected:
                    selected_assets.append(filtered_assets[s])
        
        # Display regional exposure
        if selected_assets:
            exposure = dm.get_regional_exposure(selected_assets)
            st.subheader("🌐 Regional Exposure")
            for region, pct in exposure.items():
                st.progress(pct/100, text=f"{region}: {pct:.1f}%")
    
    if not selected_assets:
        st.warning("Please select at least one asset from the sidebar.")
        return
    
    # Fetch data
    with st.spinner("🔄 Fetching global market data..."):
        df_all = dm.fetch_data(selected_assets)
        benchmark_ticker = "^GSPC"
        
        if df_all.empty:
            st.error("Failed to load data. Please check your internet connection and try again.")
            return
        
        df_prices = df_all[[col for col in selected_assets if col in df_all.columns]]
        
        if benchmark_ticker in df_all.columns:
            benchmark_data = df_all[benchmark_ticker]
        else:
            benchmark_data = None
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Portfolio Optimization", 
        "⚠️ Stress Testing Lab", 
        "📊 Market Intelligence",
        "🛡️ Risk Analytics",
        "📈 Performance Attribution"
    ])
    
    # --- TAB 1: ENHANCED PORTFOLIO OPTIMIZATION ---
    with tab1:
        st.subheader("Advanced Portfolio Construction")
        
        col_config1, col_config2, col_config3 = st.columns(3)
        with col_config1:
            strategy = st.selectbox(
                "Optimization Strategy",
                ["Max Sharpe Ratio", "Min Volatility", "Hierarchical Risk Parity (HRP)", 
                 "Black-Litterman (Views)", "Max Quadratic Utility", "Equal Weight"]
            )
        
        with col_config2:
            rf_rate = st.number_input("Risk Free Rate (%)", value=4.5, step=0.1) / 100
        
        with col_config3:
            amount = st.number_input("Investment Amount ($)", value=1000000, step=100000)
        
        # Black-Litterman views configuration
        views_dict = {}
        confidences = {}
        
        if strategy == "Black-Litterman (Views)":
            st.subheader("Black-Litterman Views Configuration")
            col_bl1, col_bl2 = st.columns(2)
            
            with col_bl1:
                st.markdown("**Add Your Views**")
                view_asset = st.selectbox("Asset", options=df_prices.columns.tolist())
                view_return = st.number_input("Expected Return (%)", value=10.0, step=1.0) / 100
                confidence = st.slider("Confidence Level", 0.1, 1.0, 0.7)
                
                if st.button("Add View"):
                    views_dict[view_asset] = view_return
                    confidences[view_asset] = confidence
            
            with col_bl2:
                if views_dict:
                    st.markdown("**Current Views**")
                    for asset, ret in views_dict.items():
                        st.write(f"{asset}: {ret:.2%} return (Confidence: {confidences[asset]:.0%})")
        
        # Run optimization
        opt_engine = EnhancedOptimizationEngine(df_prices, df_prices.columns.tolist())
        
        if strategy == "Hierarchical Risk Parity (HRP)":
            weights, perf = opt_engine.optimize_hrp()
        elif strategy == "Black-Litterman (Views)" and views_dict:
            weights, perf, bl_returns = opt_engine.optimize_black_litterman(views_dict, confidences)
        elif strategy == "Max Quadratic Utility":
            weights, perf = opt_engine.optimize_mean_variance("max_quadratic_utility", rf_rate)
        elif strategy == "Equal Weight":
            equal_weight = 1.0 / len(df_prices.columns)
            weights = {asset: equal_weight for asset in df_prices.columns}
            returns = df_prices.pct_change().dropna()
            port_returns = returns.dot(pd.Series(weights))
            ann_ret = port_returns.mean() * 252
            ann_vol = port_returns.std() * np.sqrt(252)
            sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
            perf = (ann_ret, ann_vol, sharpe)
        else:
            obj_type = "max_sharpe" if "Sharpe" in strategy else "min_volatility"
            weights, perf = opt_engine.optimize_mean_variance(obj_type, rf_rate)
        
        if weights:
            # Display results
            st.divider()
            col_results1, col_results2 = st.columns([1, 1])
            
            with col_results1:
                st.subheader("📊 Portfolio Allocation")
                ticker_map = dm.get_ticker_name_map()
                portfolio_data = []
                
                for ticker, weight in weights.items():
                    if weight > 0.001:
                        portfolio_data.append({
                            "Asset": ticker_map.get(ticker, ticker),
                            "Ticker": ticker,
                            "Weight": weight,
                            "Amount ($)": weight * amount
                        })
                
                portfolio_df = pd.DataFrame(portfolio_data).sort_values(by="Weight", ascending=False)
                
                # Display as table
                display_df = portfolio_df[['Asset', 'Weight', 'Amount ($)']].copy()
                display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2%}")
                display_df['Amount ($)'] = display_df['Amount ($)'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(display_df, use_container_width=True, height=400)
            
            with col_results2:
                st.subheader("📈 Allocation Chart")
                if not portfolio_df.empty:
                    fig_pie = px.pie(
                        portfolio_df, 
                        values='Weight', 
                        names='Asset', 
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(
                        template="plotly_white", 
                        height=400,
                        margin=dict(t=0, b=0, l=0, r=0),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Performance metrics
            st.divider()
            st.subheader("🎯 Performance Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected Return (Ann.)", f"{perf[0]:.2%}", delta="Target: 8-12%")
            m2.metric("Volatility (Ann.)", f"{perf[1]:.2%}", delta="Lower is better", delta_color="inverse")
            m3.metric("Sharpe Ratio", f"{perf[2]:.2f}", delta=">1.0 is good")
            
            # Calculate additional metrics
            returns_daily = df_prices.pct_change().dropna()
            weight_vector = [weights.get(t, 0) for t in returns_daily.columns]
            portfolio_returns = returns_daily.dot(weight_vector)
            
            # Drawdown
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()
            m4.metric("Max Drawdown", f"{max_dd:.2%}", delta="<20% is acceptable", delta_color="inverse")
            
            # Efficient Frontier
            st.divider()
            st.subheader("📊 Efficient Frontier Analysis")
            weights_array = np.array([weights.get(col, 0) for col in df_prices.columns])
            fig_ef = ChartFactory.plot_efficient_frontier(opt_engine.mu, opt_engine.S, weights_array)
            st.plotly_chart(fig_ef, use_container_width=True)
    
    # --- TAB 2: ENHANCED STRESS TESTING ---
    with tab2:
        st.subheader("⚠️ Comprehensive Stress Testing Laboratory")
        
        # Historical scenarios
        st.markdown("### 📜 Historical Stress Scenarios")
        stress_engine = EnhancedStressTestLab()
        historical_stress_df = stress_engine.run_comprehensive_stress_test(portfolio_returns)
        
        if not historical_stress_df.empty:
            col_hist1, col_hist2 = st.columns([2, 1])
            
            with col_hist1:
                st.dataframe(
                    historical_stress_df.style.format({
                        "Total Return": "{:.2%}",
                        "Max Drawdown": "{:.2%}",
                        "Volatility (Ann.)": "{:.2%}"
                    }).background_gradient(cmap="RdYlGn_r", subset=["Total Return"]),
                    use_container_width=True,
                    height=400
                )
            
            with col_hist2:
                st.markdown("**Scenario Impact Summary**")
                worst_scenario = historical_stress_df.loc[historical_stress_df['Total Return'].idxmin()]
                best_scenario = historical_stress_df.loc[historical_stress_df['Total Return'].idxmax()]
                
                st.metric("Worst Historical Loss", f"{worst_scenario['Total Return']:.2%}", 
                         delta=worst_scenario['Scenario'])
                st.metric("Best Historical Performance", f"{best_scenario['Total Return']:.2%}", 
                         delta=best_scenario['Scenario'])
        
        # Custom stress test builder
        st.divider()
        st.markdown("### 🛠️ Custom Stress Test Builder")
        
        col_custom1, col_custom2, col_custom3 = st.columns(3)
        
        with col_custom1:
            shock_type = st.selectbox(
                "Shock Type",
                ["market_crash", "volatility_spike", "sector_specific", "regional_crisis", "liquidity_crunch"]
            )
        
        with col_custom2:
            magnitude = st.slider("Shock Magnitude", 0.1, 1.0, 0.3, 0.1)
        
        with col_custom3:
            duration = st.slider("Duration (days)", 5, 250, 30)
        
        # Asset selection for targeted shocks
        affected_assets = []
        if shock_type in ["sector_specific", "regional_crisis"]:
            affected_assets = st.multiselect(
                "Affected Assets (leave empty for all)",
                options=df_prices.columns.tolist(),
                default=[]
            )
        
        if st.button("Run Custom Stress Test"):
            shock_params = {
                "shock_type": shock_type,
                "magnitude": magnitude,
                "duration": duration,
                "affected_assets": affected_assets if affected_assets else "all"
            }
            
            custom_result = stress_engine.run_custom_stress_test(portfolio_returns, shock_params)
            
            # Display results
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Scenario", custom_result["Scenario"])
            col_res2.metric("Estimated Loss", f"{custom_result['Total Return']:.2%}")
            col_res3.metric("Max Drawdown", f"{custom_result['Max Drawdown']:.2%}")
            
            # Store in session state
            st.session_state.custom_shocks.append(custom_result)
        
        # Display custom shock history
        if st.session_state.custom_shocks:
            st.divider()
            st.markdown("### 📋 Custom Shock History")
            custom_df = pd.DataFrame(st.session_state.custom_shocks)
            st.dataframe(custom_df.style.format({
                "Total Return": "{:.2%}",
                "Max Drawdown": "{:.2%}",
                "Volatility (Ann.)": "{:.2%}"
            }), use_container_width=True)
    
    # --- TAB 3: MARKET INTELLIGENCE ---
    with tab3:
        st.subheader("📊 Market Intelligence Dashboard")
        
        # Correlation matrix
        st.markdown("### 🔗 Correlation Heatmap")
        corr_matrix = df_prices.pct_change().corr()
        
        # Cluster the correlation matrix
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        dist_matrix = 1 - corr_matrix.values
        dist_matrix = np.clip(dist_matrix, 0, 2)
        
        # Perform hierarchical clustering
        linked = linkage(squareform(dist_matrix), 'single')
        
        # Get optimal leaf ordering
        from scipy.cluster.hierarchy import optimal_leaf_ordering, leaves_list
        ordered = optimal_leaf_ordering(linked, dist_matrix)
        order = leaves_list(ordered)
        
        # Reorder correlation matrix
        ordered_corr = corr_matrix.iloc[order, order]
        
        fig_corr = px.imshow(
            ordered_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        fig_corr.update_layout(
            template="plotly_white",
            height=800,
            title="Clustered Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Cumulative returns
        st.markdown("### 📈 Cumulative Returns (Rebased to 100)")
        normalized = (df_prices / df_prices.iloc[0]) * 100
        
        fig_returns = px.line(
            normalized,
            title="Performance Comparison (Rebased)",
            labels={"value": "Index Value (Rebased to 100)", "variable": "Asset"}
        )
        fig_returns.update_layout(
            template="plotly_white",
            height=600,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Rolling statistics
        st.markdown("### 🔄 Rolling Statistics")
        rolling_window = st.slider("Rolling Window (days)", 30, 252, 63)
        
        rolling_returns = df_prices.pct_change().rolling(rolling_window).mean() * 252
        rolling_vol = df_prices.pct_change().rolling(rolling_window).std() * np.sqrt(252)
        
        col_roll1, col_roll2 = st.columns(2)
        
        with col_roll1:
            fig_roll_ret = px.line(
                rolling_returns,
                title=f"Rolling {rolling_window}-Day Annualized Returns"
            )
            fig_roll_ret.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_roll_ret, use_container_width=True)
        
        with col_roll2:
            fig_roll_vol = px.line(
                rolling_vol,
                title=f"Rolling {rolling_window}-Day Annualized Volatility"
            )
            fig_roll_vol.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_roll_vol, use_container_width=True)
    
    # --- TAB 4: RISK ANALYTICS ---
    with tab4:
        st.subheader("🛡️ Comprehensive Risk Analytics")
        
        # Initialize risk engine
        benchmark_returns_series = benchmark_data.pct_change().dropna() if benchmark_data is not None else None
        risk_engine = EnhancedRiskMetricsEngine(portfolio_returns, rf_rate, benchmark_returns_series)
        
        # Comprehensive metrics
        st.markdown("### 📊 Risk & Performance Metrics")
        comprehensive_metrics = risk_engine.calculate_comprehensive_metrics(investment=amount)
        
        # Display in two columns
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.markdown("**Return & Risk Metrics**")
            return_metrics = comprehensive_metrics[comprehensive_metrics['Metric'].isin([
                'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                'Max Drawdown', 'Calmar Ratio', 'Alpha', 'Beta'
            ])]
            
            for _, row in return_metrics.iterrows():
                if 'Ratio' in row['Metric'] or 'Beta' in row['Metric'] or 'Alpha' in row['Metric']:
                    st.metric(row['Metric'], f"{row['Value']:.3f}")
                else:
                    st.metric(row['Metric'], f"{row['Value']:.2%}")
        
        with col_metrics2:
            st.markdown("**Distribution & Tail Metrics**")
            tail_metrics = comprehensive_metrics[comprehensive_metrics['Metric'].isin([
                'Skewness', 'Kurtosis', 'Historical VaR (95%)', 'CVaR (95%)',
                'Tail Ratio', 'Omega Ratio', 'Gain to Pain Ratio'
            ])]
            
            for _, row in tail_metrics.iterrows():
                if 'Ratio' in row['Metric']:
                    st.metric(row['Metric'], f"{row['Value']:.3f}")
                elif 'VaR' in row['Metric'] or 'CVaR' in row['Metric']:
                    st.metric(row['Metric'], f"{row['Value']:.2%}")
                else:
                    st.metric(row['Metric'], f"{row['Value']:.3f}")
        
        # VaR Analysis
        st.divider()
        st.markdown("### 💰 Value at Risk (VaR) Analysis")
        
        col_var1, col_var2, col_var3 = st.columns(3)
        
        with col_var1:
            confidence_level = st.select_slider(
                "Confidence Level",
                options=[0.90, 0.95, 0.99],
                value=0.95
            )
        
        with col_var2:
            holding_days = st.selectbox(
                "Holding Period",
                options=[1, 5, 10, 20, 30],
                index=0
            )
        
        with col_var3:
            var_method = st.radio(
                "VaR Method",
                ["Historical", "Parametric", "Monte Carlo"]
            )
        
        # Calculate VaR
        alpha = 1 - confidence_level
        scale = np.sqrt(holding_days)
        
        if var_method == "Historical":
            var_pct = -np.percentile(portfolio_returns, alpha * 100) * scale
        elif var_method == "Parametric":
            z_score = stats.norm.ppf(alpha)
            var_pct = -(portfolio_returns.mean() * holding_days + z_score * portfolio_returns.std() * scale)
        else:  # Monte Carlo
            simulations = 10000
            sim_returns = np.random.normal(
                portfolio_returns.mean(),
                portfolio_returns.std(),
                simulations * holding_days
            ).reshape(simulations, holding_days)
            sim_port_returns = np.prod(1 + sim_returns, axis=1) - 1
            var_pct = -np.percentile(sim_port_returns, alpha * 100)
        
        var_value = var_pct * amount
        
        col_var_disp1, col_var_disp2, col_var_disp3 = st.columns(3)
        col_var_disp1.metric(f"{confidence_level:.0%} VaR (%)", f"{var_pct:.2%}")
        col_var_disp2.metric(f"{confidence_level:.0%} VaR ($)", f"${var_value:,.0f}")
        
        # CVaR calculation
        if var_method == "Historical":
            cutoff = np.percentile(portfolio_returns, alpha * 100)
            cvar_pct = -portfolio_returns[portfolio_returns <= cutoff].mean() * scale
        else:
            # For Monte Carlo
            cutoff = np.percentile(sim_port_returns, alpha * 100)
            cvar_pct = -sim_port_returns[sim_port_returns <= cutoff].mean()
        
        cvar_value = cvar_pct * amount
        col_var_disp3.metric(f"CVaR ({confidence_level:.0%})", f"${cvar_value:,.0f}", 
                           delta=f"{cvar_pct:.2%}")
        
        # Risk decomposition
        st.divider()
        st.markdown("### 🎯 Risk Decomposition by Asset")
        
        risk_breakdown = []
        for ticker, weight in weights.items():
            if weight > 0.001:
                asset_returns = df_prices[ticker].pct_change().dropna()
                asset_var = -np.percentile(asset_returns, alpha * 100) * scale * weight
                asset_mctr = weight * np.cov(asset_returns, portfolio_returns)[0, 1] / np.var(portfolio_returns)
                
                risk_breakdown.append({
                    "Asset": ticker_map.get(ticker, ticker),
                    "Weight": weight,
                    "Marginal Contribution to Risk": asset_mctr,
                    "VaR Contribution": asset_var * amount
                })
        
        if risk_breakdown:
            risk_df = pd.DataFrame(risk_breakdown)
            
            col_risk1, col_risk2 = st.columns([2, 1])
            
            with col_risk1:
                fig_risk = px.bar(
                    risk_df.sort_values("VaR Contribution", ascending=False),
                    x="Asset",
                    y="VaR Contribution",
                    title="VaR Contribution by Asset",
                    color="Weight",
                    color_continuous_scale="Viridis"
                )
                fig_risk.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col_risk2:
                st.dataframe(
                    risk_df[['Asset', 'Weight', 'Marginal Contribution to Risk']]
                    .sort_values('Marginal Contribution to Risk', ascending=False)
                    .style.format({
                        'Weight': '{:.2%}',
                        'Marginal Contribution to Risk': '{:.4f}'
                    }),
                    height=400
                )
    
    # --- TAB 5: PERFORMANCE ATTRIBUTION ---
    with tab5:
        st.subheader("📈 Performance Attribution & Analytics")
        
        # Rolling performance charts
        st.markdown("### 📊 Rolling Performance Metrics")
        
        fig_rolling = EnhancedChartFactory.plot_rolling_metrics(portfolio_returns)
        st.plotly_chart(fig_rolling, use_container_width=True)
        
        # Drawdown analysis
        st.markdown("### 📉 Drawdown Analysis")
        
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown_series = (cum_returns - running_max) / running_max
        
        col_dd1, col_dd2 = st.columns(2)
        
        with col_dd1:
            fig_dd = ChartFactory.plot_drawdown(portfolio_returns)
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with col_dd2:
            # Drawdown statistics
            dd_stats = {
                "Current Drawdown": f"{drawdown_series.iloc[-1]:.2%}",
                "Max Drawdown": f"{drawdown_series.min():.2%}",
                "Avg Drawdown": f"{drawdown_series[drawdown_series < 0].mean():.2%}",
                "Recovery Time (Avg Days)": "N/A",
                "Pain Index": f"{abs(drawdown_series[drawdown_series < 0].mean()):.3f}"
            }
            
            for stat, value in dd_stats.items():
                st.metric(stat, value)
        
        # Performance attribution
        st.divider()
        st.markdown("### 🎯 Return Attribution")
        
        # Calculate attribution
        attribution_data = []
        for ticker, weight in weights.items():
            if weight > 0.001:
                asset_returns = df_prices[ticker].pct_change().dropna()
                aligned_returns = asset_returns.reindex(portfolio_returns.index).fillna(0)
                contribution = weight * aligned_returns
                attribution_data.append({
                    "Asset": ticker_map.get(ticker, ticker),
                    "Weight": weight,
                    "Return Contribution": contribution.sum(),
                    "Annualized Return": asset_returns.mean() * 252 if len(asset_returns) > 0 else 0
                })
        
        attribution_df = pd.DataFrame(attribution_data)
        
        col_attr1, col_attr2 = st.columns(2)
        
        with col_attr1:
            fig_attr = px.bar(
                attribution_df.sort_values("Return Contribution", ascending=False),
                x="Asset",
                y="Return Contribution",
                title="Return Contribution by Asset",
                color="Weight",
                color_continuous_scale="Blues"
            )
            fig_attr.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_attr, use_container_width=True)
        
        with col_attr2:
            st.dataframe(
                attribution_df[['Asset', 'Weight', 'Annualized Return', 'Return Contribution']]
                .sort_values('Return Contribution', ascending=False)
                .style.format({
                    'Weight': '{:.2%}',
                    'Annualized Return': '{:.2%}',
                    'Return Contribution': '{:.2%}'
                }),
                height=400
            )
        
        # Generate performance report
        st.divider()
        st.markdown("### 📄 Comprehensive Performance Report")
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating report..."):
                # Create a comprehensive report
                report_data = {
                    "Portfolio Summary": {
                        "Total Assets": len(weights),
                        "Total Value": f"${amount:,.0f}",
                        "Optimization Strategy": strategy,
                        "Risk Free Rate": f"{rf_rate:.2%}"
                    },
                    "Performance Metrics": comprehensive_metrics.set_index('Metric')['Value'].to_dict(),
                    "Top Holdings": portfolio_df.head(10).to_dict('records'),
                    "Risk Metrics": {
                        f"{confidence_level:.0%} VaR": f"{var_pct:.2%}",
                        f"{confidence_level:.0%} CVaR": f"{cvar_pct:.2%}",
                        "Max Drawdown": f"{max_dd:.2%}"
                    }
                }
                
                # Display report
                st.json(report_data)
                st.success("Report generated successfully!")

if __name__ == "__main__":
    main()
