import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION & STYLING ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND
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

# --- TRY TO IMPORT OPTIMIZATION LIBRARIES WITH FALLBACK ---
try:
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt import objective_functions
    
    # Try to import hierarchical_risk_parity
    try:
        from pypfopt.hierarchical_risk_parity import HRPOpt
        HRP_AVAILABLE = True
    except ImportError:
        HRP_AVAILABLE = False
        st.warning("⚠️ HRP optimization not available. Install latest PyPortfolioOpt for full features.")
    
    # Try to import BlackLitterman
    try:
        from pypfopt.black_litterman import BlackLittermanModel
        BLACK_LITTERMAN_AVAILABLE = True
    except ImportError:
        try:
            from pypfopt import BlackLittermanModel
            BLACK_LITTERMAN_AVAILABLE = True
        except ImportError:
            BLACK_LITTERMAN_AVAILABLE = False
    
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ PyPortfolioOpt not fully available: {e}")
    OPTIMIZATION_AVAILABLE = False
    HRP_AVAILABLE = False
    BLACK_LITTERMAN_AVAILABLE = False

# --- 2. ENHANCED DATA MANAGER WITH GLOBAL ASSETS & ROBUST DATA ALIGNMENT ---
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
        total = len(tickers)
        if total == 0:
            return exposure
            
        for region, region_tickers in self.regional_classification.items():
            region_count = sum(1 for t in tickers if t in region_tickers)
            if region_count > 0:
                exposure[region] = region_count / total * 100
        return exposure

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_and_align_data(_self, selected_tickers: List[str], 
                           start_date: str = "2018-01-01",
                           end_date: Optional[str] = None,
                           min_data_length: int = 100) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Robust data fetching with proper alignment and forward-filling.
        
        Returns:
            df_prices: Cleaned price DataFrame with aligned dates
            benchmark_series: S&P 500 benchmark data (if available)
            data_quality_report: Dictionary with data quality metrics
        """
        if not selected_tickers:
            return pd.DataFrame(), pd.Series(), {}
        
        # Always include S&P 500 as benchmark
        benchmark_ticker = "^GSPC"
        all_tickers = list(set(selected_tickers + [benchmark_ticker]))
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Download data
            data = yf.download(
                all_tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                threads=True,
                progress=False,
                timeout=30
            )
            
            # Extract closing prices
            prices_dict = {}
            data_quality = {}
            
            for ticker in all_tickers:
                try:
                    if len(all_tickers) == 1:
                        if 'Close' in data.columns:
                            prices_dict[ticker] = data['Close']
                    else:
                        if ticker in data.columns.levels[0]:
                            prices_dict[ticker] = data[ticker]['Close']
                    
                    # Store initial data quality metrics
                    if ticker in prices_dict:
                        series = prices_dict[ticker]
                        original_length = len(series)
                        non_na_count = series.count()
                        na_count = original_length - non_na_count
                        na_percentage = (na_count / original_length * 100) if original_length > 0 else 100
                        
                        data_quality[ticker] = {
                            "original_length": original_length,
                            "non_na_count": non_na_count,
                            "na_count": na_count,
                            "na_percentage": na_percentage,
                            "start_date": series.index.min() if non_na_count > 0 else None,
                            "end_date": series.index.max() if non_na_count > 0 else None,
                            "data_days": (series.index.max() - series.index.min()).days if non_na_count > 1 else 0
                        }
                        
                except Exception as e:
                    st.warning(f"Could not extract data for {ticker}: {str(e)[:100]}")
                    continue
            
            if not prices_dict:
                st.error("No data could be fetched for any ticker.")
                return pd.DataFrame(), pd.Series(), {}
            
            # Create DataFrame from dictionary
            df_raw = pd.DataFrame(prices_dict)
            
            # Step 1: Remove duplicate column names (FIX FOR NARWHALS ERROR)
            # This happens when the same ticker appears multiple times
            seen = set()
            unique_cols = []
            for col in df_raw.columns:
                if col not in seen:
                    seen.add(col)
                    unique_cols.append(col)
                else:
                    # Handle duplicate columns
                    st.warning(f"Removing duplicate column: {col}")
            
            df_raw = df_raw[unique_cols]
            
            # Step 2: Remove tickers with insufficient data
            valid_tickers = []
            for ticker in df_raw.columns:
                non_na_count = df_raw[ticker].count()
                if non_na_count >= min_data_length:
                    valid_tickers.append(ticker)
                else:
                    st.warning(f"Removing {ticker}: insufficient data ({non_na_count} points)")
            
            if not valid_tickers:
                st.error("No tickers have sufficient data.")
                return pd.DataFrame(), pd.Series(), {}
            
            df_filtered = df_raw[valid_tickers]
            
            # Step 3: Forward fill then backward fill to handle missing values
            df_filled = df_filtered.ffill().bfill()
            
            # Step 4: Drop any remaining rows with NA
            df_clean = df_filled.dropna()
            
            # Step 5: Verify we have enough data after cleaning
            if len(df_clean) < min_data_length:
                st.error(f"Insufficient data after cleaning. Only {len(df_clean)} data points available.")
                return pd.DataFrame(), pd.Series(), {}
            
            # Step 6: Separate benchmark from portfolio assets
            portfolio_tickers = [t for t in selected_tickers if t in df_clean.columns]
            benchmark_data = pd.Series()
            
            if benchmark_ticker in df_clean.columns:
                benchmark_data = df_clean[benchmark_ticker]
                # Remove benchmark from portfolio if not explicitly selected
                if benchmark_ticker not in selected_tickers:
                    df_portfolio = df_clean[portfolio_tickers]
                else:
                    df_portfolio = df_clean[portfolio_tickers + [benchmark_ticker]]
            else:
                df_portfolio = df_clean[portfolio_tickers]
                st.warning("Benchmark data not available for comparison.")
            
            # Final data quality report
            final_report = {
                "initial_tickers": len(all_tickers),
                "valid_tickers": len(valid_tickers),
                "final_tickers": len(df_portfolio.columns),
                "total_rows": len(df_clean),
                "start_date": df_clean.index.min().strftime("%Y-%m-%d"),
                "end_date": df_clean.index.max().strftime("%Y-%m-%d"),
                "data_range_days": (df_clean.index.max() - df_clean.index.min()).days,
                "ticker_details": data_quality,
                "alignment_status": "SUCCESS" if len(df_clean) >= min_data_length else "FAILED",
                "missing_data_summary": {
                    ticker: f"{data_quality.get(ticker, {}).get('na_percentage', 100):.1f}%" 
                    for ticker in valid_tickers if ticker in data_quality
                }
            }
            
            st.success(f"✅ Data alignment complete. Final dataset: {len(df_clean)} rows × {len(df_portfolio.columns)} assets")
            return df_portfolio, benchmark_data, final_report
            
        except Exception as e:
            st.error(f"Data fetching error: {str(e)[:200]}")
            return pd.DataFrame(), pd.Series(), {}
    
    def display_data_quality_report(self, report: Dict):
        """Display comprehensive data quality report."""
        if not report:
            return
        
        with st.expander("📊 Data Quality & Alignment Report", expanded=True):
            st.markdown('<div class="data-success">✅ Data Alignment Successful</div>', unsafe_allow_html=True)
            
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
            
            # Show data completeness table
            st.subheader("📈 Data Completeness by Asset")
            if "missing_data_summary" in report:
                completeness_data = []
                for ticker, missing_pct in report["missing_data_summary"].items():
                    details = report.get("ticker_details", {}).get(ticker, {})
                    completeness_data.append({
                        "Ticker": ticker,
                        "Asset Name": self.get_ticker_name_map().get(ticker, ticker),
                        "Data Points": details.get("non_na_count", 0),
                        "Missing %": float(missing_pct.replace("%", "")),
                        "Start Date": details.get("start_date", "N/A"),
                        "End Date": details.get("end_date", "N/A")
                    })
                
                if completeness_data:
                    df_completeness = pd.DataFrame(completeness_data)
                    df_completeness = df_completeness.sort_values("Missing %", ascending=True)
                    
                    # Add color coding for missing data
                    def color_missing(val):
                        if val < 5:
                            return 'background-color: #d1fae5; color: #065f46'
                        elif val < 20:
                            return 'background-color: #fef3c7; color: #92400e'
                        else:
                            return 'background-color: #fee2e2; color: #991b1b'
                    
                    styled_df = df_completeness.style.format({
                        'Data Points': '{:,}',
                        'Missing %': '{:.1f}%'
                    }).applymap(color_missing, subset=['Missing %'])
                    
                    st.dataframe(styled_df, use_container_width=True, height=300)

# --- 3. ENHANCED OPTIMIZATION ENGINE WITH DATA VALIDATION ---
class EnhancedOptimizationEngine:
    def __init__(self, df_prices: pd.DataFrame):
        """
        Initialize with validated price data.
        
        Args:
            df_prices: DataFrame with aligned price data (no NaN, equal lengths)
        """
        if df_prices.empty:
            raise ValueError("Empty price DataFrame provided")
        
        # Validate data before optimization
        self._validate_data(df_prices)
        
        self.df = df_prices
        self.returns = df_prices.pct_change().dropna()
        if OPTIMIZATION_AVAILABLE:
            try:
                self.mu = expected_returns.mean_historical_return(self.df)
                self.S = risk_models.sample_cov(self.df)
            except:
                self.mu = None
                self.S = None
        else:
            self.mu = None
            self.S = None
        self.num_assets = len(df_prices.columns)
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate that data is suitable for optimization."""
        # Check for NaN
        if df.isnull().any().any():
            nan_count = df.isnull().sum().sum()
            raise ValueError(f"Data contains {nan_count} NaN values")
        
        # Check for sufficient length
        if len(df) < 100:
            raise ValueError(f"Insufficient data points: {len(df)}. Need at least 100.")
        
        # Check for zero or negative prices
        if (df <= 0).any().any():
            negative_count = (df <= 0).sum().sum()
            raise ValueError(f"Data contains {negative_count} zero or negative prices")
        
        # Check asset length consistency
        lengths = [len(df[col].dropna()) for col in df.columns]
        if max(lengths) - min(lengths) > 10:  # Allow small differences
            st.warning(f"Asset lengths vary: min={min(lengths)}, max={max(lengths)}")
    
    def optimize_mean_variance(self, objective="max_sharpe", risk_free_rate=0.04, gamma=None):
        """Robust mean-variance optimization."""
        if not OPTIMIZATION_AVAILABLE or self.mu is None or self.S is None:
            st.error("Optimization not available. Install PyPortfolioOpt.")
            return None, None
            
        try:
            ef = EfficientFrontier(self.mu, self.S)
            
            if gamma:
                ef.add_objective(objective_functions.L2_reg, gamma=gamma)
            
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
            # Fallback to equal weights
            equal_weight = 1.0 / self.num_assets
            weights = {asset: equal_weight for asset in self.df.columns}
            ret = self.returns.mean().mean() * 252 if self.mu is None else self.mu.mean()
            vol = self.returns.std().mean() * np.sqrt(252) if self.S is None else np.sqrt(np.diag(self.S).mean())
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            return weights, (ret, vol, sharpe)
    
    def optimize_hrp(self):
        """Hierarchical Risk Parity optimization with error handling."""
        if not OPTIMIZATION_AVAILABLE or not HRP_AVAILABLE:
            st.error("HRP optimization not available. Install latest PyPortfolioOpt.")
            return None, None
            
        try:
            # Ensure we have enough data
            if len(self.returns) < 100:
                raise ValueError(f"Insufficient returns data: {len(self.returns)} points")
            
            hrp = HRPOpt(returns=self.returns)
            weights = hrp.optimize()
            cleaned_weights = hrp.clean_weights()
            
            # Calculate performance
            port_returns = self.returns.dot(pd.Series(cleaned_weights))
            ann_ret = port_returns.mean() * 252
            ann_vol = port_returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            return cleaned_weights, (ann_ret, ann_vol, sharpe)
            
        except Exception as e:
            st.warning(f"HRP optimization failed: {e}. Using equal weights.")
            equal_weight = 1.0 / self.num_assets
            weights = {asset: equal_weight for asset in self.df.columns}
            port_returns = self.returns.dot(pd.Series(weights))
            ann_ret = port_returns.mean() * 252
            ann_vol = port_returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            return weights, (ann_ret, ann_vol, sharpe)
    
    def optimize_black_litterman(self, views_dict, confidences, delta=2.5, risk_aversion=1):
        """Black-Litterman optimization (if available)."""
        if not BLACK_LITTERMAN_AVAILABLE:
            st.error("Black-Litterman model not available. Install latest PyPortfolioOpt.")
            return self.optimize_mean_variance("max_sharpe")
        
        try:
            # Create market equilibrium weights (equal weight assumption)
            market_weights = np.array([1/self.num_assets] * self.num_assets)
            
            # Create Black-Litterman model
            bl = BlackLittermanModel(
                cov_matrix=self.S,
                pi="equal",
                absolute_views=views_dict,
                omega="idzorek",
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
    def __init__(self, portfolio_returns: pd.Series, risk_free_rate: float = 0.04, 
                 benchmark_returns: Optional[pd.Series] = None):
        self.returns = portfolio_returns
        self.rf = risk_free_rate
        self.benchmark_returns = benchmark_returns
    
    def calculate_comprehensive_metrics(self, investment: float = 1000000) -> pd.DataFrame:
        """Calculate comprehensive risk and performance metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['Annual Return'] = self.returns.mean() * 252
        metrics['Annual Volatility'] = self.returns.std() * np.sqrt(252)
        metrics['Skewness'] = self.returns.skew()
        metrics['Kurtosis'] = self.returns.kurtosis()
        
        # Ratios
        if metrics['Annual Volatility'] > 0:
            metrics['Sharpe Ratio'] = (metrics['Annual Return'] - self.rf) / metrics['Annual Volatility']
        else:
            metrics['Sharpe Ratio'] = 0
        
        # Sortino Ratio
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['Sortino Ratio'] = (metrics['Annual Return'] - self.rf) / downside_std if downside_std > 0 else 0
        
        # Drawdown metrics
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        metrics['Max Drawdown'] = drawdown.min()
        metrics['Calmar Ratio'] = metrics['Annual Return'] / abs(metrics['Max Drawdown']) if abs(metrics['Max Drawdown']) > 0 else 0
        
        # Omega Ratio
        threshold = self.rf / 252
        excess_returns = self.returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        metrics['Omega Ratio'] = gains / losses if losses > 0 else np.inf
        
        # Beta and Alpha (if benchmark available)
        if self.benchmark_returns is not None:
            aligned = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
            if len(aligned) > 10:
                cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0][1]
                var = np.var(aligned.iloc[:, 1])
                metrics['Beta'] = cov / var if var > 0 else 0
                benchmark_ann_return = aligned.iloc[:, 1].mean() * 252
                metrics['Alpha'] = (metrics['Annual Return'] - self.rf) - metrics['Beta'] * (benchmark_ann_return - self.rf)
            else:
                metrics['Beta'] = 0
                metrics['Alpha'] = 0
        else:
            metrics['Beta'] = 0
            metrics['Alpha'] = 0
        
        # VaR metrics
        metrics['Historical VaR (95%)'] = -np.percentile(self.returns, 5)
        metrics['CVaR (95%)'] = -self.returns[self.returns <= np.percentile(self.returns, 5)].mean()
        
        return pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })

# --- 5. ENHANCED STRESS TESTING MODULE ---
class EnhancedStressTestLab:
    @staticmethod
    def run_custom_stress_test(portfolio_returns: pd.Series, shock_params: Dict) -> Dict:
        """Run custom stress test based on user-defined parameters."""
        base_vol = portfolio_returns.std()
        
        if shock_params['shock_type'] == 'market_crash':
            shock_return = -shock_params['magnitude'] / shock_params['duration']
            simulated = pd.Series([shock_return] * shock_params['duration'])
            total_return = (1 + simulated).prod() - 1
            max_dd = total_return
            
        elif shock_params['shock_type'] == 'volatility_spike':
            shock_vol = base_vol * (1 + shock_params['magnitude'])
            simulated = np.random.normal(portfolio_returns.mean(), shock_vol, shock_params['duration'])
            simulated = pd.Series(simulated)
            total_return = (1 + simulated).prod() - 1
            max_dd = ((1 + simulated).cumprod() / (1 + simulated).cumprod().cummax() - 1).min()
            
        else:
            simulated = portfolio_returns.tail(shock_params['duration'])
            total_return = (1 + simulated).prod() - 1
            max_dd = ((1 + simulated).cumprod() / (1 + simulated).cumprod().cummax() - 1).min()
        
        vol = simulated.std() * np.sqrt(252) if len(simulated) > 0 else 0
        
        return {
            "Scenario": f"Custom: {shock_params['shock_type'].replace('_', ' ').title()}",
            "Magnitude": f"{shock_params['magnitude']:.1%}",
            "Duration": f"{shock_params['duration']} days",
            "Total Return": total_return,
            "Max Drawdown": max_dd,
            "Volatility (Ann.)": vol
        }
    
    @staticmethod
    def run_historical_stress_test(portfolio_returns: pd.Series) -> pd.DataFrame:
        """Run historical stress scenarios."""
        historical_scenarios = {
            "COVID-19 Crash (2020)": {"start": "2020-02-19", "end": "2020-03-23"},
            "2022 Inflation/Correction": {"start": "2022-01-03", "end": "2022-06-30"},
            "2018 Trade War": {"start": "2018-10-01", "end": "2018-12-24"},
            "Turkey 2018 Currency Crisis": {"start": "2018-08-01", "end": "2018-09-01"},
            "Banking Turmoil (2023)": {"start": "2023-03-01", "end": "2023-03-31"},
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
                        "Period": f"{params['start']} to {params['end']}",
                        "Total Return": total_return,
                        "Max Drawdown": max_dd,
                        "Volatility (Ann.)": vol
                    })
            except:
                continue
        
        return pd.DataFrame(results)

# --- 6. VISUALIZATION ENGINE ---
class EnhancedChartFactory:
    @staticmethod
    def plot_efficient_frontier(mu, S, optimal_weights=None):
        """Plot efficient frontier with optimal portfolio."""
        if mu is None or S is None:
            return go.Figure()
            
        n_samples = 5000
        w = np.random.dirichlet(np.ones(len(mu)), n_samples)
        rets = w.dot(mu)
        stds = np.sqrt(np.diag(w @ S @ w.T))
        sharpes = rets / stds

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stds, y=rets, mode='markers',
            marker=dict(size=5, color=sharpes, colorscale='Viridis', showscale=True),
            name='Feasible Set'
        ))
        
        if optimal_weights is not None:
            opt_ret = optimal_weights.dot(mu)
            opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(S, optimal_weights)))
            fig.add_trace(go.Scatter(
                x=[opt_vol], y=[opt_ret], mode='markers',
                marker=dict(symbol='star', size=25, color='#e63946'),
                name='Optimal Portfolio'
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
        """Plot drawdown chart."""
        wealth_index = (1 + returns_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill='tozeroy', fillcolor='rgba(230, 57, 70, 0.2)',
            line=dict(color='#e63946', width=2),
            name='Drawdown'
        ))
        fig.update_layout(
            title=title,
            yaxis_tickformat=".1%",
            template="plotly_white",
            height=400
        )
        return fig

# --- 7. MAIN APPLICATION ---
def main():
    st.markdown('<div class="main-header">🌐 QUANTUM | Global Institutional Analytics Platform</div>', unsafe_allow_html=True)
    
    # Initialize data manager
    dm = EnhancedDataManager()
    
    # Initialize session state
    if 'custom_shocks' not in st.session_state:
        st.session_state.custom_shocks = []
    if 'selected_assets' not in st.session_state:
        st.session_state.selected_assets = []
    
    with st.sidebar:
        st.header("🌍 Global Asset Selection")
        
        # Quick portfolio presets
        st.subheader("Quick Portfolios")
        col_preset1, col_preset2 = st.columns(2)
        
        with col_preset1:
            if st.button("Global 60/40", use_container_width=True):
                st.session_state.selected_assets = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
        
        with col_preset2:
            if st.button("Emerging Markets", use_container_width=True):
                st.session_state.selected_assets = ["EEM", "THYAO.IS", "BABA", "005930.KS"]
        
        st.divider()
        
        # Asset selection
        selected_assets = []
        default_assets = ["SPY", "TLT", "GLD", "AAPL", "THYAO.IS"]
        
        for category, assets in dm.universe.items():
            with st.expander(f"📊 {category}", expanded=(category in ["US ETFs (Major & Active)", "Global Mega Caps"])):
                selected = st.multiselect(
                    f"Select from {category}",
                    options=list(assets.keys()),
                    default=[k for k in assets.keys() if assets[k] in default_assets],
                    key=f"select_{category}"
                )
                for s in selected:
                    selected_assets.append(assets[s])
        
        # Use session state if quick portfolio was selected
        if st.session_state.selected_assets:
            selected_assets = st.session_state.selected_assets
        
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
                st.progress(pct/100, text=f"{region}: {pct:.1f}%")
    
    if not selected_assets:
        st.warning("Please select at least one asset from the sidebar.")
        return
    
    # Fetch and align data
    with st.spinner("🔄 Fetching and aligning data..."):
        df_prices, benchmark_data, data_report = dm.fetch_and_align_data(
            selected_tickers=selected_assets,
            start_date=start_date.strftime("%Y-%m-%d"),
            min_data_length=min_data_length
        )
    
    # Display data quality report
    if data_report:
        dm.display_data_quality_report(data_report)
    
    if df_prices.empty:
        st.error("❌ No valid data available after alignment. Please select different assets or adjust date range.")
        return
    
    # Show data summary
    st.success(f"✅ Data ready for analysis: {len(df_prices)} data points, {len(df_prices.columns)} assets")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview", 
        "🎯 Portfolio Optimization", 
        "⚠️ Stress Testing",
        "📊 Risk Analytics",
        "🔗 Correlation Analysis"
    ])
    
    # --- TAB 1: DATA OVERVIEW ---
    with tab1:
        st.subheader("📊 Data Overview & Visualization")
        
        # Show normalized price chart
        st.markdown("### 📈 Normalized Price Performance")
        normalized = (df_prices / df_prices.iloc[0]) * 100
        
        # FIX FOR NARWHALS DUPLICATE ERROR: Ensure unique column names
        # Add a small suffix to duplicate names if they exist
        normalized_cols = normalized.columns.tolist()
        seen = {}
        new_cols = []
        
        for col in normalized_cols:
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
                st.warning(f"Renaming duplicate column: {col} -> {new_col}")
            else:
                seen[col] = 0
                new_col = col
            new_cols.append(new_col)
        
        normalized.columns = new_cols
        
        # Use Plotly Graph Objects instead of Plotly Express to avoid Narwhals error
        fig = go.Figure()
        for column in normalized.columns:
            ticker_name = dm.get_ticker_name_map().get(column.split('_')[0], column)
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized[column],
                mode='lines',
                name=ticker_name,
                hovertemplate=f'{ticker_name}<br>Date: %{{x}}<br>Value: %{{y:.1f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title="All Assets Rebased to 100",
            xaxis_title="Date",
            yaxis_title="Index Value (Rebased to 100)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics
        st.markdown("### 📊 Summary Statistics")
        returns = df_prices.pct_change().dropna()
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Mean Return (Ann.)", f"{returns.mean().mean() * 252:.2%}")
        with col_stat2:
            st.metric("Avg Volatility (Ann.)", f"{returns.std().mean() * np.sqrt(252):.2%}")
        with col_stat3:
            corr_values = returns.corr().values
            corr_values = corr_values[~np.eye(corr_values.shape[0], dtype=bool)]  # Remove diagonal
            avg_corr = corr_values.mean()
            st.metric("Avg Correlation", f"{avg_corr:.2f}")
    
    # --- TAB 2: PORTFOLIO OPTIMIZATION (Only if available) ---
    with tab2:
        if not OPTIMIZATION_AVAILABLE:
            st.warning("Portfolio optimization requires PyPortfolioOpt. Install with: pip install PyPortfolioOpt")
            st.info("Running in data analysis mode only. Optimization features will be disabled.")
        else:
            st.subheader("🎯 Portfolio Construction & Optimization")
            
            # Optimization configuration
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            with col_conf1:
                # Filter out HRP if not available
                strategy_options = ["Max Sharpe Ratio", "Min Volatility", "Equal Weight"]
                if HRP_AVAILABLE:
                    strategy_options.insert(2, "Hierarchical Risk Parity (HRP)")
                if BLACK_LITTERMAN_AVAILABLE:
                    strategy_options.append("Black-Litterman (Views)")
                
                strategy = st.selectbox("Optimization Strategy", strategy_options)
            
            with col_conf2:
                rf_rate = st.number_input("Risk Free Rate (%)", value=4.5, step=0.1) / 100
            
            with col_conf3:
                amount = st.number_input("Investment Amount ($)", value=1000000, step=100000)
            
            # Black-Litterman views (if selected)
            views_dict = {}
            confidences = {}
            
            if strategy == "Black-Litterman (Views)" and BLACK_LITTERMAN_AVAILABLE:
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
            if st.button("Run Optimization", type="primary", use_container_width=True):
                with st.spinner("Running optimization..."):
                    try:
                        engine = EnhancedOptimizationEngine(df_prices)
                        
                        if strategy == "Hierarchical Risk Parity (HRP)" and HRP_AVAILABLE:
                            weights, perf = engine.optimize_hrp()
                        elif strategy == "Black-Litterman (Views)" and views_dict and BLACK_LITTERMAN_AVAILABLE:
                            weights, perf, _ = engine.optimize_black_litterman(views_dict, confidences)
                        elif strategy == "Equal Weight":
                            equal_weight = 1.0 / len(df_prices.columns)
                            weights = {asset: equal_weight for asset in df_prices.columns}
                            port_returns = engine.returns.dot(pd.Series(weights))
                            ann_ret = port_returns.mean() * 252
                            ann_vol = port_returns.std() * np.sqrt(252)
                            sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
                            perf = (ann_ret, ann_vol, sharpe)
                        else:
                            obj_type = "max_sharpe" if "Sharpe" in strategy else "min_volatility"
                            weights, perf = engine.optimize_mean_variance(obj_type, rf_rate)
                        
                        if weights and perf:
                            # Display results
                            st.success("✅ Optimization completed!")
                            
                            # Portfolio allocation
                            col_res1, col_res2 = st.columns([1, 1])
                            
                            with col_res1:
                                st.subheader("📊 Portfolio Allocation")
                                ticker_map = dm.get_ticker_name_map()
                                portfolio_data = []
                                
                                for ticker, weight in weights.items():
                                    if weight > 0.001:
                                        portfolio_data.append({
                                            "Asset": ticker_map.get(ticker, ticker),
                                            "Weight": weight,
                                            "Amount ($)": weight * amount
                                        })
                                
                                portfolio_df = pd.DataFrame(portfolio_data).sort_values(by="Weight", ascending=False)
                                
                                display_df = portfolio_df[['Asset', 'Weight', 'Amount ($)']].copy()
                                display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2%}")
                                display_df['Amount ($)'] = display_df['Amount ($)'].apply(lambda x: f"${x:,.0f}")
                                st.dataframe(display_df, use_container_width=True, height=400)
                            
                            with col_res2:
                                st.subheader("📈 Allocation Chart")
                                if not portfolio_df.empty:
                                    fig_pie = px.pie(
                                        portfolio_df, 
                                        values='Weight', 
                                        names='Asset', 
                                        hole=0.4
                                    )
                                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                    fig_pie.update_layout(
                                        template="plotly_white", 
                                        height=400,
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Performance metrics
                            st.divider()
                            st.subheader("🎯 Performance Summary")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Expected Return (Ann.)", f"{perf[0]:.2%}")
                            m2.metric("Volatility (Ann.)", f"{perf[1]:.2%}")
                            m3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                            
                            # Calculate drawdown
                            portfolio_returns_series = engine.returns.dot(pd.Series(weights))
                            cum_returns = (1 + portfolio_returns_series).cumprod()
                            running_max = cum_returns.cummax()
                            drawdown = (cum_returns - running_max) / running_max
                            max_dd = drawdown.min()
                            m4.metric("Max Drawdown", f"{max_dd:.2%}")
                            
                            # Store portfolio returns for other tabs
                            st.session_state.portfolio_returns = portfolio_returns_series
                            st.session_state.weights = weights
                            
                            # Efficient Frontier
                            st.divider()
                            st.subheader("📊 Efficient Frontier Analysis")
                            weights_array = np.array([weights.get(col, 0) for col in df_prices.columns])
                            fig_ef = EnhancedChartFactory.plot_efficient_frontier(engine.mu, engine.S, weights_array)
                            st.plotly_chart(fig_ef, use_container_width=True)
                        else:
                            st.error("Optimization failed to produce valid results.")
                        
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)[:200]}")
    
    # --- TAB 3: STRESS TESTING ---
    with tab3:
        st.subheader("⚠️ Stress Testing Laboratory")
        
        # Check if we have portfolio returns
        if 'portfolio_returns' not in st.session_state:
            st.warning("Please run portfolio optimization first to generate portfolio returns for stress testing.")
        else:
            portfolio_returns = st.session_state.portfolio_returns
            
            # Historical stress tests
            st.markdown("### 📜 Historical Stress Scenarios")
            stress_engine = EnhancedStressTestLab()
            historical_df = stress_engine.run_historical_stress_test(portfolio_returns)
            
            if not historical_df.empty:
                st.dataframe(
                    historical_df.style.format({
                        "Total Return": "{:.2%}",
                        "Max Drawdown": "{:.2%}",
                        "Volatility (Ann.)": "{:.2%}"
                    }).background_gradient(cmap="RdYlGn_r", subset=["Total Return"]),
                    use_container_width=True,
                    height=300
                )
            
            # Custom stress test builder
            st.divider()
            st.markdown("### 🛠️ Custom Stress Test Builder")
            
            col_custom1, col_custom2, col_custom3 = st.columns(3)
            
            with col_custom1:
                shock_type = st.selectbox(
                    "Shock Type",
                    ["market_crash", "volatility_spike", "sector_specific", "liquidity_crunch"]
                )
            
            with col_custom2:
                magnitude = st.slider("Shock Magnitude", 0.1, 1.0, 0.3, 0.1, format="%.1f")
            
            with col_custom3:
                duration = st.slider("Duration (days)", 5, 250, 30)
            
            if st.button("Run Custom Stress Test", type="primary"):
                shock_params = {
                    "shock_type": shock_type,
                    "magnitude": magnitude,
                    "duration": duration
                }
                
                custom_result = stress_engine.run_custom_stress_test(portfolio_returns, shock_params)
                
                # Display results
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Scenario", custom_result["Scenario"])
                col_res2.metric("Estimated Loss", f"{custom_result['Total Return']:.2%}")
                col_res3.metric("Max Drawdown", f"{custom_result['Max Drawdown']:.2%}")
                
                # Store in session state
                st.session_state.custom_shocks.append(custom_result)
            
            # Drawdown analysis
            st.divider()
            st.markdown("### 📉 Drawdown Analysis")
            fig_dd = EnhancedChartFactory.plot_drawdown(portfolio_returns)
            st.plotly_chart(fig_dd, use_container_width=True)
    
    # --- TAB 4: RISK ANALYTICS ---
    with tab4:
        st.subheader("🛡️ Risk Analytics")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Please run portfolio optimization first.")
        else:
            portfolio_returns = st.session_state.portfolio_returns
            
            # Initialize risk engine
            risk_engine = EnhancedRiskMetricsEngine(
                portfolio_returns, 
                rf_rate=0.04,
                benchmark_returns=benchmark_data.pct_change().dropna() if not benchmark_data.empty else None
            )
            
            # Comprehensive metrics
            st.markdown("### 📊 Risk & Performance Metrics")
            metrics_df = risk_engine.calculate_comprehensive_metrics()
            
            # Display metrics in two columns
            col_met1, col_met2 = st.columns(2)
            
            with col_met1:
                return_metrics = metrics_df[metrics_df['Metric'].isin([
                    'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                    'Max Drawdown', 'Calmar Ratio'
                ])]
                for _, row in return_metrics.iterrows():
                    if 'Ratio' in row['Metric']:
                        st.metric(row['Metric'], f"{row['Value']:.3f}")
                    else:
                        st.metric(row['Metric'], f"{row['Value']:.2%}")
            
            with col_met2:
                other_metrics = metrics_df[~metrics_df['Metric'].isin([
                    'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                    'Max Drawdown', 'Calmar Ratio'
                ])]
                for _, row in other_metrics.iterrows():
                    if 'Ratio' in row['Metric'] or 'Beta' in row['Metric'] or 'Alpha' in row['Metric']:
                        st.metric(row['Metric'], f"{row['Value']:.3f}")
                    elif 'VaR' in row['Metric'] or 'CVaR' in row['Metric']:
                        st.metric(row['Metric'], f"{row['Value']:.2%}")
                    else:
                        st.metric(row['Metric'], f"{row['Value']:.3f}")
            
            # VaR Analysis
            st.divider()
            st.markdown("### 💰 Value at Risk Analysis")
            
            col_var1, col_var2 = st.columns(2)
            with col_var1:
                confidence = st.select_slider("Confidence Level", [0.90, 0.95, 0.99], value=0.95)
            
            with col_var2:
                holding_days = st.selectbox("Holding Period", [1, 5, 10, 20], index=0)
            
            # Calculate VaR
            alpha = 1 - confidence
            scale = np.sqrt(holding_days)
            var_pct = -np.percentile(portfolio_returns, alpha * 100) * scale
            var_value = var_pct * 1000000  # Assuming $1M investment
            
            col_var_disp1, col_var_disp2 = st.columns(2)
            col_var_disp1.metric(f"{confidence:.0%} VaR", f"{var_pct:.2%}")
            col_var_disp2.metric(f"{confidence:.0%} VaR ($1M)", f"${var_value:,.0f}")
    
    # --- TAB 5: CORRELATION ANALYSIS ---
    with tab5:
        st.subheader("🔗 Correlation Analysis")
        
        returns = df_prices.pct_change().dropna()
        
        # Correlation matrix
        st.markdown("### 📊 Correlation Matrix")
        corr_matrix = returns.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        fig_corr.update_layout(
            template="plotly_white",
            height=700,
            title="Asset Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Top correlations
        st.markdown("### 📈 Highest & Lowest Correlations")
        col_corr1, col_corr2 = st.columns(2)
        
        with col_corr1:
            # Get top positive correlations
            corr_series = corr_matrix.unstack()
            corr_series = corr_series[corr_series < 1]  # Remove self-correlations
            top_corrs = corr_series.sort_values(ascending=False).head(10)
            
            st.subheader("Highest Correlations")
            for idx, (pair, value) in enumerate(top_corrs.items()):
                asset1, asset2 = pair
                name1 = dm.get_ticker_name_map().get(asset1, asset1)
                name2 = dm.get_ticker_name_map().get(asset2, asset2)
                st.metric(f"{name1} & {name2}", f"{value:.3f}")
        
        with col_corr2:
            # Get top negative correlations
            bottom_corrs = corr_series.sort_values(ascending=True).head(10)
            
            st.subheader("Lowest Correlations")
            for idx, (pair, value) in enumerate(bottom_corrs.items()):
                asset1, asset2 = pair
                name1 = dm.get_ticker_name_map().get(asset1, asset1)
                name2 = dm.get_ticker_name_map().get(asset2, asset2)
                st.metric(f"{name1} & {name2}", f"{value:.3f}")

if __name__ == "__main__":
    main()
