import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
from typing import Optional, List, Dict, Tuple
import requests
import time
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path

def load_css():
    """Load custom CSS from file"""
    css_file = Path(__file__).parent / "assets" / "style.css"
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Add custom fonts
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS
load_css()

class AdvancedStockAnalyzer:
    def __init__(self, symbol: str):
        """
        Initialize stock analyzer with support for multiple exchanges
        
        :param symbol: Stock symbol (e.g., 'AAPL', 'INFY.NS', 'RELIANCE.BO')
        """
        self.symbol = symbol
        self.data = None
        self.exchanges = {
            'NYSE': '',          # US Stocks
            'NASDAQ': '',         # US Tech Stocks
            'NSE': '.NS',         # National Stock Exchange (India)
            'BSE': '.BO',         # Bombay Stock Exchange (India)
            'GLOBAL': ''          # Global stocks
        }
    
    def get_stock_info(self) -> dict:
        """
        Fetch stock information using yfinance
        
        Returns:
            Dictionary containing stock information
        """
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            # Ensure we have a dictionary and not a list
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
                
            # Ensure we have a dictionary
            if not isinstance(info, dict):
                info = {}
                
            # Add basic info if missing
            if 'symbol' not in info:
                info['symbol'] = self.symbol
                
            # Calculate market cap if not available
            if 'marketCap' not in info and 'currentPrice' in info and 'sharesOutstanding' in info:
                info['marketCap'] = info['currentPrice'] * info['sharesOutstanding']
                
            return info
            
        except Exception as e:
            print(f"Error fetching stock info for {self.symbol}: {e}")
            return {}
    
    def fetch_stock_data(self, period: str = '1y', max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch stock data using Yahoo Finance with improved reliability
        
        Args:
            period: Time period for data retrieval (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame containing the stock data or empty DataFrame if failed
        """
        import yfinance as yf
        import time
        from datetime import datetime, timedelta
        
        def try_download(symbol: str, period: str) -> Tuple[bool, pd.DataFrame]:
            """Helper function to try downloading data with retries"""
            for attempt in range(max_retries):
                try:
                    # Suppress yfinance logs
                    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
                    # Download with group_by='ticker' for consistent data structure
                    data = yf.download(
                        symbol,
                        period=period,
                        progress=False,
                        auto_adjust=True,  # Automatically adjust OHLC
                        threads=True,      # Use threads for faster download
                        group_by='ticker'  # Group by ticker for consistent structure
                    )
                    
                    if not data.empty:
                        # If we get a MultiIndex DataFrame, convert to single level
                        if isinstance(data.columns, pd.MultiIndex):
                            # Get the first ticker's data
                            ticker = data.columns.levels[0][0]
                            data = data[ticker]
                        return True, data
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                    time.sleep(1)  # Wait before retry
            return False, pd.DataFrame()
        
        # Clean and validate the symbol
        symbol = self.symbol.strip().upper()
        
        # Common exchange suffixes to try (starting with no suffix for US stocks)
        exchange_suffixes = ['']  # Try without suffix first for US stocks
        
        # Only try other exchanges if not a US stock (no dot in symbol)
        if '.' not in symbol:
            exchange_suffixes.extend(['.NS', '.BO', '.NSE', '.BS', '.AX', '.L', '.PA', '.DE', '.TO'])
        
        # Try different period formats if the first attempt fails
        periods_to_try = [period, '1y', '6mo', '3mo']
        
        for suffix in exchange_suffixes:
            current_symbol = f"{symbol}{suffix}"
            for p in periods_to_try:
                success, data = try_download(current_symbol, p)
                if success and not data.empty:
                    # Ensure we have required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in data.columns for col in required_columns):
                        # Add symbol info to the dataframe
                        data['Symbol'] = current_symbol
                        if not data.index.name:
                            data.index.name = 'Date'
                        self.symbol = current_symbol
                        return data
        
        # If we get here, all attempts failed
        st.error(f"❌ Could not fetch data for {symbol} after multiple attempts")
        st.warning("""
        Common issues:
        1. Check if the stock symbol is correct
        2. Try adding an exchange suffix (e.g., '.NS' for NSE, '.BO' for BSE)
        3. The stock may be delisted or not available on Yahoo Finance
        """)
        return pd.DataFrame()
        
        # Show example symbols
        st.markdown("""
        **Example Symbols:**
        - US Stocks: `AAPL`, `MSFT`, `GOOGL`
        - Indian Stocks: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
        - Other: `BHP.AX` (Australia), `HSBA.L` (UK)
        
        **Troubleshooting:**
        1. Check your internet connection
        2. Try a different symbol
        3. The market may be closed
        4. Wait a few minutes and try again
        """)
        
        return pd.DataFrame()
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators with proper data alignment
        
        :param data: DataFrame with stock price data (can be MultiIndex from yfinance or dictionary with ticker key)
        :return: DataFrame with added indicator columns
        """
        try:
            if data is None or (hasattr(data, 'empty') and data.empty):
                st.warning("⚠️ No data provided for indicator calculation")
                return pd.DataFrame()
                
            # If data is a dictionary with ticker as key, extract the first value
            if isinstance(data, dict):
                if not data:
                    st.warning("⚠️ Empty data dictionary provided")
                    return pd.DataFrame()
                # Get the first ticker's data
                ticker_data = next(iter(data.values()))
                if hasattr(ticker_data, 'empty') and ticker_data.empty:
                    st.warning("⚠️ No data available for the selected ticker")
                    return pd.DataFrame()
                df = ticker_data.copy(deep=True)
            else:
                # Make a deep copy of the input data to avoid modifying the original
                df = data.copy(deep=True)
            
            # If DataFrame has MultiIndex columns, convert to single level
            if isinstance(df.columns, pd.MultiIndex):
                # Get the first level of the MultiIndex (ticker symbols)
                ticker = df.columns.get_level_values(0)[0]
                # Convert to regular DataFrame with single-level columns
                df = df[ticker].copy()
                
            # If we still don't have a DataFrame, try to create one from the data
            if not isinstance(df, pd.DataFrame):
                st.error(f"❌ Unexpected data format: {type(df)}")
                return pd.DataFrame()
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.error(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
                
            # Convert columns to numeric one by one with error handling
            for col in required_columns:
                try:
                    # Ensure we're working with a Series, not a DataFrame
                    if isinstance(df[col], pd.DataFrame):
                        # If it's a DataFrame, take the first column
                        df[col] = pd.to_numeric(df[col].iloc[:, 0], errors='coerce')
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check if conversion resulted in all NaN values
                    if df[col].isna().all():
                        st.warning(f"⚠️ Column '{col}' contains non-numeric data that couldn't be converted")
                        st.write(f"Sample values from {col}:", df[col].head().to_dict())
                        return pd.DataFrame()
                        
                except Exception as e:
                    st.error(f"❌ Error processing column '{col}': {str(e)}")
                    st.write(f"Column '{col}' type: {type(df[col])}")
                    if hasattr(df[col], 'iloc'):
                        st.write(f"Sample values from {col}:", df[col].iloc[:5].to_dict() if len(df) > 0 else "No data")
                    return pd.DataFrame()
            
            # Drop any rows with missing price data
            initial_count = len(df)
            df = df.dropna(subset=required_columns, how='any')
            if len(df) < initial_count:
                st.warning(f"⚠️ Dropped {initial_count - len(df)} rows with missing data")
            
            if len(df) < 30:  # Minimum data points needed for indicators
                st.warning(f"⚠️ Insufficient data points ({len(df)}) for accurate indicators. Need at least 30.")
                # Add empty columns for expected output structure
                indicator_columns = ['RSI', 'MACD', 'MACD_Signal', 'BB_Middle', 
                                   'BB_Std', 'BB_High', 'BB_Low',
                                   'RSI_Signal', 'MACD_Signal', 'BB_Signal']
                for col in indicator_columns:
                    df[col] = np.nan if not col.endswith('_Signal') else 'Neutral'
                return df
            
            try:
                # Calculate RSI with proper alignment
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                
                # Handle division by zero and ensure proper alignment
                rs = gain / loss.replace(0, float('inf'))
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Calculate MACD with proper alignment
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                # Calculate Bollinger Bands with proper alignment
                rolling_mean = df['Close'].rolling(window=20, min_periods=1).mean()
                rolling_std = df['Close'].rolling(window=20, min_periods=1).std(ddof=0)
                
                df['BB_Middle'] = rolling_mean
                df['BB_Std'] = rolling_std
                df['BB_High'] = rolling_mean + (2 * rolling_std)
                df['BB_Low'] = rolling_mean - (2 * rolling_std)
                
                # Generate trading signals with proper alignment
                df['RSI_Signal'] = pd.Series(index=df.index, dtype='object')
                df['RSI_Signal'] = np.where(
                    df['RSI'] < 30, 'Buy',
                    np.where(df['RSI'] > 70, 'Sell', 'Hold')
                )
                
                df['MACD_Signal'] = pd.Series(index=df.index, dtype='object')
                df['MACD_Signal'] = np.where(
                    df['MACD'] > df['MACD_Signal'], 'Buy', 'Sell'
                )
                
                df['BB_Signal'] = pd.Series(index=df.index, dtype='object')
                df['BB_Signal'] = np.where(
                    df['Close'] < df['BB_Low'], 'Buy',
                    np.where(df['Close'] > df['BB_High'], 'Sell', 'Hold')
                )
                
                # Clean up any infinite or NaN values
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
                
                # Forward fill then backward fill any remaining NaNs
                df = df.ffill().bfill()
                
                # Ensure all required signal columns exist and have the correct type
                for col in ['RSI_Signal', 'MACD_Signal', 'BB_Signal']:
                    if col not in df.columns or df[col].isna().all():
                        df[col] = 'Neutral'
                    # Ensure string type for signal columns
                    df[col] = df[col].astype(str).str.strip()
                
                return df
                
            except Exception as calc_error:
                st.error(f"❌ Error in indicator calculation: {str(calc_error)}")
                # Ensure we return a DataFrame with all expected columns
                indicator_columns = ['RSI', 'MACD', 'MACD_Signal', 'BB_Middle', 
                                   'BB_Std', 'BB_High', 'BB_Low',
                                   'RSI_Signal', 'MACD_Signal', 'BB_Signal']
                for col in indicator_columns:
                    if col not in df.columns:
                        df[col] = np.nan if not col.endswith('_Signal') else 'Neutral'
                return df
                
        except Exception as e:
            st.error(f"❌ Fatal error in calculate_indicators: {str(e)}")
            # Ensure we return a properly structured DataFrame even in case of error
            df = data.copy() if 'data' in locals() and data is not None else pd.DataFrame()
            indicator_columns = ['RSI', 'MACD', 'MACD_Signal', 'BB_Middle', 
                               'BB_Std', 'BB_High', 'BB_Low',
                               'RSI_Signal', 'MACD_Signal', 'BB_Signal']
            for col in indicator_columns:
                if col not in df.columns:
                    df[col] = np.nan if not col.endswith('_Signal') else 'Neutral'
            return df
    
    def generate_combined_signal(self, data: pd.DataFrame) -> str:
        """
        Generate a combined buy/sell signal based on multiple indicators
        
        :param data: DataFrame with indicator columns (can be MultiIndex)
        :return: Final buy/sell recommendation with confidence level
        """
        try:
            if data is None or data.empty:
                st.warning("⚠️ No data available for signal generation")
                return 'Neutral (No Data)'
            
            # Handle MultiIndex if present
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten the columns by joining the levels with an underscore
                data = data.copy()
                data.columns = ['_'.join(col).strip('_') for col in data.columns.values]
            
            # Ensure all required columns exist, create them if missing
            required_columns = ['RSI_Signal', 'MACD_Signal', 'BB_Signal']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = 'Neutral'  # Default to Neutral if signal is missing
            
            try:
                # Get the last 5 days of signals for better context
                recent_signals = data[required_columns].dropna(how='all').tail(5)
                
                if recent_signals.empty:
                    st.warning("⚠️ No valid signals found in the recent data")
                    return 'Neutral (No Signals)'
                
                # Get the most recent signals
                latest_row = recent_signals.iloc[-1]
                
                # Convert all signal values to strings and standardize case
                signal_values = []
                for val in latest_row.values:
                    if pd.isna(val):
                        signal_values.append('Neutral')
                    else:
                        signal_values.append(str(val).strip().title())
                
                # Count occurrences of each signal type in the latest data point
                buy_count = signal_values.count('Buy')
                sell_count = signal_values.count('Sell')
                hold_count = signal_values.count('Hold')
                neutral_count = signal_values.count('Neutral')
                
                # Calculate signal strength based on recent trend
                trend_strength = 0
                for col in required_columns:
                    # Count signal changes in the last 5 days
                    signals = recent_signals[col].str.strip().str.title()
                    if len(signals) >= 2:
                        changes = (signals != signals.shift()).sum()
                        trend_strength += 1.0 / (changes + 1)  # More stable signals get higher weight
                
                # Adjust signal based on trend strength
                signal_strength = 'Strong ' if trend_strength > 2.0 else ''
                
                # Determine overall signal with priority to stronger signals
                if buy_count > sell_count and buy_count > 0:
                    return f"{signal_strength}Buy" if trend_strength > 1.0 else "Buy"
                elif sell_count > buy_count and sell_count > 0:
                    return f"{signal_strength}Sell" if trend_strength > 1.0 else "Sell"
                elif neutral_count == len(required_columns):
                    return 'Neutral'
                else:
                    # If we have mixed signals, check the trend
                    if trend_strength > 2.0:
                        return 'Hold (Strong Trend)'
                    return 'Hold' if hold_count > 0 else 'Neutral'
                    
            except Exception as e:
                st.error(f"❌ Error processing signals: {str(e)}")
                if st.session_state.get('debug', False):
                    st.error(f"Latest row data: {latest_row}")
                    st.error(f"Required columns: {required_columns}")
                return 'Neutral (Error)'
            
        except Exception as e:
            st.error(f"❌ Unexpected error in signal generation: {str(e)}")
            return 'Neutral (Error)'
    
    def create_interactive_chart(self, data: pd.DataFrame):
        """
        Create an interactive Plotly chart with multiple indicators
        
        :param data: DataFrame with stock and indicator data
        :return: Plotly figure object
        """
        try:
            if data.empty:
                st.warning("No data available for chart")
                return None
                
            # Make a copy to avoid modifying original data
            plot_data = data.copy()
            
            # Ensure we have a datetime index
            if 'Date' in plot_data.columns:
                plot_data.set_index('Date', inplace=True)
            
            # Create subplots with proper layout
            fig = sp.make_subplots(
                rows=4, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=(
                    f'{self.symbol} Price',
                    'Relative Strength Index (RSI)',
                    'Moving Average Convergence Divergence (MACD)',
                    'Volume'
                )
            )
            
            # Add candlestick chart with metallic red and green colors
            fig.add_trace(
                go.Candlestick(
                    x=plot_data.index,
                    open=plot_data['Open'],
                    high=plot_data['High'],
                    low=plot_data['Low'],
                    close=plot_data['Close'],
                    name='Price',
                    increasing_line_color='#00C853',  # Brighter green for up moves
                    increasing_fillcolor='rgba(0, 200, 83, 0.8)',
                    decreasing_line_color='#ff1744',  # Brighter red for down moves
                    decreasing_fillcolor='rgba(255, 23, 68, 0.8)',
                    line=dict(width=1.5),
                    whiskerwidth=1.0
                ),
                row=1, col=1
            )
            
            # Add moving averages if they exist
            for ma, color in [('MA20', '#3498db'), ('MA50', '#f39c12'), ('MA200', '#9b59b6')]:
                if ma in plot_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data[ma],
                            name=ma,
                            line=dict(color=color, width=1.5)
                        ),
                        row=1, col=1
                    )
            
            # Add RSI if it exists
            if 'RSI' in plot_data.columns:
                # Add RSI line
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data['RSI'],
                        name='RSI',
                        line=dict(color='#3498db', width=1.5)
                    ),
                    row=2, col=1
                )
                
                # Add RSI overbought/oversold levels with better styling
                fig.add_hline(
                    y=70, 
                    line_dash='dash', 
                    line_color='#ef5350', 
                    opacity=0.5, 
                    row=2, 
                    col=1,
                    annotation_text='Overbought',
                    annotation_position='top right'
                )
                fig.add_hline(
                    y=30, 
                    line_dash='dash', 
                    line_color='#26a69a', 
                    opacity=0.5, 
                    row=2, 
                    col=1,
                    annotation_text='Oversold',
                    annotation_position='bottom right'
                )
                
                # Set RSI y-axis range and title
                fig.update_yaxes(
                    range=[0, 100], 
                    row=2, 
                    col=1,
                    title_text='RSI',
                    title_standoff=10
                )
            
            # Add MACD if it exists
            if all(col in plot_data.columns for col in ['MACD', 'Signal_Line']):
                # MACD line
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data['MACD'],
                        name='MACD',
                        line=dict(color='#3498db', width=1.5)
                    ),
                    row=3, col=1
                )
                
                # Signal line
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data['Signal_Line'],
                        name='Signal',
                        line=dict(color='#f39c12', width=1.5)
                    ),
                    row=3, col=1
                )
                
                # MACD Histogram
                if 'MACD_Hist' in plot_data.columns:
                    colors = ['#26a69a' if val >= 0 else '#ef5350' 
                            for val in plot_data['MACD_Hist']]
                    fig.add_trace(
                        go.Bar(
                            x=plot_data.index,
                            y=plot_data['MACD_Hist'],
                            name='MACD Histogram',
                            marker_color=colors,
                            opacity=0.6
                        ),
                        row=3, col=1
                    )
                
                # Add zero line for MACD
                fig.add_hline(
                    y=0,
                    line_width=1,
                    line_dash='solid',
                    line_color='#7f8c8d',
                    opacity=0.7,
                    row=3,
                    col=1
                )
                
                # Update MACD subplot title
                fig.update_yaxes(
                    title_text='MACD',
                    row=3,
                    col=1
                )
                
            # Add Volume if it exists
            if 'Volume' in plot_data.columns:
                colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' 
                         for _, row in plot_data.iterrows()]
                fig.add_trace(
                    go.Bar(
                        x=plot_data.index,
                        y=plot_data['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=4, col=1
                )
                
                # Update Volume subplot title
                fig.update_yaxes(
                    title_text='Volume',
                    row=4,
                    col=1
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                legend=dict(orientation='h', y=1.02, yanchor='bottom'),
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor='white',
                xaxis=dict(rangeslider=dict(visible=False)),
                xaxis4=dict(title_text='Date')
            )
            
            # Update x-axis ranges to be the same across all subplots
            fig.update_xaxes(matches='x')
            
            # Update layout
            fig.update_layout(
                height=1000,
                title_text=f'{self.symbol} Technical Analysis',
                hovermode='x unified',
                xaxis4_rangeslider_visible=False,
                xaxis4_rangeslider_thickness=0.1,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update y-axis titles
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating interactive chart: {str(e)}")
            return None

def interpret_trading_signal(signal):
    """
    Provide conservative signal interpretation
    
    :param signal: Trading signal 
    :return: Detailed signal explanation
    """
    # Conservative Signal Interpretation
    conservative_interpretation = {
        'Neutral': {
            'description': 'Market Uncertainty Detected',
            'recommendation': 'Caution Advised',
            'risk_level': 'Medium',
            'action_points': [
                'Insufficient clear market signals',
                'High volatility or mixed indicators',
                'Recommended to wait for more definitive trend',
                'Consider diversification strategies',
                'Monitor market conditions closely'
            ],
            'additional_insights': [
                'Technical indicators show conflicting patterns',
                'No strong directional momentum',
                'External factors may be influencing market',
                'Potential for rapid market changes'
            ]
        }
    }
    
    # Always return Neutral interpretation
    return conservative_interpretation.get('Neutral', {
        'description': 'Market Analysis Inconclusive',
        'recommendation': 'Exercise Extreme Caution',
        'risk_level': 'High',
        'action_points': ['Insufficient data for recommendation']
    })

def main():
    # Initialize session state
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    
    # Initial landing page with Get Started button
    if not st.session_state.show_analysis:
        # Full-width header with gradient
        st.markdown("""
        <div style='background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); 
                    padding: 4rem 2rem; 
                    border-radius: 16px; 
                    margin: -1.5rem -1.5rem 2rem -1.5rem; 
                    color: white; 
                    text-align: center;'>
            <h1 style='color: white; font-size: 3rem; margin-bottom: 1rem;'>Stock Trend Analyzer</h1>
            <p style='font-size: 1.2rem; max-width: 700px; margin: 0 auto 2rem;'>
                Advanced technical analysis and real-time stock signals to help you make informed trading decisions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights in a grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='premium-card' style='text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem;'>📊</div>
                <h3>Real-time Analysis</h3>
                <p>Get up-to-date technical indicators and signals for any stock</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='premium-card' style='text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem;'>📈</div>
                <h3>Smart Signals</h3>
                <p>AI-powered buy/sell signals based on multiple indicators</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='premium-card' style='text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem;'>🌐</div>
                <h3>Global Markets</h3>
                <p>Analyze stocks from NSE, NYSE, NASDAQ, and more</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Call to action
        st.markdown("""
        <div style='text-align: center; margin: 3rem 0;'>
            <h2 class='gradient-text' style='margin-bottom: 1.5rem;'>Start Analyzing Now</h2>
            <p style='max-width: 600px; margin: 0 auto 2rem; color: #666;'>
                Join thousands of traders who use our platform to make better investment decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the Get Started button with animation
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button('Get Started for Free →', type='primary', use_container_width=True):
                st.session_state.show_analysis = True
                st.rerun()
        
        # Footer
        st.markdown("""
        <div style='text-align: center; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid #eee; color: #888;'>
            <p>© 2023 Stock Trend Analyzer. All rights reserved.</p>
        </div>
        """, unsafe_allow_html=True)
        
        return
    
    # Main analysis interface with premium styling
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;'>
        <h1 style='margin: 0;'>Stock Analysis <span class='gradient-text'>Dashboard</span></h1>
        <div style='display: flex; gap: 1rem;'>
            <button onclick="window.location.href='#analysis'" class='css-1x8cf1d edgvbvh10' style='border-radius: 8px;'>📊 Analysis</button>
            <button onclick="window.location.href='#signals'" class='css-1x8cf1d edgvbvh10' style='border-radius: 8px;'>📈 Signals</button>
            <button onclick="window.location.href='#portfolio'" class='css-1x8cf1d edgvbvh10' style='border-radius: 8px;'>💼 Portfolio</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Predefined Stock Lists with premium styling
    stock_lists = {
        'NSE': [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
            'INFY.NS', 'HINDUNILVR.NS', 'ITC.NS', 'BHARTIARTL.NS',
            'KOTAKBANK.NS', 'LT.NS', 'HCLTECH.NS', 'AXISBANK.NS'
        ],
        'NASDAQ': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 
            'NVDA', 'PYPL', 'ADBE', 'NFLX', 'INTC', 'CSCO'
        ],
        'NYSE': [
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS',
            'BAC', 'XOM', 'VZ', 'WMT'
        ],
        'Crypto': [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD',
            'SOL-USD', 'ADA-USD', 'DOGE-USD', 'DOT-USD'
        ]
    }
    
    # Sidebar for stock selection with improved styling
    st.sidebar.markdown('### 📊 Stock Analysis')
    
    # Exchange selection with icons
    exchange_icons = {
        'NSE': '🇮🇳 NSE',
        'NASDAQ': '🇺🇸 NASDAQ',
        'NYSE': '🏛️ NYSE',
        'Crypto': '₿ Crypto'
    }
    
    # Display exchange selection with icons
    selected_exchange = st.sidebar.selectbox(
        'Select Exchange',
        options=list(exchange_icons.keys()),
        format_func=lambda x: exchange_icons[x]
    )
    
    # Add some space
    st.sidebar.markdown('---')
    
    # Stock symbol selection
    stock_symbol = st.sidebar.selectbox(
        'Select Stock',
        options=stock_lists[selected_exchange],
        index=0
    )
    
    # Manual symbol input
    manual_symbol = st.sidebar.text_input('Or enter custom symbol:', '')
    
    # Use manual symbol if provided, otherwise use selected symbol
    final_symbol = manual_symbol.strip() if manual_symbol.strip() else stock_symbol
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        'Time Period',
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'YTD', 'Max'],
        index=3  # Default to 1y
    )
    
    # Analyze button with better styling
    analyze_clicked = st.sidebar.button(
        '🔍 Analyze Stock', 
        type='primary',
        use_container_width=True
    )
    
    # Add some space at the bottom
    st.sidebar.markdown('---')
    
    # Analyze button logic
    if analyze_clicked or final_symbol:
        try:
            # Create stock analyzer
            analyzer = AdvancedStockAnalyzer(final_symbol)
            
            # Fetch stock data
            stock_data = analyzer.fetch_stock_data()
            
            if not stock_data.empty:
                # Calculate indicators
                stock_data_with_indicators = analyzer.calculate_indicators(stock_data)
                
                # Generate combined signal
                final_signal = analyzer.generate_combined_signal(stock_data_with_indicators)
                
                # Interpret Trading Signal
                signal_details = interpret_trading_signal(final_signal)
                
                # Display stock information with premium design
                with st.spinner('Analyzing {}...'.format(str(final_symbol))):
                    analyzer = AdvancedStockAnalyzer(final_symbol)
                    df = analyzer.fetch_stock_data(period=time_period)
                    
                    if not df.empty:
                        # Get stock info
                        stock_info = analyzer.get_stock_info()
                        current_price = float(df['Close'].iloc[-1])
                        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
                        price_change = float(current_price - prev_close)
                        percent_change = float((price_change / prev_close) * 100) if prev_close != 0 else 0.0
                        
                        # Premium stock header with gradient background
                        # Format market cap with proper error handling
                        try:
                            market_cap = float(stock_info.get('marketCap', 0))
                            market_cap_str = f'${market_cap/1e9:.2f}B' if market_cap > 0 else 'N/A'
                        except (TypeError, ValueError):
                            market_cap_str = 'N/A'
                        
                        # Format P/E ratio
                        pe_ratio = stock_info.get('trailingPE', 'N/A')
                        pe_ratio_str = f'{pe_ratio:.2f}' if isinstance(pe_ratio, (int, float)) else str(pe_ratio)
                        
                        # Format 52-week range
                        try:
                            low = float(stock_info.get('fiftyTwoWeekLow', 0))
                            high = float(stock_info.get('fiftyTwoWeekHigh', 0))
                            week_range = f'${low:.2f} - ${high:.2f}'
                        except (TypeError, ValueError):
                            week_range = 'N/A'
                            
                        # Get short name from stock info
                        short_name = str(stock_info.get('shortName', final_symbol))
                        
                        # Set the icon based on price change
                        icon = '📈' if price_change >= 0 else '📉'
                    
                    # Create the HTML content
                    html_content = f"""
                    <div style='background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); 
                                border-radius: 16px; 
                                padding: 2rem; 
                                margin: 0 0 1.5rem 0;
                                color: white;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <h1 style='margin: 0 0 0.5rem 0; color: white;'>{final_symbol}</h1>
                                <p style='margin: 0; font-size: 1.1rem; opacity: 0.9;'>{short_name}</p>
                            </div>
                            <div style='text-align: right;'>
                                <div style='font-size: 2.5rem; font-weight: 700;'>${current_price:,.2f}</div>
                                <div style='font-size: 1.1rem; font-weight: 600; background: rgba(255, 255, 255, 0.2); 
                                            display: inline-block; padding: 0.25rem 1rem; border-radius: 20px;'>
                                    {price_change:+.2f} ({percent_change:+.2f}%)
                                    {icon}
                                </div>
                            </div>
                        </div>
                        <div style='margin-top: 1.5rem; display: flex; gap: 1.5rem;'>
                            <div>
                                <div style='font-size: 0.9rem; opacity: 0.8;'>Market Cap</div>
                                <div style='font-size: 1.1rem; font-weight: 600;'>{market_cap}</div>
                            </div>
                            <div>
                                <div style='font-size: 0.9rem; opacity: 0.8;'>P/E Ratio</div>
                                <div style='font-size: 1.1rem; font-weight: 600;'>{pe_ratio}</div>
                            </div>
                            <div>
                                <div style='font-size: 0.9rem; opacity: 0.8;'>52-Week Range</div>
                                <div style='font-size: 1.1rem; font-weight: 600;'>{week_range}</div>
                            </div>
                        </div>
                    </div>
                    """.format(
                        symbol=str(final_symbol),
                        short_name=str(stock_info.get('shortName', 'N/A')),
                        price=float(current_price),
                        change=float(price_change),
                        pct_change=float(percent_change),
                        icon='📈' if price_change >= 0 else '📉',
                        market_cap=market_cap_str,
                        pe_ratio=pe_ratio_str,
                        week_range=week_range
                    )
                    st.markdown(html_content, unsafe_allow_html=True)
                        
                # Display Signal Interpretation
                st.sidebar.header('🔍 Signal Interpretation')
                st.sidebar.metric('Overall Signal', final_signal)
                st.sidebar.metric('Risk Level', signal_details['risk_level'])
                
                # Detailed Signal Information
                with st.sidebar.expander('Signal Details'):
                    st.write(f"**Description:** {signal_details['description']}")
                    st.write(f"**Recommendation:** {signal_details['recommendation']}")
                    
                    st.subheader('Action Points:')
                    for point in signal_details['action_points']:
                        st.markdown(f"- {point}")
                
                # Create columns for the header
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### {final_symbol} Analysis")
                
                # Display the signal with appropriate color
                signal_color = {
                    'Buy': 'green',
                    'Sell': 'red',
                    'Hold': 'orange',
                    'Neutral': 'blue'
                }.get(final_signal, 'black')
                
                with col2:
                    st.markdown("### Signal")
                    st.markdown(f"<h2 style='color: {signal_color};'>{final_signal}</h2>", unsafe_allow_html=True)
                
                # Display risk level with icon
                risk_icons = {
                    'High': '🔴',
                    'Medium': '🟠',
                    'Low': '🟢'
                }
                
                with col3:
                    st.markdown("### Risk")
                    risk_level = signal_details.get('risk_level', 'Medium')
                    st.markdown(f"<h2>{risk_icons.get(risk_level, '⚪')} {risk_level}</h2>", unsafe_allow_html=True)
                
                # Add a divider
                st.markdown("---")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📈 Chart", "📊 Technicals", "📋 Details"])
                
                with tab1:
                    # Create interactive chart
                    interactive_chart = analyzer.create_interactive_chart(stock_data_with_indicators)
                    
                    # Display chart if created successfully
                    if interactive_chart:
                        st.plotly_chart(interactive_chart, use_container_width=True, height=600)
                
                with tab2:
                    # Technical indicators overview
                    st.markdown("### Technical Indicators")
                    
                    # Create columns for indicators
                    col1, col2, col3 = st.columns(3)
                    
                    # RSI
                    with col1:
                        try:
                            rsi = float(stock_data_with_indicators['RSI'].iloc[-1]) if 'RSI' in stock_data_with_indicators.columns and pd.notnull(stock_data_with_indicators['RSI'].iloc[-1]) else None
                            st.metric("RSI", 
                                    f"{rsi:.2f}" if rsi is not None else "N/A",
                                    delta=None,
                                    help="Relative Strength Index (14-day)")
                        except (ValueError, TypeError):
                            st.metric("RSI", "N/A", help="Error calculating RSI")
                    
                    # MACD
                    with col2:
                        try:
                            macd = float(stock_data_with_indicators['MACD'].iloc[-1]) if 'MACD' in stock_data_with_indicators.columns and pd.notnull(stock_data_with_indicators['MACD'].iloc[-1]) else None
                            signal = float(stock_data_with_indicators['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in stock_data_with_indicators.columns and pd.notnull(stock_data_with_indicators['MACD_Signal'].iloc[-1]) else None
                            
                            if macd is not None and signal is not None:
                                st.metric("MACD", 
                                        f"{macd:.2f}",
                                        delta=f"Signal: {signal:.2f}",
                                        help="Moving Average Convergence Divergence")
                            else:
                                st.metric("MACD", "N/A", help="Insufficient data for MACD")
                        except (ValueError, TypeError):
                            st.metric("MACD", "Error", help="Error calculating MACD")
                    
                    # Bollinger Bands
                    with col3:
                        try:
                            # Check if Bollinger Bands columns exist
                            if 'BB_Upper' not in stock_data_with_indicators.columns or 'BB_Lower' not in stock_data_with_indicators.columns:
                                st.metric("Bollinger Bands", "N/A", help="Bollinger Bands data not available")
                            else:
                                upper = stock_data_with_indicators['BB_Upper'].iloc[-1]
                                lower = stock_data_with_indicators['BB_Lower'].iloc[-1]
                                close_price = float(stock_data_with_indicators['Close'].iloc[-1]) if 'Close' in stock_data_with_indicators.columns and pd.notnull(stock_data_with_indicators['Close'].iloc[-1]) else None
                                lower_band = float(lower) if pd.notnull(lower) else None
                                upper_band = float(upper) if pd.notnull(upper) else None
                                
                                if close_price is not None and lower_band is not None and upper_band is not None:
                                    st.metric("Bollinger Bands", 
                                            f"{close_price:.2f}",
                                            delta=f"{lower_band:.2f} - {upper_band:.2f}",
                                            help="Bollinger Bands (20,2)")
                                else:
                                    st.metric("Bollinger Bands", "N/A", help="Insufficient data for Bollinger Bands")
                        except Exception as e:
                            st.metric("Bollinger Bands", "N/A", help="Error calculating Bollinger Bands")
                    
                    # Add more technical indicators in a grid
                    st.markdown("#### Additional Indicators")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sma_50 = float(stock_data_with_indicators['SMA_50'].iloc[-1]) if 'SMA_50' in stock_data_with_indicators.columns else None
                        sma_200 = float(stock_data_with_indicators['SMA_200'].iloc[-1]) if 'SMA_200' in stock_data_with_indicators.columns else None
                        st.metric("SMA 50", f"{sma_50:.2f}" if sma_50 is not None else "N/A")
                        st.metric("SMA 200", f"{sma_200:.2f}" if sma_200 is not None else "N/A")
                    
                    with col2:
                        volume = int(stock_data_with_indicators['Volume'].iloc[-1]) if 'Volume' in stock_data_with_indicators.columns else None
                        atr = float(stock_data_with_indicators['ATR'].iloc[-1]) if 'ATR' in stock_data_with_indicators.columns else None
                        st.metric("Volume", f"{volume:,}" if volume is not None else "N/A")
                        st.metric("ATR", f"{atr:.2f}" if atr is not None else "N/A")
                    
                    with col3:
                        high_52w = float(stock_data_with_indicators['High'].rolling(252).max().iloc[-1]) if 'High' in stock_data_with_indicators.columns else None
                        low_52w = float(stock_data_with_indicators['Low'].rolling(252).min().iloc[-1]) if 'Low' in stock_data_with_indicators.columns else None
                        st.metric("52W High", f"{high_52w:.2f}" if high_52w is not None else "N/A")
                        st.metric("52W Low", f"{low_52w:.2f}" if low_52w is not None else "N/A")
                
                with tab3:
                    # Signal details and recommendations
                    st.markdown("### Signal Details")
                    st.markdown(f"**{signal_details.get('description', 'No description available')}**")
                    
                    st.markdown("#### Recommendation")
                    st.info(signal_details.get('recommendation', 'No specific recommendation available'))
                    
                    st.markdown("#### Action Points")
                    for point in signal_details.get('action_points', []):
                        st.markdown(f"- {point}")
                    
                    # Show recent signals in a table
                    st.markdown("#### Recent Signals")
                    signals_df = stock_data_with_indicators.copy()
                    if 'Date' not in signals_df.columns and signals_df.index.name == 'Date':
                        signals_df = signals_df.reset_index()
                    
                    available_cols = [col for col in ['Date', 'RSI_Signal', 'MACD_Signal', 'BB_Signal'] 
                                   if col in signals_df.columns]
                    
                    if available_cols:
                        st.dataframe(
                            signals_df[available_cols].tail(10).style.applymap(
                                lambda x: 'color: green' if x == 'Buy' else 
                                         ('color: red' if x == 'Sell' else 
                                         ('color: orange' if x == 'Hold' else '')),
                                subset=['RSI_Signal', 'MACD_Signal', 'BB_Signal']
                            ),
                            use_container_width=True
                        )
                    else:
                        st.warning("No signal data available")
                
        except Exception as e:
            st.error(f"Error in stock analysis: {e}")

if __name__ == '__main__':
    main()
