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
            
            # Create subplots
            fig = sp.make_subplots(
                rows=4, cols=1, 
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
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=plot_data.index,
                    open=plot_data['Open'],
                    high=plot_data['High'],
                    low=plot_data['Low'],
                    close=plot_data['Close'],
                    name='Price',
                    increasing_line_color='#2ecc71',
                    decreasing_line_color='#e74c3c'
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
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data['RSI'],
                        name='RSI',
                        line=dict(color='#3498db', width=1.5)
                    ),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_hline(
                    y=70, line_dash="dash", 
                    line_color="red", 
                    opacity=0.5, 
                    row=2, col=1,
                    annotation_text="Overbought"
                )
                fig.add_hline(
                    y=30, line_dash="dash", 
                    line_color="green", 
                    opacity=0.5,
                    row=2, col=1,
                    annotation_text="Oversold"
                )
            
            # Add MACD if it exists
            if all(col in plot_data.columns for col in ['MACD', 'Signal_Line']):
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data['MACD'],
                        name='MACD',
                        line=dict(color='#3498db', width=1.5)
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data['Signal_Line'],
                        name='Signal Line',
                        line=dict(color='#f39c12', width=1.5)
                    ),
                    row=3, col=1
                )
                
                # Add histogram
                plot_data['Histogram'] = plot_data['MACD'] - plot_data['Signal_Line']
                colors = ['#2ecc71' if val >= 0 else '#e74c3c' 
                         for val in plot_data['Histogram']]
                fig.add_trace(
                    go.Bar(
                        x=plot_data.index,
                        y=plot_data['Histogram'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=3, col=1
                )
            
            # Add Volume
            if 'Volume' in plot_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=plot_data.index,
                        y=plot_data['Volume'],
                        name='Volume',
                        marker_color='#3498db',
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=4, col=1
                )
            
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
    st.title('Advanced Stock Trend Analyzer 📈')
    
    # Predefined Stock Lists
    stock_lists = {
        'NYSE': [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 
            'FB', 'TSLA', 'NFLX', 'INTC', 'CSCO'
        ],
        'NASDAQ': [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 
            'META', 'TSLA', 'NFLX', 'ADBE', 'PYPL'
        ],
        'NSE': [
            'INFY.NS', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 
            'ICICIBANK.NS', 'HINDUNILVR.NS', 'AXISBANK.NS', 
            'MARUTI.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS'
        ],
        'BSE': [
            'INFY.BO', 'RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 
            'ICICIBANK.BO', 'HINDUNILVR.BO', 'AXISBANK.BO', 
            'MARUTI.BO', 'BAJFINANCE.BO', 'KOTAKBANK.BO'
        ]
    }
    
    # Sidebar for stock selection
    st.sidebar.header('Stock Analysis Parameters')
    
    # Exchange selection
    selected_exchange = st.sidebar.selectbox('Select Exchange', list(stock_lists.keys()))
    
    # Stock symbol selection
    stock_symbol = st.sidebar.selectbox(
        'Select Stock Symbol', 
        stock_lists[selected_exchange]
    )
    
    # Manual input option
    st.sidebar.markdown('---')
    manual_symbol = st.sidebar.text_input(
        'Or Enter Custom Stock Symbol', 
        placeholder='e.g., AAPL, INFY.NS'
    )
    
    # Use manual symbol if provided
    final_symbol = manual_symbol if manual_symbol else stock_symbol
    
    # Analyze button
    if st.sidebar.button('Analyze Stock'):
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
                
                # Display final trading signal
                st.subheader(f'Trading Signal: {final_signal}')
                
                # Create interactive chart
                interactive_chart = analyzer.create_interactive_chart(stock_data_with_indicators)
                
                # Display chart if created successfully
                if interactive_chart:
                    st.plotly_chart(interactive_chart, use_container_width=True)
                
                # Display recent signals
                st.subheader('Recent Signals')
                # Reset index to make Date a column if it's the index
                signals_df = stock_data_with_indicators.copy()
                if 'Date' not in signals_df.columns and signals_df.index.name == 'Date':
                    signals_df = signals_df.reset_index()
                
                # Select only the required columns if they exist
                available_cols = [col for col in ['Date', 'RSI_Signal', 'MACD_Signal', 'BB_Signal'] 
                               if col in signals_df.columns]
                if available_cols:
                    st.dataframe(signals_df[available_cols].tail(10))
                else:
                    st.warning("No signal data available")
                
        except Exception as e:
            st.error(f"Error in stock analysis: {e}")

if __name__ == '__main__':
    main()
