# Stock Trend Analyzer

## Overview
An AI-powered tool for analyzing and predicting stock trends using technical indicators.

## Features
- Buy/Sell Signals Generation
- Technical Indicators Analysis
- Real-time Stock Data Fetching
- Interactive Visualization

## 🚀 Deployment Instructions

### Local Development
1. Clone the repository
   ```bash
   git clone https://github.com/Ashish7889/StockTrend-Analyzer-Real-Time-Buy-Sell-Prediction-System.git
   cd StockTrend-Analyzer-Real-Time-Buy-Sell-Prediction-System
   ```
2. Create and activate virtual environment
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app locally
   ```bash
   streamlit run stock_trend_analyzer.py
   ```
   If still you are facing any issues like data exhange not found then try to update the yahoo finance in the command prompt 
make sure you are using yahoo finance latest version only then it works

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and select your repository
4. Set the main file path to `stock_trend_analyzer.py`
5. Click "Deploy!"

### Required Environment Variables
Set these in your Streamlit Cloud settings:
- `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage API key

## Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages

## Dependencies
- Streamlit
- Alpha Vantage API
- Pandas
- Matplotlib
- TA-Lib
