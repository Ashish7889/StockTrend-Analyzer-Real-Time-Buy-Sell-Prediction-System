"""
Test script for stock data fetching functionality
"""
from stock_trend_analyzer import AdvancedStockAnalyzer

def test_stock_fetch(symbols):
    """Test stock data fetching for given symbols"""
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Testing symbol: {symbol}")
        print(f"{'='*50}")
        
        try:
            analyzer = AdvancedStockAnalyzer(symbol)
            data = analyzer.fetch_stock_data(period='1mo')
            
            if not data.empty:
                print(f"✅ Successfully fetched data for {symbol}")
                print(f"Data shape: {data.shape}")
                print("Latest data points:")
                print(data.tail(2))
            else:
                print(f"❌ No data returned for {symbol}")
                
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")

if __name__ == "__main__":
    # Test with various stock symbols
    test_symbols = [
        'AAPL',         # US stock
        'MSFT',         # US stock
        'RELIANCE.NS',  # Indian stock (NSE)
        'TCS.NS',       # Indian stock (NSE)
        'BHP.AX',       # Australian stock
        'HSBA.L',       # UK stock
        '005930.KS'     # Samsung Korea
    ]
    
    print("Starting stock data fetch test...")
    test_stock_fetch(test_symbols)
    print("\nTest completed.")
