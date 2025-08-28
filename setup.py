from setuptools import setup, find_packages

setup(
    name="stock-trend-analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'streamlit==1.29.0',
        'yfinance==0.2.31',
        'pandas==2.1.4',
        'numpy==1.26.2',
        'plotly==5.18.0',
        'scikit-learn==1.3.2',
        'requests==2.31.0',
        'python-dateutil==2.8.2',
        'pytz==2023.3',
        'protobuf==3.20.3',
    ],
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="A stock trend analyzer with buy/sell signals",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ashish7889/StockTrend-Analyzer-Real-Time-Buy-Sell-Prediction-System",
)
