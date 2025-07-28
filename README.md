# Backtesting Project

A comprehensive backtesting framework for trading strategies including gap and fade, volatility-based strategies, and machine learning models.

## 🚀 Features

- **Gap and Fade Strategy**: Backtesting for gap up/down trading strategies
- **Volatility Backtesting**: Advanced volatility-based trading algorithms
- **Account-Based Testing**: Position sizing and risk management
- **Data Download**: Automated historical data fetching from Polygon API
- **Performance Analysis**: Comprehensive trade analysis and reporting
- **Chart Generation**: Automated chart creation for strategy performance

## 📋 Requirements

- Python 3.8 or higher
- Polygon API key (for data download)
- 8GB+ RAM recommended for large datasets

## 🛠️ Installation

### Quick Start

1. **Clone or download the project**
   ```bash
   git clone <your-repo-url>
   cd backtesting
   ```

2. **Run the installation script**
   ```bash
   python install_requirements.py
   ```

3. **Set up your API key**
   - Create a `polygon.env` file in the project root
   - Add your Polygon API key: `POLYGON_API_KEY=your_api_key_here`

### Manual Installation

If you prefer manual installation:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
backtesting/
├── GAP_FADE/                    # Gap and fade strategy files
│   ├── volatility_backtester.py # Main volatility backtester
│   ├── acoutbasedtest.py       # Account-based testing
│   └── equity_charts/          # Generated equity curves
├── ml_model/                    # Machine learning models
├── historical-DATA/             # Minute-level historical data
├── daily-historical-DATA/       # Daily historical data
├── analysis_charts/             # Analysis and performance charts
├── backtester.py               # Basic gap and fade backtester
├── download_data.py            # Data download script
├── stock_filter.py             # Stock filtering utilities
└── requirements.txt            # Python dependencies
```

## 🎯 Usage

### 1. Download Historical Data

```bash
python download_data.py
```

### 2. Run Basic Gap and Fade Backtest

```bash
python backtester.py
```

### 3. Run Volatility Backtesting

```bash
python GAP_FADE/volatility_backtester.py
```

### 4. Generate Analysis Charts

```bash
python generate_analysis_charts.py
```

## 📊 Key Scripts

### `volatility_backtester.py`
- Advanced volatility-based trading strategy
- Position sizing and risk management
- Multi-timeframe analysis
- Performance reporting

### `backtester.py`
- Basic gap and fade strategy implementation
- Simple entry/exit logic
- Daily trade simulation

### `download_data.py`
- Downloads historical minute and daily data
- Uses Polygon API for data fetching
- Handles rate limiting and error recovery

### `stock_filter.py`
- Filters stocks based on various criteria
- Volume, price, and volatility filters
- Creates filtered stock lists for backtesting

## 🔧 Configuration

### Risk Management Settings
- `RISK_PER_TRADE`: Dollar amount risked per trade
- `PROFIT_TARGET_R`: Profit target in R-multiples
- `INITIAL_ACCOUNT_BALANCE`: Starting account balance

### Data Settings
- `MINUTE_DATA_DIR`: Directory for minute-level data
- `DAILY_DATA_DIR`: Directory for daily data
- `FILTERED_STOCKS_FILE`: File containing filtered stock list

## 📈 Output Files

The backtesting generates several output files:

- **Trade Logs**: CSV files with detailed trade information
- **Equity Charts**: PNG files showing account equity curves
- **Performance Reports**: Text files with strategy statistics
- **Analysis Charts**: Various performance visualization charts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is for educational and research purposes. Please ensure compliance with your local regulations when using for actual trading.

## ⚠️ Disclaimer

This software is for educational purposes only. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `polygon.env` file is properly formatted
2. **Memory Issues**: Reduce the number of stocks or time period for large datasets
3. **Data Missing**: Run `download_data.py` to fetch required historical data

### Getting Help

- Check the error messages in the console output
- Verify all required files are present
- Ensure Python version is 3.8 or higher

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code comments for specific functions
3. Create an issue in the repository

---

**Happy Backtesting! 📈** 