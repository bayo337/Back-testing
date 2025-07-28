import pandas as pd
import numpy as np
import os
import mplfinance as mpf
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
FILTERED_STOCKS_FILE = 'filtered_stocks.csv'
MINUTE_DATA_DIR = 'historical-DATA'
OUTPUT_DIR = 'analysis_charts'

def calculate_indicators(df):
    """Calculates all the required technical indicators for the dataframe."""
    # VWAP (Volume-Weighted Average Price)
    df['vwap'] = (df['volume'] * (df['high'] + df['low']) / 2).cumsum() / df['volume'].cumsum()

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_prev_close = (df['high'] - df['close'].shift()).abs()
    low_prev_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Rate of Change (ROC)
    df['roc'] = ((df['close'] - df['close'].shift(14)) / df['close'].shift(14)) * 100

    # SMAs and EMAs
    for period in [5, 9, 13, 20]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    
    return df

def generate_full_charts(df, ticker, date_str, interval_str):
    """Generates and saves a candlestick chart with all indicators."""
    ticker_output_dir = os.path.join(OUTPUT_DIR, ticker)
    os.makedirs(ticker_output_dir, exist_ok=True)
    
    chart_file = os.path.join(ticker_output_dir, f'{ticker}_{date_str}_{interval_str}_full.png')
    
    # Define which indicators to plot
    plots = [
        # Main Panel (Price, VWAP, SMAs, Bollinger Bands)
        mpf.make_addplot(df['vwap'], color='g', panel=0, ylabel='VWAP'),
        mpf.make_addplot(df[['sma_9', 'sma_20']], panel=0),
        mpf.make_addplot(df[['bb_upper', 'bb_lower']], color='gray', linestyle='--', panel=0),
        
        # Other Panels
        mpf.make_addplot(df['obv'], color='purple', panel=1, ylabel='OBV'),
        
        mpf.make_addplot(df['macd'], color='blue', panel=2, ylabel='MACD'),
        mpf.make_addplot(df['macd_signal'], color='orange', panel=2),
        mpf.make_addplot(df['macd_hist'], type='bar', color='gray', panel=2, alpha=0.5),

        mpf.make_addplot(df['stoch_k'], color='blue', panel=3, ylabel='Stoch'),
        mpf.make_addplot(df['stoch_d'], color='orange', panel=3),

        mpf.make_addplot(df['rsi'], color='r', panel=4, ylabel='RSI'),
        mpf.make_addplot(df['atr'], color='c', panel=5, ylabel='ATR'),
    ]

    # Define vertical lines for market open and close
    vlines = dict(vlines=[f'{date_str} 09:30', f'{date_str} 16:00'], colors='gray', linestyle='--')

    print(f"  - Generating full {interval_str} chart for {ticker} on {date_str}...")
    mpf.plot(df, type='candle', style='yahoo',
             title=f'{ticker} - {date_str} ({interval_str} Full)',
             volume=True, addplot=plots,
             panel_ratios=(10, 3, 3, 3, 3, 2),
             figscale=2.5,
             savefig=chart_file,
             vlines=vlines,
             warn_too_much_data=1500) # Suppress the "too much data" warning
    print(f"  - Chart saved to: {chart_file}")

def generate_simple_charts(df, ticker, date_str, interval_str):
    """Generates and saves a simple candlestick chart with only price and volume."""
    ticker_output_dir = os.path.join(OUTPUT_DIR, ticker)
    os.makedirs(ticker_output_dir, exist_ok=True)
    
    chart_file = os.path.join(ticker_output_dir, f'{ticker}_{date_str}_{interval_str}_simple.png')
    
    # Define vertical lines for market open and close
    vlines = dict(vlines=[f'{date_str} 09:30', f'{date_str} 16:00'], colors='gray', linestyle='--')

    print(f"  - Generating simple {interval_str} chart for {ticker} on {date_str}...")
    mpf.plot(df, type='candle', style='yahoo',
             title=f'{ticker} - {date_str} ({interval_str} Simple)',
             volume=True,
             figscale=1.5,
             savefig=chart_file,
             vlines=vlines,
             warn_too_much_data=1500) # Suppress the "too much data" warning
    print(f"  - Chart saved to: {chart_file}")

def save_indicator_data(df, ticker, date_str, interval_str):
    """Saves the dataframe with all calculated indicators to a CSV file."""
    ticker_output_dir = os.path.join(OUTPUT_DIR, ticker)
    os.makedirs(ticker_output_dir, exist_ok=True)
    
    csv_file = os.path.join(ticker_output_dir, f'{ticker}_{date_str}_{interval_str}_indicators.csv')
    print(f"  - Saving indicator data to: {csv_file}")
    df.to_csv(csv_file)
    
def process_stock_date(job_info):
    """
    Handles all processing for a single stock on a single date.
    Designed to be run in a separate process.
    """
    ticker, date_str = job_info
    print(f"\nAnalyzing {ticker} for date: {date_str}...")

    minute_file = os.path.join(MINUTE_DATA_DIR, f'{date_str}.csv')
    if not os.path.exists(minute_file):
        print(f"  - Minute data file not found for {date_str}. Skipping.")
        return

    try:
        minute_df = pd.read_csv(minute_file)
        stock_df = minute_df[minute_df['ticker'] == ticker].copy()
        
        if stock_df.empty:
            print(f"  - No data for ticker {ticker} in file {date_str}.csv. Skipping.")
            return

        # Correctly handle the timezone conversion
        stock_df['timestamp'] = pd.to_datetime(stock_df['window_start'], utc=True).dt.tz_convert('US/Eastern')
        stock_df = stock_df.set_index('timestamp')
        stock_df = stock_df.drop(columns=['window_start'])

        # --- Process 1-Minute Data ---
        df_1min = calculate_indicators(stock_df.copy())
        generate_full_charts(df_1min, ticker, date_str, '1min')
        generate_simple_charts(df_1min, ticker, date_str, '1min')
        save_indicator_data(df_1min, ticker, date_str, '1min')
        
        # --- Process 5-Minute Data ---
        df_5min = stock_df.resample('5T').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        if not df_5min.empty:
            df_5min = calculate_indicators(df_5min.copy())
            generate_full_charts(df_5min, ticker, date_str, '5min')
            generate_simple_charts(df_5min, ticker, date_str, '5min')
            save_indicator_data(df_5min, ticker, date_str, '5min')

    except Exception as e:
        print(f"An error occurred while processing {ticker} on {date_str}: {e}")

def run_chart_generator():
    """Main function to read filtered stocks and generate all outputs in parallel."""
    try:
        filtered_df = pd.read_csv(FILTERED_STOCKS_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{FILTERED_STOCKS_FILE}' was not found. Please run the filter script first.")
        return

    # Create a list of all jobs to be processed
    jobs = []
    # Using unique tickers and their corresponding dates
    for index, row in filtered_df.drop_duplicates(subset=['ticker', 'date']).iterrows():
        jobs.append((row['ticker'], row['date']))

    # Run all jobs in the list
    if jobs:
        print(f"Starting analysis for {len(jobs)} stock/date events...")
        # Use ProcessPoolExecutor to run jobs in parallel
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(process_stock_date, jobs)

    print("\n\n--- Analysis complete! ---")

if __name__ == '__main__':
    run_chart_generator() 