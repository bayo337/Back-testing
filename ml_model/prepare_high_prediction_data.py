import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import timedelta
from functools import partial

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DAILY_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'daily-historical-DATA'))
ANALYSIS_CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'analysis_charts'))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'ml_training_data_high_prediction.csv')

# Window sizes for feature extraction (in minutes)
PRE_PEAK_WINDOW_SIZE = 5    # Number of candles before the peak to analyze
POST_PEAK_WINDOW_SIZE = 5   # Number of candles after the peak to analyze

# Dataset configuration
MAX_SAMPLES_PER_CLASS = 6000  # Maximum samples for winners/losers in the final dataset

# --- Global Variable for Multiprocessing ---
# This variable holds the lookup table for previous day's closing prices
# It's loaded once and shared across all worker processes
prev_close_lookup = None

# --- Helper Functions ---

def init_worker():
    """
    Initializer for each worker process in the multiprocessing pool.
    
    This function is called once for each worker process and loads the 
    previous close lookup table into the global scope of that worker.
    This avoids passing large data structures between processes.
    """
    global prev_close_lookup
    if prev_close_lookup is None:
        print("Initializing worker...")
        prev_close_lookup = get_previous_closes()

def get_previous_closes():
    """
    Loads all daily data to create a fast lookup dictionary for the previous day's close.
    
    This function:
    1. Reads all CSV files from the daily data directory
    2. Calculates the previous day's closing price for each stock
    3. Creates a fast lookup table indexed by (ticker, date)
    
    Returns:
        pd.Series: A multi-index series with (ticker, date_str) as index and 
                   previous_close as values
    """
    print("Loading all daily data to calculate previous closes...")
    all_files = [os.path.join(DAILY_DATA_DIR, f) for f in os.listdir(DAILY_DATA_DIR) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"No daily data files found in '{DAILY_DATA_DIR}'.")

    # Load all daily files with progress bar for better user feedback
    df_list = [pd.read_csv(f, low_memory=False) for f in tqdm(all_files, desc="Loading daily files")]
    df = pd.concat(df_list, ignore_index=True)
    
    # Clean and prepare the data
    df.dropna(subset=['ticker', 'close'], inplace=True)
    df['date'] = pd.to_datetime(df['window_start'], unit='ns').dt.date
    df.sort_values(by=['ticker', 'date'], inplace=True)
    
    # Calculate previous day's close for each stock
    df['previous_close'] = df.groupby('ticker')['close'].shift(1)
    df.dropna(subset=['previous_close'], inplace=True)
    
    # Convert date to string format for indexing
    df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    
    # Create the lookup table
    prev_close_lookup = df.set_index(['ticker', 'date_str'])['previous_close']
    print(f"Created previous close lookup table for {len(prev_close_lookup)} stock-day entries.")
    return prev_close_lookup

def create_features_from_window(df_window):
    """
    Creates aggregated features from a time window of indicator data.
    
    This function takes a DataFrame containing indicator data for a specific
    time window (e.g., 5 candles before the peak) and calculates statistical
    features like mean, std, min, max, and slope for each indicator.
    
    Args:
        df_window (pd.DataFrame): DataFrame containing OHLCV and indicator data
                                 for a specific time window
    
    Returns:
        dict: Dictionary containing aggregated features for each indicator
              Features include: _mean, _std, _min, _max, _slope
    """
    features = {}
    if df_window.empty:
        return features

    # Get all numeric columns (indicators)
    indicator_cols = df_window.select_dtypes(include=np.number).columns.tolist()
    time_series = np.arange(len(df_window))

    for col in indicator_cols:
        # Clean the data by handling infinities and NaN values
        data = df_window[col].replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(method='ffill')
        if data.isnull().all() or len(data.dropna()) < 1:
            continue
        
        # Calculate basic statistical features
        features[f'{col}_mean'] = data.mean()
        features[f'{col}_std'] = data.std()
        features[f'{col}_min'] = data.min()
        features[f'{col}_max'] = data.max()
        
        # Calculate slope (trend direction) using linear regression
        try:
            valid_data = data.dropna()
            if len(valid_data) > 1:
                 valid_time = time_series[valid_data.index.map(df_window.index.get_loc)]
                 slope, _ = np.polyfit(valid_time, valid_data, 1)
                 features[f'{col}_slope'] = slope
            else:
                 features[f'{col}_slope'] = 0
        except (np.linalg.LinAlgError, ValueError):
            features[f'{col}_slope'] = 0

    return features

def create_temporal_features(df, peak_index, date_str):
    """
    Creates temporal features to analyze when peaks occur and their timing patterns.
    
    This function analyzes the timing of the peak relative to market hours and
    creates features that help identify optimal trading windows.
    
    Args:
        df (pd.DataFrame): Full day's data with timestamp column
        peak_index (int): Index of the peak candle in the DataFrame
        date_str (str): Date string in 'YYYY-MM-DD' format
    
    Returns:
        dict: Dictionary containing temporal features including:
              - minutes_since_market_open: Time from market open to peak
              - minutes_until_market_close: Time from peak to market close
              - peak_time_percentage: Peak time as percentage of trading day
              - time_category: Categorical time period (0-3)
              - Binary features for each time category
    """
    features = {}
    
    # Define market hours
    market_open = pd.to_datetime(date_str + ' 09:30:00').tz_localize('America/New_York')
    market_close = pd.to_datetime(date_str + ' 16:00:00').tz_localize('America/New_York')
    
    peak_time = df.iloc[peak_index]['timestamp']
    
    # Calculate time-based features
    minutes_since_open = (peak_time - market_open).total_seconds() / 60
    features['minutes_since_market_open'] = minutes_since_open
    
    minutes_until_close = (market_close - peak_time).total_seconds() / 60
    features['minutes_until_market_close'] = minutes_until_close
    
    # Peak time as percentage of trading day (390 minutes = 6.5 hours)
    total_trading_minutes = 390
    features['peak_time_percentage'] = (minutes_since_open / total_trading_minutes) * 100
    
    # Categorize time of day
    peak_hour = peak_time.hour
    peak_minute = peak_time.minute
    
    # Time categories:
    # 0: Early morning (9:30-10:30)
    # 1: Mid morning (10:30-12:00)
    # 2: Early afternoon (12:00-14:00)
    # 3: Late afternoon (14:00-16:00)
    if peak_hour == 9 or (peak_hour == 10 and peak_minute <= 30):
        features['time_category'] = 0  # Early morning
    elif (peak_hour == 10 and peak_minute > 30) or peak_hour == 11:
        features['time_category'] = 1  # Mid morning
    elif peak_hour == 12 or peak_hour == 13:
        features['time_category'] = 2  # Early afternoon
    else:
        features['time_category'] = 3  # Late afternoon
    
    # Create binary features for each time category
    features['is_early_morning'] = 1 if features['time_category'] == 0 else 0
    features['is_mid_morning'] = 1 if features['time_category'] == 1 else 0
    features['is_early_afternoon'] = 1 if features['time_category'] == 2 else 0
    features['is_late_afternoon'] = 1 if features['time_category'] == 3 else 0
    
    return features

def create_magnitude_features(df, peak_index, prev_close, trigger_time):
    """
    Creates magnitude-based features to analyze price movement patterns.
    
    This function analyzes the size and characteristics of the price move
    to help identify which magnitude ranges lead to the best fade opportunities.
    
    Args:
        df (pd.DataFrame): Full day's data with OHLCV columns
        peak_index (int): Index of the peak candle
        prev_close (float): Previous day's closing price
        trigger_time (pd.Timestamp): Time when the initial trigger occurred
    
    Returns:
        dict: Dictionary containing magnitude features including:
              - peak_move_percentage: Total move from prev_close to peak
              - trigger_move_percentage: Move when the initial trigger hit
              - additional_move_after_trigger: Additional move after trigger
              - magnitude_category: Categorical magnitude (0-3)
              - Binary features for each magnitude category
              - volume_weighted_move: Move weighted by volume
    """
    features = {}
    
    peak_high = df.iloc[peak_index]['high']
    trigger_high = df[df['timestamp'] >= trigger_time]['high'].iloc[0]
    
    # Calculate magnitude features
    features['peak_move_percentage'] = ((peak_high - prev_close) / prev_close) * 100
    features['trigger_move_percentage'] = ((trigger_high - prev_close) / prev_close) * 100
    features['additional_move_after_trigger'] = features['peak_move_percentage'] - features['trigger_move_percentage']
    
    # Categorize magnitude of the move
    # 0: Small move (< 40%)
    # 1: Medium move (40-60%)
    # 2: Large move (60-80%)
    # 3: Extreme move (> 80%)
    if features['peak_move_percentage'] < 40:
        features['magnitude_category'] = 0  # Small move
    elif features['peak_move_percentage'] < 60:
        features['magnitude_category'] = 1  # Medium move
    elif features['peak_move_percentage'] < 80:
        features['magnitude_category'] = 2  # Large move
    else:
        features['magnitude_category'] = 3  # Extreme move
    
    # Create binary features for each magnitude category
    features['is_small_move'] = 1 if features['magnitude_category'] == 0 else 0
    features['is_medium_move'] = 1 if features['magnitude_category'] == 1 else 0
    features['is_large_move'] = 1 if features['magnitude_category'] == 2 else 0
    features['is_extreme_move'] = 1 if features['magnitude_category'] == 3 else 0
    
    # Volume-weighted magnitude (normalized by 1M shares)
    volume_before_peak = df.loc[:peak_index, 'volume'].sum()
    features['volume_weighted_move'] = features['peak_move_percentage'] * (volume_before_peak / 1000000)
    
    return features

# --- Main Processing Function ---

def process_intraday_file(file_path):
    """
    Processes a single intraday file to identify and analyze peak events.
    
    This function implements the core logic for identifying profitable fade opportunities:
    1. Looks for stocks that move 20% or more during the scan window (9:20-9:40 AM)
    2. Identifies the peak high after the trigger
    3. Calculates the subsequent drop percentage
    4. Extracts features from pre-peak, at-peak, and post-peak windows
    5. Adds temporal and magnitude analysis features
    
    Args:
        file_path (str): Path to the 1-minute indicator CSV file
    
    Returns:
        dict or None: Dictionary containing all features and metadata for the event,
                     or None if the event doesn't meet criteria
    """
    try:
        # Parse ticker and date from filename (format: TICKER_YYYY-MM-DD_1min_indicators.csv)
        parts = os.path.basename(file_path).split('_')
        ticker, date_str = parts[0], parts[1]

        # Get previous day's closing price
        try:
            prev_close = prev_close_lookup.loc[(ticker, date_str)]
        except KeyError:
            return None

        # Load and prepare the intraday data
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        if df.empty or 'high' not in df.columns: 
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')

        # Calculate percentage move from previous close
        df['move_pct'] = ((df['high'] - prev_close) / prev_close) * 100
        
        # Define scan window: 9:20 AM to 9:40 AM (20 minutes before market open to 10 minutes after)
        market_open_time = pd.to_datetime(date_str + ' 09:30:00').tz_localize('America/New_York')
        scan_start_time = market_open_time - timedelta(minutes=20)
        scan_end_time = market_open_time + timedelta(minutes=10)
        
        # Look for 20% moves during the scan window
        scan_df = df[(df['timestamp'] >= scan_start_time) & (df['timestamp'] <= scan_end_time)]
        
        # Find the trigger time (first 20% move)
        trigger_time = None
        if not scan_df.empty and scan_df['move_pct'].max() >= 20.0:
            trigger_row = scan_df[scan_df['move_pct'] >= 20.0].iloc[0]
            trigger_time = trigger_row['timestamp']
        else:
            # Fallback: check the entire day if no trigger found in scan window
            full_day_triggers = df[df['move_pct'] >= 20.0]
            if not full_day_triggers.empty:
                trigger_time = full_day_triggers.iloc[0]['timestamp']

        if trigger_time is None: 
            return None

        # Find the peak high after the trigger
        df_after_trigger = df[df['timestamp'] >= trigger_time]
        if df_after_trigger.empty: 
            return None
            
        peak_index = df_after_trigger['high'].idxmax()
        peak_high = df.loc[peak_index, 'high']
        
        # Apply filtering criteria
        if peak_high <= 0.50:  # Minimum price requirement
            return None
        if df.loc[:peak_index, 'volume'].sum() <= 1000000:  # Minimum volume requirement
            return None

        # Extract data windows for feature creation
        # Use max(0, ...) to handle peaks near the start of the day
        pre_peak_window = df.iloc[max(0, peak_index - PRE_PEAK_WINDOW_SIZE) : peak_index]
        post_peak_window = df.iloc[peak_index + 1 : peak_index + 1 + POST_PEAK_WINDOW_SIZE]
        at_peak_window = df.iloc[peak_index:peak_index+1]  # Single peak candle

        # Calculate the drop percentage (target variable)
        df_after_peak = df.loc[peak_index + 1:]
        subsequent_low = df_after_peak['low'].min() if not df_after_peak.empty else peak_high
        drop_pct = ((peak_high - subsequent_low) / peak_high) * 100 if peak_high > 0 else 0

        # Create all feature sets
        pre_peak_features_raw = create_features_from_window(pre_peak_window)
        pre_peak_features = {f'{k}_pre': v for k, v in pre_peak_features_raw.items()}

        at_peak_features_raw = create_features_from_window(at_peak_window)
        at_peak_features = {f'{k}_peak': v for k, v in at_peak_features_raw.items()}

        post_peak_features_raw = create_features_from_window(post_peak_window)
        post_peak_features = {f'{k}_post': v for k, v in post_peak_features_raw.items()}

        # Add temporal and magnitude analysis features
        temporal_features = create_temporal_features(df, peak_index, date_str)
        magnitude_features = create_magnitude_features(df, peak_index, prev_close, trigger_time)

        # Combine all features into final data row
        final_data_row = {
            'ticker': ticker,
            'date': date_str,
            'drop_pct': drop_pct,  # Target variable for ML
            'peak_index': peak_index,
        }
        
        # Add all feature sets
        final_data_row.update(pre_peak_features)
        final_data_row.update(at_peak_features)
        final_data_row.update(post_peak_features)
        final_data_row.update(temporal_features)
        final_data_row.update(magnitude_features)
        
        return final_data_row
        
    except Exception:
        # Silently handle errors to avoid stopping the entire process
        return None

# --- Main Orchestrator ---

def main():
    """
    Main function that orchestrates the entire data preparation process.
    
    This function:
    1. Scans for all 1-minute indicator files
    2. Processes each file to find valid peak events
    3. Analyzes temporal and magnitude patterns
    4. Creates a balanced dataset for machine learning
    5. Saves the final dataset with all features
    """
    print("Scanning for all 1-minute indicator files...")
    all_files = [os.path.join(root, file) for root, _, files in os.walk(ANALYSIS_CHARTS_DIR) 
                 for file in files if file.endswith('_1min_indicators.csv')]
    if not all_files:
        print("No 1-minute indicator files found. Exiting.")
        return
    print(f"Found {len(all_files)} total intraday files to process.")

    print("\nProcessing files to find valid peak events...")
    # Use multiprocessing to speed up processing, but limit CPU usage to 70%
    # This leaves 30% for other tasks
    cpu_count = os.cpu_count()
    max_workers = max(1, int(cpu_count * 0.7))  # Use 70% of available CPUs
    print(f"Using {max_workers} worker processes (70% of {cpu_count} available CPUs)")
    
    with multiprocessing.Pool(processes=max_workers, initializer=init_worker) as pool:
        all_results = list(tqdm(pool.imap(process_intraday_file, all_files), total=len(all_files)))
    
    # Filter out None results
    valid_results = [r for r in all_results if r is not None and r]
    print(f"\nFound {len(valid_results)} valid events that met all criteria.")

    if not valid_results:
        print("No events met the criteria. Exiting.")
        return
        
    # Create DataFrame and clean data
    df = pd.DataFrame(valid_results)
    df.dropna(subset=['drop_pct'], inplace=True)  # Ensure drop_pct is valid

    # NEW: Analyze temporal and magnitude patterns
    print("\n=== Temporal and Magnitude Analysis ===")
    
    # Time analysis
    if 'minutes_since_market_open' in df.columns:
        avg_peak_time = df['minutes_since_market_open'].mean()
        print(f"Average peak occurs {avg_peak_time:.1f} minutes after market open")
        
        # Find best performing time categories
        time_performance = df.groupby('time_category')['drop_pct'].agg(['mean', 'count'])
        print("\nDrop percentage by time category:")
        print(time_performance)
        
        best_time_category = time_performance['mean'].idxmax()
        print(f"Best performing time category: {best_time_category}")
    
    # Magnitude analysis
    if 'peak_move_percentage' in df.columns:
        magnitude_performance = df.groupby('magnitude_category')['drop_pct'].agg(['mean', 'count'])
        print("\nDrop percentage by magnitude category:")
        print(magnitude_performance)
        
        best_magnitude_category = magnitude_performance['mean'].idxmax()
        print(f"Best performing magnitude category: {best_magnitude_category}")
        
        # Find optimal move percentage range
        move_ranges = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
        for low, high in move_ranges:
            mask = (df['peak_move_percentage'] >= low) & (df['peak_move_percentage'] < high)
            if mask.sum() > 0:
                avg_drop = df[mask]['drop_pct'].mean()
                count = mask.sum()
                print(f"Move {low}-{high}%: Avg drop {avg_drop:.2f}% ({count} samples)")

    # Create balanced dataset for machine learning
    if len(df) < 2:
        print("Not enough data to create a balanced dataset. Exiting.")
        return

    # Use median split for balanced classes
    median_drop = df['drop_pct'].median()
    print(f"\nMedian drop percentage is {median_drop:.2f}%. Using this to stratify the dataset.")

    # Split into high drops (winners) and low drops (losers)
    high_drops_population = df[df['drop_pct'] >= median_drop]
    low_drops_population = df[df['drop_pct'] < median_drop]

    # Determine sample size (use the smaller of the two populations)
    n_samples = min(MAX_SAMPLES_PER_CLASS, len(high_drops_population), len(low_drops_population))

    if n_samples == 0:
        print("Could not create balanced classes. Not enough samples in one of the categories.")
        return

    print(f"Selecting {n_samples} random samples from the high drop group and {n_samples} from the low drop group.")
    
    # Create balanced dataset with random sampling
    high_drops_sample = high_drops_population.sample(n=n_samples, random_state=42)
    high_drops_sample['target'] = 1  # Winners (big drops)
    
    low_drops_sample = low_drops_population.sample(n=n_samples, random_state=42)
    low_drops_sample['target'] = 0  # Losers (small drops)
    
    # Combine and shuffle the final dataset
    final_df = pd.concat([high_drops_sample, low_drops_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the final dataset
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully created and saved ML dataset to '{OUTPUT_FILE}' with {len(final_df)} samples.")
    print(f"Dataset now includes temporal and magnitude features for better pattern recognition.")

if __name__ == '__main__':
    # Set multiprocessing start method for Windows compatibility
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn')
    main()
