import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime, timedelta
from functools import partial

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DAILY_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'daily-historical-DATA'))
ANALYSIS_CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'analysis_charts'))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'ml_training_data.csv')
PRE_PEAK_WINDOW_SIZE = 20
POST_PEAK_WINDOW_SIZE = 5

# --- Global Variable for Multiprocessing ---
# We use a global variable to hold the large lookup table.
# This avoids passing it as an argument, which can cause issues on Windows with large objects.
prev_close_lookup = None

# --- Helper Functions ---

def init_worker():
    """
    Initializer for each worker process. Loads the lookup data into the global scope of the worker.
    """
    print("Initializing worker...")
    global prev_close_lookup
    prev_close_lookup = get_previous_closes()

def get_previous_closes():
    """
    Loads all daily data to create a fast lookup dictionary for the previous day's close.
    """
    print("Loading all daily data to calculate previous closes...")
    all_files = [os.path.join(DAILY_DATA_DIR, f) for f in os.listdir(DAILY_DATA_DIR) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"No daily data files found in '{DAILY_DATA_DIR}'.")

    df = pd.concat([pd.read_csv(f, low_memory=False) for f in all_files], ignore_index=True)
    df.dropna(subset=['ticker', 'close'], inplace=True)
    df['date'] = pd.to_datetime(df['window_start'], unit='ns').dt.date
    df.sort_values(by=['ticker', 'date'], inplace=True)
    df['previous_close'] = df.groupby('ticker')['close'].shift(1)
    df.dropna(subset=['previous_close'], inplace=True)
    
    df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    
    # Create a multi-index series for fast lookup
    prev_close_lookup = df.set_index(['ticker', 'date_str'])['previous_close']
    print(f"Created previous close lookup table for {len(prev_close_lookup)} stock-day entries.")
    return prev_close_lookup

def create_features_from_window(df_window):
    """
    From a data window, create aggregated features.
    """
    features = {}
    indicator_cols = df_window.select_dtypes(include=np.number).columns.tolist()
    time_series = np.arange(len(df_window))

    for col in indicator_cols:
        data = df_window[col].replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(method='ffill')
        if data.isnull().any():
            continue
        
        features[f'{col}_mean'] = data.mean()
        features[f'{col}_std'] = data.std()
        features[f'{col}_min'] = data.min()
        features[f'{col}_max'] = data.max()
        
        try:
            valid_data = data.dropna()
            if len(valid_data) > 1:
                 # Use index from valid_data to align with time_series
                 valid_time = time_series[valid_data.index.map(df_window.index.get_loc)]
                 slope, _ = np.polyfit(valid_time, valid_data, 1)
                 features[f'{col}_slope'] = slope
            else:
                 features[f'{col}_slope'] = 0
        except (np.linalg.LinAlgError, ValueError):
            features[f'{col}_slope'] = 0

    return features

# --- Main Processing Function ---

def process_intraday_file(file_path):
    """
    Processes a single intraday file against all the new criteria.
    """
    try:
        # 1. Parse info from filename
        parts = os.path.basename(file_path).split('_')
        ticker, date_str = parts[0], parts[1]

        # 2. Get previous close
        try:
            # The worker will now access the global prev_close_lookup
            prev_close = prev_close_lookup.loc[(ticker, date_str)]
        except KeyError:
            return None

        # 3. Load intraday data
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        if df.empty or 'high' not in df.columns: return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')

        # 4. Check conditions
        df['move_pct'] = ((df['high'] - prev_close) / prev_close) * 100
        
        # --- MODIFIED: Expanded pre-market scan window ---
        market_open_time = pd.to_datetime(date_str + ' 09:30:00').tz_localize('America/New_York')
        pre_market_start_time = market_open_time - timedelta(minutes=10) # Start scan at 9:20 AM
        ten_mins_after_open = market_open_time + timedelta(minutes=10) # End scan at 9:40 AM
        
        # Scan window is now 9:20 AM to 9:40 AM
        pre_market_scan_df = df[(df['timestamp'] >= pre_market_start_time) & (df['timestamp'] <= ten_mins_after_open)]
        
        trigger_time = None
        if not pre_market_scan_df.empty and pre_market_scan_df['move_pct'].max() >= 40.0:
            trigger_row = pre_market_scan_df[pre_market_scan_df['move_pct'] >= 40.0].iloc[0]
            trigger_time = trigger_row['timestamp']
        else:
            # Fallback to check the entire day if the pre-market condition is not met
            full_day_triggers = df[df['move_pct'] >= 40.0]
            if not full_day_triggers.empty:
                trigger_time = full_day_triggers.iloc[0]['timestamp']

        if trigger_time is None: return None

        # 5. Find peak *after* the trigger
        df_after_trigger = df[df['timestamp'] >= trigger_time]
        if df_after_trigger.empty: return None
            
        peak_index = df_after_trigger['high'].idxmax()
        peak_high = df.loc[peak_index, 'high']
        
        # 6. Check volume and price conditions
        if peak_high <= 0.50: return None
        if df.loc[:peak_index, 'volume'].sum() <= 1000000: return None
            
        # 7. Check pre and post window availability
        if peak_index < PRE_PEAK_WINDOW_SIZE or (peak_index + POST_PEAK_WINDOW_SIZE + 1) > len(df):
             return None

        # 8. All conditions met, extract data
        df_after_peak = df.loc[peak_index + 1:]
        subsequent_low = df_after_peak['low'].min() if not df_after_peak.empty else peak_high
        drop_pct = ((peak_high - subsequent_low) / peak_high) * 100 if peak_high > 0 else 0

        # --- MODIFIED: Create features and add prefixes directly here ---
        pre_peak_window = df.iloc[peak_index - PRE_PEAK_WINDOW_SIZE : peak_index]
        pre_peak_features_raw = create_features_from_window(pre_peak_window)
        pre_peak_features = {f'{k}_pre': v for k, v in pre_peak_features_raw.items()}

        at_peak_window = df.iloc[peak_index:peak_index+1]
        at_peak_features_raw = create_features_from_window(at_peak_window)
        at_peak_features = {f'{k}_peak': v for k, v in at_peak_features_raw.items()}

        post_peak_window = df.iloc[peak_index+1 : peak_index+1+POST_PEAK_WINDOW_SIZE]
        post_peak_features_raw = create_features_from_window(post_peak_window)
        post_peak_features = {f'{k}_post': v for k, v in post_peak_features_raw.items()}


        # --- Combine all data for the final row ---
        final_data_row = {
            'ticker': ticker,
            'date': date_str,
            'drop_pct': drop_pct,
            'peak_index': peak_index,
        }
        
        final_data_row.update(pre_peak_features)
        final_data_row.update(at_peak_features)
        final_data_row.update(post_peak_features)
        
        return final_data_row
        
    except Exception:
        return None

# --- Main Orchestrator ---

def main():
    # The lookup data is now loaded by each worker in the initializer.
    # prev_close_lookup = get_previous_closes()

    print("Scanning for all 1-minute indicator files...")
    all_files = [os.path.join(root, file) for root, _, files in os.walk(ANALYSIS_CHARTS_DIR) for file in files if file.endswith('_1min_indicators.csv')]
    if not all_files:
        print("No 1-minute indicator files found. Exiting.")
        return
    print(f"Found {len(all_files)} total intraday files to process.")

    # We no longer need to pass the lookup table as an argument here
    # worker_func = partial(process_intraday_file, prev_close_lookup=prev_close_lookup)

    print("\nProcessing files to find valid events based on new criteria...")
    # The initializer sets up the global variable for each worker by calling the function itself
    with multiprocessing.Pool(initializer=init_worker) as pool:
        all_results = list(tqdm(pool.imap(process_intraday_file, all_files), total=len(all_files)))
    
    valid_results = [r for r in all_results if r is not None]
    print(f"Found {len(valid_results)} events that met all criteria.")

    if not valid_results:
        print("No events met the criteria. Exiting.")
        return
        
    df = pd.DataFrame(valid_results)
    df = df.sort_values(by='drop_pct', ascending=False).reset_index(drop=True)

    n_samples = min(6000, len(df) // 2)
    if n_samples == 0 and len(df) > 0: n_samples = 1

    print(f"\nDataset has {len(df)} valid events. Will select top and bottom {n_samples} for the final dataset.")
    top_drops = df.head(n_samples).copy()
    top_drops['target'] = 1
    bottom_drops = df.tail(n_samples).copy()
    bottom_drops['target'] = 0
    final_df = pd.concat([top_drops, bottom_drops])
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully created and saved ML dataset to '{OUTPUT_FILE}' with {len(final_df)} samples.")

if __name__ == '__main__':
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn')
    main() 