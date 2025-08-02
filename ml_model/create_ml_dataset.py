import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import glob

# --- Configuration ---
# Get the directory where the script is located to build robust paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Updated paths to be absolute, based on the script's location
ANALYSIS_CHART_DIR = os.path.join(SCRIPT_DIR, '..', 'analysis_charts')
ML_DATA_FILE = os.path.join(SCRIPT_DIR, 'ml_training_data.csv')
CANDLE_WINDOW_SIZE = 21 # Define the look-back window for features

def get_trade_indicator_slice(trade_info, num_candles=CANDLE_WINDOW_SIZE):
    """
    Finds and returns a slice of indicator data for a given trade.
    If not enough preceding data is available, it will be padded at the beginning.
    """
    try:
        date_str = pd.to_datetime(trade_info['date']).strftime('%Y-%m-%d')
        ticker = trade_info['ticker']
        indicator_file = os.path.join(ANALYSIS_CHART_DIR, ticker, f'{ticker}_{date_str}_1min_indicators.csv')

        if not os.path.exists(indicator_file):
            return None

        indicator_df = pd.read_csv(indicator_file, index_col='timestamp', parse_dates=True)

        # Improvement: Intelligently use 'entry_time' if available, otherwise fallback to 'exit_time'.
        # This makes the feature extraction more accurate if entry_time is logged.
        time_col = 'exit_time' # Default
        if 'entry_time' in trade_info.index and pd.notna(trade_info['entry_time']):
            time_col = 'entry_time'

        event_time_str = pd.to_datetime(trade_info[time_col], format='%H:%M:%S').time().strftime('%H:%M:%S')
        
        trade_entry_df = indicator_df.between_time('09:30:00', event_time_str)

        if trade_entry_df.empty:
            return None

        # Take the last `num_candles` available
        final_slice = trade_entry_df.iloc[-num_candles:]

        # Improvement: Pad data instead of discarding it.
        # If there aren't enough candles (e.g., for a trade early in the day),
        # pad with NaNs at the beginning. This makes more data usable.
        if len(final_slice) < num_candles:
            padding_size = num_candles - len(final_slice)
            padding_df = pd.DataFrame(np.nan, index=range(padding_size), columns=final_slice.columns)
            final_slice = pd.concat([padding_df, final_slice.reset_index(drop=True)], ignore_index=True)

        return final_slice

    except Exception:
        # Silently fail for a single trade to not stop the whole process
        return None

def engineer_features_for_slice(df):
    """
    Engineers new features for a single trade slice.
    This version creates all new features separately and concatenates them
    at the end to avoid pandas PerformanceWarning about fragmentation.
    """
    df_out = df.copy()
    new_features = {}
    # Ensure only numeric columns are used for calculations
    numeric_cols = df.select_dtypes(include=np.number).columns
    periods = [1, 3, 5]

    # --- 1. Change Over Time (Momentum) Features ---
    for col in numeric_cols:
        for p in periods:
            new_features[f'{col}_diff_{p}'] = df[col].diff(periods=p)
            new_features[f'{col}_pct_change_{p}'] = df[col].pct_change(periods=p)

    # --- 2. Indicator Relationship (Spread) Features ---
    # Check for columns before creating features to avoid errors
    if 'ema_9' in df.columns and 'ema_20' in df.columns:
        new_features['ema_spread_9_20'] = df['ema_9'] - df['ema_20']
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        new_features['stoch_spread_k_d'] = df['stoch_k'] - df['stoch_d']
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_mid' in df.columns:
        bb_width = df['bb_upper'] - df['bb_lower']
        new_features['bb_width'] = bb_width
        new_features['bb_width_normalized'] = bb_width / (df['bb_mid'].replace(0, 1e-9))

    # --- 3. Advanced Volume & Volatility Features ---
    if 'volume' in df.columns:
        volume_sma_20 = df['volume'].rolling(window=20, min_periods=1).mean()
        new_features['volume_sma_20'] = volume_sma_20
        new_features['volume_ratio'] = df['volume'] / (volume_sma_20.replace(0, 1e-9))
        new_features['volume_acceleration'] = df['volume'].diff().diff()
        
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        candle_range = df['high'] - df['low']
        new_features['candle_range'] = candle_range
        new_features['normalized_range'] = candle_range / (df['close'].replace(0, 1e-9))
        if 'volume' in df.columns:
            new_features['effort_vs_result'] = df['volume'] / (candle_range.replace(0, 1e-9))

    # --- 4. Advanced Divergence and Relational Features ---
    if 'rsi' in df.columns and 'high' in df.columns and not df['high'].isnull().all():
        high_max_idx = df['high'].idxmax()
        rsi_max_idx = df['rsi'].idxmax()
        divergence_value = 1 if high_max_idx != rsi_max_idx else 0
        new_features['price_rsi_divergence'] = pd.Series(divergence_value, index=df.index)

    if 'vwap' in df.columns and 'close' in df.columns:
        new_features['close_vs_vwap_ratio'] = df['close'] / (df['vwap'].replace(0, 1e-9))
    
    if 'macd_hist' in df.columns:
        new_features['macd_hist_slope'] = df['macd_hist'].diff()
    
    if 'atr' in df.columns:
        new_features['atr_acceleration'] = df['atr'].diff().diff()

    # Combine all new features into a DataFrame
    features_df = pd.DataFrame(new_features, index=df.index)
    
    # Concatenate original df with the new features df
    df_out = pd.concat([df_out, features_df], axis=1)

    # Clean up NaNs and Infs at the very end
    df_out.replace([np.inf, -np.inf], 0, inplace=True)
    df_out.fillna(0, inplace=True)
    
    return df_out

def flatten_features(df):
    """
    Flattens the time-series data from the indicator slice into a single row.
    """
    df = df.reset_index(drop=True)
    df_numeric = df.select_dtypes(include=np.number)
    flat = df_numeric.unstack().to_frame().T
    
    # Improvement: Make column naming dynamic based on the window size.
    # This makes the logic robust to changes in CANDLE_WINDOW_SIZE.
    t_minus = len(df) - 1
    flat.columns = [f'{col[0]}_t{col[1] - t_minus}' for col in flat.columns]
    return flat

def main():
    """
    Main function to orchestrate the creation of the master ML dataset.
    This version dynamically finds all trade log files in the project.
    """
    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    print("-" * 60)
    print(f"Searching for all 'trade_logs_csv' directories in: {project_root}")
    print("-" * 60)

    backtest_files = []
    # Walk through all directories starting from the project root
    for root, dirs, _ in os.walk(project_root):
        if 'trade_logs_csv' in dirs:
            trade_log_dir = os.path.join(root, 'trade_logs_csv')
            search_pattern = os.path.join(trade_log_dir, '*_1min.csv')
            found_files = glob.glob(search_pattern)
            if found_files:
                print(f"  -> Found {len(found_files)} files in: {trade_log_dir}")
                backtest_files.extend(found_files)

    if not backtest_files:
        print("\nError: Could not find any '*_1min.csv' files in any 'trade_logs_csv' subdirectories.")
        return

    backtest_files = sorted(list(set(backtest_files)))
    
    print("-" * 60)
    print(f"\nFound a total of {len(backtest_files)} unique backtest files to process.")
    
    master_df = pd.concat(
        [pd.read_csv(file) for file in tqdm(backtest_files, desc="Reading all trade files")],
        ignore_index=True
    )

    master_df.drop_duplicates(subset=['ticker', 'date', 'entry_attempt'], inplace=True)
    print(f"Collected {len(master_f)} unique trades after deduplication.")
    
    winners = master_df[master_df['pnl_in_dollars'] > 0]
    losers = master_df[master_df['pnl_in_dollars'] <= 0] # Include break-even as losers

    print(f"Total winners available: {len(winners)}, Total losers available: {len(losers)}")

    num_samples = min(len(winners), len(losers))
    if num_samples == 0:
        print("Error: No winning or losing trades found to create a dataset.")
        return

    print(f"Creating a balanced dataset with {num_samples} winners and {num_samples} losers.")
    
    # Improvement: Use random sampling instead of nlargest/nsmallest.
    # This creates a more representative dataset by not just focusing on extreme outcomes,
    # leading to a better-generalized model.
    balanced_winners = winners.sample(n=num_samples, random_state=42)
    balanced_losers = losers.sample(n=num_samples, random_state=42)
    
    combined_trades = pd.concat([balanced_winners, balanced_losers], ignore_index=True)

    feature_list = []
    for _, trade in tqdm(combined_trades.iterrows(), total=len(combined_trades), desc="Engineering Features"):
        indicator_slice = get_trade_indicator_slice(trade)
        if indicator_slice is not None:
            engineered_slice = engineer_features_for_slice(indicator_slice)
            features = flatten_features(engineered_slice)
            features['target'] = 1 if trade['pnl_in_dollars'] > 0 else 0
            feature_list.append(features)

    if not feature_list:
        print("Could not generate any features. Exiting.")
        return

    final_ml_dataset = pd.concat(feature_list, ignore_index=True)

    final_ml_dataset.to_csv(ML_DATA_FILE, index=False)
    print(f"\nMachine learning dataset created successfully!")
    print(f" -> Saved to: {ML_DATA_FILE}")
    print(f" -> Total samples: {len(final_ml_dataset)}")
    print(f" -> Number of features: {len(final_ml_dataset.columns) - 1}")

if __name__ == '__main__':
    main()
