import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import glob

# --- Configuration ---
# Get the directory where the script is located to build robust paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Updated paths to be absolute, based on the script's location
TRADE_LOGS_DIR = os.path.join(SCRIPT_DIR, '..', 'GAP_FADE', 'trade_logs_csv')
ANALYSIS_CHART_DIR = os.path.join(SCRIPT_DIR, '..', 'analysis_charts')
ML_DATA_FILE = os.path.join(SCRIPT_DIR, 'ml_training_data.csv')
NUM_SAMPLES_PER_FILE = 1000 # Number of top/bottom trades to pull from each backtest file

def get_trade_indicator_slice(trade_info, num_candles=21):
    """
    Finds and returns a slice of indicator data for a given trade.
    """
    try:
        date_str = pd.to_datetime(trade_info['date']).strftime('%Y-%m-%d')
        ticker = trade_info['ticker']
        indicator_file = os.path.join(ANALYSIS_CHART_DIR, ticker, f'{ticker}_{date_str}_1min_indicators.csv')

        if not os.path.exists(indicator_file): return None

        indicator_df = pd.read_csv(indicator_file)
        indicator_df['timestamp'] = pd.to_datetime(indicator_df['timestamp'])
        indicator_df = indicator_df.set_index('timestamp')
        
        entry_time_str = pd.to_datetime(trade_info['exit_time'], format='%H:%M:%S').time().strftime('%H:%M:%S')
        trade_entry_df = indicator_df.between_time('09:30:00', entry_time_str)

        if len(trade_entry_df) < num_candles: return None

        entry_candle_index = len(trade_entry_df) - 1
        start_index = entry_candle_index - (num_candles - 1)
        return trade_entry_df.iloc[start_index : entry_candle_index + 1]

    except Exception:
        return None

def engineer_features_for_slice(df):
    """
    Engineers new features for a single 21-candle trade slice.
    This version creates all new features separately and concatenates them
    at the end to avoid pandas PerformanceWarning about fragmentation.
    """
    df_out = df.copy()
    new_features = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    periods = [1, 3, 5]

    # --- 1. Change Over Time (Momentum) Features ---
    for col in numeric_cols:
        for p in periods:
            new_features[f'{col}_diff_{p}'] = df[col].diff(periods=p)
            new_features[f'{col}_pct_change_{p}'] = df[col].pct_change(periods=p)

    # --- 2. Indicator Relationship (Spread) Features ---
    if 'ema_9' in df.columns and 'ema_20' in df.columns:
        new_features['ema_spread_9_20'] = df['ema_9'] - df['ema_20']
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        new_features['stoch_spread_k_d'] = df['stoch_k'] - df['stoch_d']
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        new_features['bb_width'] = df['bb_upper'] - df['bb_lower']
        new_features['bb_width_normalized'] = new_features['bb_width'] / (df['bb_mid'] + 1e-9)

    # --- 3. Advanced Volume & Volatility Features (Transaction Proxies) ---
    volume_sma_20 = df['volume'].rolling(window=20, min_periods=1).mean()
    new_features['volume_sma_20'] = volume_sma_20
    new_features['volume_ratio'] = df['volume'] / (volume_sma_20 + 1e-9)

    candle_range = df['high'] - df['low']
    new_features['candle_range'] = candle_range
    new_features['normalized_range'] = candle_range / (df['close'] + 1e-9)
    new_features['effort_vs_result'] = df['volume'] / (candle_range + 1e-9)

    # --- 4. Advanced Divergence and Relational Features ---
    if 'rsi' in df.columns:
        high_max_idx = df['high'].idxmax()
        rsi_max_idx = df['rsi'].idxmax()
        # Create a Series with the same index as the original df to ensure proper alignment
        divergence_value = 1 if high_max_idx != rsi_max_idx else 0
        new_features['price_rsi_divergence'] = pd.Series(divergence_value, index=df.index)

    if 'vwap' in df.columns:
        new_features['close_vs_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-9)
    
    new_features['volume_acceleration'] = df['volume'].diff().diff()

    if 'macd_hist' in df.columns:
        new_features['macd_hist_slope'] = df['macd_hist'].diff()
    
    if 'atr' in df.columns:
        new_features['atr_acceleration'] = df['atr'].diff().diff()

    # Combine all new features into a DataFrame
    features_df = pd.DataFrame(new_features)
    
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
    # Adjust for the new 21-candle window (t-20 to t-0)
    flat.columns = [f'{col[0]}_t{col[1] - 20}' for col in flat.columns]
    return flat

def main():
    """
    Main function to orchestrate the creation of the master ML dataset.
    This new version dynamically finds all trade log files in the project.
    """
    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    print("-" * 60)
    print(f"Searching for all 'trade_logs_csv' directories in: {project_root}")
    print("-" * 60)

    backtest_files = []
    # Walk through all directories starting from the project root
    for root, dirs, files in os.walk(project_root):
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

    # Remove duplicates just in case and sort for consistency
    backtest_files = sorted(list(set(backtest_files)))
    
    print("-" * 60)
    print(f"\nFound a total of {len(backtest_files)} unique backtest files to process.")
    
    master_df = pd.concat(
        [pd.read_csv(file) for file in tqdm(backtest_files, desc="Reading all trade files")],
        ignore_index=True
    )

    # --- Critical Step: Deduplicate ---
    master_df.drop_duplicates(subset=['ticker', 'date', 'entry_attempt'], inplace=True)
    print(f"Collected {len(master_df)} unique trades after deduplication.")
    
    # Separate winners and losers from the master list
    winners = master_df[master_df['pnl_in_dollars'] > 0]
    losers = master_df[master_df['pnl_in_dollars'] < 0]

    print(f"Total winners available: {len(winners)}, Total losers available: {len(losers)}")

    # We can now create a balanced dataset for training
    # Let's aim for a large, balanced dataset, taking the smaller of the two groups as the limit
    num_samples = min(len(winners), len(losers))
    if num_samples == 0:
        print("Error: No winning or losing trades found to create a dataset.")
        return

    print(f"Creating a balanced dataset with {num_samples} winners and {num_samples} losers.")
    
    balanced_winners = winners.nlargest(num_samples, 'pnl_in_dollars')
    balanced_losers = losers.nsmallest(num_samples, 'pnl_in_dollars')
    
    combined_trades = pd.concat([balanced_winners, balanced_losers], ignore_index=True)

    # Process all unique trades to get their indicator data and engineer features
    feature_list = []
    for _, trade in tqdm(combined_trades.iterrows(), total=len(combined_trades), desc="Engineering Features"):
        indicator_slice = get_trade_indicator_slice(trade)
        if indicator_slice is not None:
            engineered_slice = engineer_features_for_slice(indicator_slice)
            features = flatten_features(engineered_slice)
            # Add the pnl as the eventual target
            features['target'] = 1 if trade['pnl_in_dollars'] > 0 else 0
            feature_list.append(features)

    if not feature_list:
        print("Could not generate any features. Exiting.")
        return

    # Combine all feature rows into the final dataset
    final_ml_dataset = pd.concat(feature_list, ignore_index=True)

    # Save the final dataset
    final_ml_dataset.to_csv(ML_DATA_FILE, index=False)
    print(f"\nMachine learning dataset created successfully!")
    print(f" -> Saved to: {ML_DATA_FILE}")
    print(f" -> Total samples: {len(final_ml_dataset)}")
    print(f" -> Number of features: {len(final_ml_dataset.columns) - 1}")

if __name__ == '__main__':
    main() 