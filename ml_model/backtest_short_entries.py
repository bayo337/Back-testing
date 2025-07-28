import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import joblib
from functools import partial
import multiprocessing

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ANALYSIS_CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'analysis_charts'))
MODEL_FILE = os.path.join(SCRIPT_DIR, 'trade_win_predictor.joblib')
DATA_FILE = os.path.join(SCRIPT_DIR, 'ml_training_data.csv')
PRE_PEAK_WINDOW_SIZE = 20

# --- Feature Creation Function (copied from prepare_high_prediction_data.py) ---
# This function is essential for generating features for the model in real-time.
def create_features_from_window(df_window, prefix):
    features = {}
    indicator_cols = df_window.select_dtypes(include=np.number).columns.tolist()
    time_series = np.arange(len(df_window))

    for col in indicator_cols:
        data = df_window[col].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
        if data.isnull().any(): continue
        
        features[f'{col}_{prefix}_mean'] = data.mean()
        features[f'{col}_{prefix}_std'] = data.std()
        features[f'{col}_{prefix}_max'] = data.max()
        features[f'{col}_{prefix}_min'] = data.min()
        
        # Calculate slope if there's more than one point
        if len(data) > 1:
            slope, _ = np.polyfit(time_series, data, 1)
            features[f'{col}_{prefix}_slope'] = slope
        else:
            features[f'{col}_{prefix}_slope'] = 0

    return features

# --- Backtesting Functions ---

def backtest_pre_high_strategy(event, model):
    """
    Backtests a pre-high entry strategy using the ML model.
    Enters short if the model predicts a 'Winner' with high confidence.
    """
    try:
        ticker = event['ticker']
        date_str = event['date']
        peak_index = event['peak_index']
        
        file_path = os.path.join(ANALYSIS_CHARTS_DIR, ticker, f"{ticker}_{date_str}_1min_indicators.csv")
        if not os.path.exists(file_path): return None
        
        df = pd.read_csv(file_path)
        
        # Simulate checking the 60 minutes before the actual peak
        for i in range(peak_index - 60, peak_index):
            if i < PRE_PEAK_WINDOW_SIZE: continue

            window_df = df.iloc[i - PRE_PEAK_WINDOW_SIZE : i]
            
            # We need to generate all 3 sets of features to match model's training data
            features_pre = create_features_from_window(window_df, 'pre')
            features_peak = {f'{k}_peak': v for k, v in df.iloc[i].to_dict().items() if isinstance(v, (int, float))}
            # Post-peak features will be blank/zero as they haven't happened yet
            placeholder_post_cols = [c.replace('_pre', '_post') for c in features_pre.keys()]
            features_post = {k: 0 for k in placeholder_post_cols}
            
            features = {**features_pre, **features_peak, **features_post}
            
            # Ensure all required columns for the model are present
            model_features = pd.DataFrame([features], columns=model.feature_names_in_)
            model_features.fillna(0, inplace=True)

            # Get model's prediction probability
            prediction_proba = model.predict_proba(model_features)[0]
            
            # ENTRY SIGNAL: High confidence (>99%) of being a "Winner" (class 1)
            if prediction_proba[1] > 0.99:
                entry_price = df.loc[i, 'close']
                
                # Evaluate the trade's outcome
                trade_outcome_df = df.iloc[i+1:]
                
                # SUCCESS: Price drops 10% before it rises 5%
                profit_target = entry_price * 0.90
                stop_loss = entry_price * 1.05
                
                profit_hit_time = trade_outcome_df[trade_outcome_df['low'] <= profit_target].index.min()
                loss_hit_time = trade_outcome_df[trade_outcome_df['high'] >= stop_loss].index.min()
                
                # Check if profit_hit_time is not NaN and occurred before loss_hit_time (or if loss was never hit)
                if pd.notna(profit_hit_time) and (pd.isna(loss_hit_time) or profit_hit_time < loss_hit_time):
                    return 'success'
                else:
                    return 'failure'
        
        return 'no_signal'
        
    except Exception:
        return None


def backtest_post_high_strategy(event, debug=False):
    """
    Backtests a post-high entry strategy.
    Enters short after the peak when the price breaks the structure of the peak candle.
    """
    try:
        ticker = event['ticker']
        date_str = event['date']
        peak_index = event['peak_index']

        file_path = os.path.join(ANALYSIS_CHARTS_DIR, ticker, f"{ticker}_{date_str}_1min_indicators.csv")
        if not os.path.exists(file_path): return None
        
        df = pd.read_csv(file_path)

        # --- MODIFIED: Improved Post-High Strategy ---
        # Find the candle that made the high
        peak_candle_low = df.loc[peak_index, 'low']
        
        if debug: print(f"\n--- Debugging {ticker} on {date_str} ---")
        if debug: print(f"Peak Index: {peak_index}, Peak Candle Low: {peak_candle_low:.4f}")

        # Look at data from the peak onwards
        post_peak_df = df.iloc[peak_index + 1:].copy()
        
        # ENTRY SIGNAL: First close below the low of the peak candle
        entry_candle = post_peak_df[post_peak_df['close'] < peak_candle_low]
        
        if not entry_candle.empty:
            entry_index = entry_candle.index[0]
            entry_price = df.loc[entry_index, 'close']
            
            if debug: print(f"Signal Found at index {entry_index}. Entry Price: {entry_price:.4f}")

            # Evaluate the trade's outcome from the *next* candle onwards
            trade_outcome_df = df.iloc[entry_index + 1:]
            
            # SUCCESS: Price drops 10% before it rises 5%
            profit_target = entry_price * 0.90
            stop_loss = entry_price * 1.05
            
            if debug: print(f"Profit Target: {profit_target:.4f}, Stop Loss: {stop_loss:.4f}")

            for idx, row in trade_outcome_df.iterrows():
                if debug: print(f"  - Checking index {idx}: High={row['high']:.4f}, Low={row['low']:.4f}")
                if row['low'] <= profit_target:
                    if debug: print(f"  --> SUCCESS: Low hit profit target.")
                    return 'success'
                if row['high'] >= stop_loss:
                    if debug: print(f"  --> FAILURE: High hit stop loss.")
                    return 'failure'

            if debug: print("  --> No outcome reached by end of day.")
            return 'failure' # End of day is a failure
        
        if debug: print("No signal found (close below peak low).")
        return 'no_signal'

    except Exception as e:
        if debug: print(f"An error occurred: {e}")
        return None

# --- Main Orchestrator ---

def main():
    print("Loading model and data...")
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
        print("Error: Model or data file not found. Please run the preparation and training scripts first.")
        return
        
    model = joblib.load(MODEL_FILE)
    events_df = pd.read_csv(DATA_FILE)
    
    # We only need the 'Winner' events for backtesting short entries
    winner_events = events_df[events_df['target'] == 1].to_dict('records')
    
    if not winner_events:
        print("No 'Winner' events found in the dataset to backtest. Exiting.")
        return

    print(f"\nBacktesting strategies on {len(winner_events)} 'Winner' events...")

    # --- DEBUGGING: Run on a single event first ---
    print("\n--- Running in DEBUG mode on a single event ---")
    backtest_post_high_strategy(winner_events[0], debug=True)
    
    # --- Run Post-High Backtest ---
    print("\n[1] Running Improved Post-High (Break of Structure) Strategy Backtest...")
    # Temporarily disable multiprocessing for easier debugging if needed
    # post_high_results = [backtest_post_high_strategy(event) for event in tqdm(winner_events)]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        post_high_results = list(tqdm(pool.imap(backtest_post_high_strategy, winner_events), total=len(winner_events)))
        
    post_high_success = post_high_results.count('success')
    post_high_failure = post_high_results.count('failure')
    post_high_no_signal = post_high_results.count('no_signal')

    print("\n--- Post-High Strategy Results ---")
    if (post_high_success + post_high_failure) > 0:
        post_high_success_rate = (post_high_success / (post_high_success + post_high_failure)) * 100
        print(f"Signals Found: {post_high_success + post_high_failure}/{len(winner_events)}")
        print(f"Successful Trades: {post_high_success}")
        print(f"Failed Trades: {post_high_failure}")
        print(f"Success Rate: {post_high_success_rate:.2f}%")
    else:
        print("No signals were generated by the post-high strategy.")
        
    # --- Run Pre-High Backtest ---
    print("\n[2] Running Pre-High (ML-based) Strategy Backtest...")
    # This requires passing the model to the function, so we use partial
    pre_high_func = partial(backtest_pre_high_strategy, model=model)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
       pre_high_results = list(tqdm(pool.imap(pre_high_func, winner_events), total=len(winner_events)))
        
    pre_high_success = pre_high_results.count('success')
    pre_high_failure = pre_high_results.count('failure')
    pre_high_no_signal = pre_high_results.count('no_signal')

    print("\n--- Pre-High Strategy Results ---")
    if (pre_high_success + pre_high_failure) > 0:
       pre_high_success_rate = (pre_high_success / (pre_high_success + pre_high_failure)) * 100
       print(f"Signals Found: {pre_high_success + pre_high_failure}/{len(winner_events)}")
       print(f"Successful Trades: {pre_high_success}")
       print(f"Failed Trades: {pre_high_failure}")
       print(f"Success Rate: {pre_high_success_rate:.2f}%")
    else:
       print("No signals were generated by the pre-high strategy.")
 
if __name__ == '__main__':
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main() 