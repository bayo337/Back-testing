import pandas as pd
import os
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- Configuration ---
DAILY_DATA_DIR = 'daily-historical-DATA'
OUTPUT_FILE = 'final_filtered_stocks.csv'
START_DATE = '2020-08-03'

# --- Filter Criteria ---
MIN_TOTAL_GAIN_PCT = 0.50  # 50%
MIN_VOLUME = 3_000_000
MIN_PRICE = 1.00

# Cache to store previous day's dataframes to avoid re-reading
prev_day_cache = {}

def get_previous_day_df(current_date):
    """Fetches the dataframe for the previous trading day, using a cache."""
    # Look back up to 5 days to find the last valid trading day's file
    for i in range(1, 6):
        prev_date = current_date - timedelta(days=i)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        
        if prev_date_str in prev_day_cache:
            return prev_day_cache[prev_date_str]

        daily_file_path = os.path.join(DAILY_DATA_DIR, f'{prev_date_str}.csv')
        if os.path.exists(daily_file_path):
            try:
                df = pd.read_csv(daily_file_path).set_index('ticker')
                prev_day_cache[prev_date_str] = df # Store in cache
                return df
            except Exception:
                return None
    return None

def process_day(date):
    """
    Processes a single day's data file to find stocks that meet the combined gain criteria.
    """
    date_str = date.strftime('%Y-%m-%d')
    current_day_path = os.path.join(DAILY_DATA_DIR, f'{date_str}.csv')

    if not os.path.exists(current_day_path):
        return []

    try:
        current_df = pd.read_csv(current_day_path)
    except Exception:
        return []

    prev_day_df = get_previous_day_df(date)
    if prev_day_df is None:
        return []

    # Merge current day's data with previous day's close for efficient calculation
    merged_df = current_df.join(prev_day_df[['close']], on='ticker', rsuffix='_prev')
    merged_df.dropna(subset=['close_prev'], inplace=True) # Drop stocks that didn't trade the previous day

    # Apply volume and price filters first to reduce the dataset
    filtered = merged_df[
        (merged_df['volume'] >= MIN_VOLUME) &
        (merged_df['close'] >= MIN_PRICE)
    ].copy()

    # Calculate total gain from previous close to current day's high
    filtered['total_gain'] = (filtered['high'] - filtered['close_prev']) / filtered['close_prev']

    # Final filter for stocks that meet the total gain criteria
    final_selection = filtered[filtered['total_gain'] >= MIN_TOTAL_GAIN_PCT]

    if final_selection.empty:
        return []

    results = []
    for _, row in final_selection.iterrows():
        results.append({
            'date': date_str,
            'ticker': row['ticker'],
            'volume': row['volume'],
            'percent_change': row['total_gain'] * 100 # Store as percentage
        })
    return results

def run_combined_filter():
    """
    Runs the new combined filtering logic across all daily data files.
    """
    all_dates = pd.to_datetime(pd.date_range(start=START_DATE, end=pd.Timestamp.today()))
    
    print(f"Scanning {len(all_dates)} days for stocks meeting the combined gain criteria...")
    
    all_candidates = []
    
    # Use ThreadPoolExecutor to process days in parallel (I/O bound task)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Wrap with tqdm for a progress bar
        results = list(tqdm(executor.map(process_day, all_dates), total=len(all_dates), desc="Filtering Days"))
        
        for daily_results in results:
            if daily_results:
                all_candidates.extend(daily_results)

    if not all_candidates:
        print("No candidates met the criteria. The output file will not be created.")
        return
        
    final_df = pd.DataFrame(all_candidates).sort_values(by='date').reset_index(drop=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- Combined Filtering Complete ---")
    print(f"Found {len(final_df)} candidates meeting the combined gain criteria.")
    print(f"New candidate list saved to '{OUTPUT_FILE}'")


if __name__ == '__main__':
    run_combined_filter() 