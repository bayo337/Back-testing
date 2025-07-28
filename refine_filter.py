import pandas as pd
import os
from tqdm import tqdm

# --- Configuration ---
SOURCE_FILE = 'filtered_stocks.csv'
OUTPUT_FILE = 'refined_filtered_stocks.csv'
DAILY_DATA_DIR = 'daily-historical-DATA'

# --- New Stricter Filter Criteria ---
MIN_VOLUME = 3_000_000
MIN_PRICE = 1.00
MIN_MOVE_UP_PCT = 0.50 # 50%

# Cache to store loaded daily files to speed up the process
daily_data_cache = {}

def get_close_price(ticker, trade_date_str):
    """
    Fetches the closing price for a given ticker on a specific date.
    Uses a cache to avoid re-reading the same daily file multiple times.
    """
    if trade_date_str in daily_data_cache:
        daily_df = daily_data_cache[trade_date_str]
    else:
        daily_file_path = os.path.join(DAILY_DATA_DIR, f'{trade_date_str}.csv')
        if not os.path.exists(daily_file_path):
            return None
        try:
            daily_df = pd.read_csv(daily_file_path)
            daily_data_cache[trade_date_str] = daily_df
        except Exception as e:
            print(f"Warning: Could not read {daily_file_path}. Error: {e}")
            return None

    stock_data = daily_df[daily_df['ticker'] == ticker]
    if not stock_data.empty:
        return stock_data.iloc[0]['close']
    
    return None

def refine_stock_list():
    """
    Reads the original filtered list and applies stricter criteria
    to create a smaller, more refined list of candidates.
    """
    print(f"Loading original candidates from '{SOURCE_FILE}'...")
    try:
        source_df = pd.read_csv(SOURCE_FILE)
    except FileNotFoundError:
        print(f"Error: Source file '{SOURCE_FILE}' not found. Please ensure it exists.")
        return

    # --- FINAL FIX: Correctly parse the percentage string ---
    # 1. Remove the '%' sign from the string.
    # 2. Convert the remaining string to a numeric type (float).
    # 3. Divide by 100 to get the decimal representation (e.g., 68.77 -> 0.6877).
    source_df['percent_change'] = pd.to_numeric(source_df['percent_change'].str.replace('%', ''), errors='coerce') / 100.0
    
    # We can then drop any rows where the conversion failed.
    source_df.dropna(subset=['percent_change'], inplace=True)

    print(f"Found {len(source_df)} valid candidates after cleaning. Applying stricter filters...")
    
    refined_candidates = []
    
    # Use tqdm for a progress bar
    for _, row in tqdm(source_df.iterrows(), total=source_df.shape[0], desc="Refining List"):
        # 1. Volume Filter
        if row['volume'] < MIN_VOLUME:
            continue

        # 2. Price Filter (requires reading daily file)
        trade_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        close_price = get_close_price(row['ticker'], trade_date)
        
        if close_price is None or close_price <= MIN_PRICE:
            continue
            
        # 3. Stricter "Big Move Up" Filter
        # FIX: Applying this rule to all rows since 'filter_type' column does not exist.
        if row['percent_change'] < MIN_MOVE_UP_PCT:
            continue

        # If all checks pass, add the row to our refined list
        refined_candidates.append(row)

    if not refined_candidates:
        print("No candidates met the stricter criteria. The output file will not be created.")
        return

    refined_df = pd.DataFrame(refined_candidates)
    refined_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- Filtering Complete ---")
    print(f"Original number of candidates: {len(source_df)}")
    print(f"Refined number of candidates:  {len(refined_df)}")
    print(f"Reduction of {len(source_df) - len(refined_df)} candidates ({100 - (len(refined_df)/len(source_df)*100):.2f}% reduction).")
    print(f"\nNew candidate list saved to '{OUTPUT_FILE}'")


if __name__ == '__main__':
    refine_stock_list() 