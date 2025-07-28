import pandas as pd
import os
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

DAILY_DATA_DIR = 'daily-historical-DATA'
OUTPUT_FILE = 'filtered_stocks.csv'

def get_previous_day_data(current_date_str):
    """
    Looks for the previous day's data file and loads it if it exists.
    """
    current_date = date.fromisoformat(current_date_str)
    # We only need to look back a few days to find the last trading day
    for i in range(1, 5):
        prev_date = current_date - timedelta(days=i)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        file_path = os.path.join(DAILY_DATA_DIR, f'{prev_date_str}.csv')
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading previous day's data ({prev_date_str}): {e}")
                return None
    return None

def process_single_file(filename):
    """
    Processes a single day's data file to find stocks that meet the criteria.
    This function is designed to be run in a separate thread.
    """
    current_date_str = filename.replace('.csv', '')
    print(f"Processing data for: {current_date_str}")
    
    file_path = os.path.join(DAILY_DATA_DIR, filename)
    try:
        today_data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Could not read file {filename}. Error: {e}")
        return []

    prev_day_data = get_previous_day_data(current_date_str)
    
    filtered_for_day = []

    for index, row in today_data.iterrows():
        ticker = row['ticker']
        volume = row['volume']
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        
        # --- Filter 1: Gap up 20% with high volume ---
        if prev_day_data is not None and volume >= 1000000:
            prev_day_stock = prev_day_data[prev_day_data['ticker'] == ticker]
            if not prev_day_stock.empty:
                prev_close = prev_day_stock.iloc[0]['close']
                if prev_close > 0:
                    gap_up_pct = (open_price - prev_close) / prev_close
                    if gap_up_pct >= 0.20:
                        result = {'date': current_date_str, 'ticker': ticker, 'condition': 'Gap up > 20%', 'volume': volume, 'percent_change': f"{gap_up_pct:.2%}"}
                        filtered_for_day.append(result)
                        print(f"  - Found: {ticker} (Gap up > 20%)")
        
        # --- Filter 2: Big move up from open (30%+) with high volume ---
        if open_price > 0 and volume >= 1000000:
            move_up_pct = (high_price - open_price) / open_price
            if move_up_pct >= 0.30:
                result = {'date': current_date_str, 'ticker': ticker, 'condition': 'Move up > 30%', 'volume': volume, 'percent_change': f"{move_up_pct:.2%}"}
                filtered_for_day.append(result)
                print(f"  - Found: {ticker} (Move up > 30%)")
        
        # --- Filter 3: Big move down from open (15%+) with high volume ---
        if open_price > 0 and volume >= 1000000:
            move_down_pct = (open_price - low_price) / open_price
            if move_down_pct >= 0.15:
                result = {'date': current_date_str, 'ticker': ticker, 'condition': 'Move down > 15%', 'volume': volume, 'percent_change': f"{move_down_pct:.2%}"}
                filtered_for_day.append(result)
                print(f"  - Found: {ticker} (Move down > 15%)")
                
    return filtered_for_day

def run_stock_filter():
    """
    Iterates through all daily data files using multiple threads and applies filtering logic.
    """
    print("Starting stock filtering process with multithreading...")
    
    all_filtered_stocks = []
    
    try:
        data_files = sorted([f for f in os.listdir(DAILY_DATA_DIR) if f.endswith('.csv')])
    except FileNotFoundError:
        print(f"Error: The directory '{DAILY_DATA_DIR}' was not found.")
        return

    # Using ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all files to the executor
        future_to_file = {executor.submit(process_single_file, filename): filename for filename in data_files}
        
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                # Get the result from the future
                result_list = future.result()
                if result_list:
                    all_filtered_stocks.extend(result_list)
            except Exception as exc:
                print(f'{filename} generated an exception: {exc}')

    if all_filtered_stocks:
        filtered_df = pd.DataFrame(all_filtered_stocks)
        filtered_df.drop_duplicates(subset=['date', 'ticker', 'condition'], inplace=True)
        # Reorder columns for better readability
        filtered_df = filtered_df[['date', 'ticker', 'condition', 'volume', 'percent_change']]
        filtered_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nFiltering complete. Results saved to '{OUTPUT_FILE}'.")
    else:
        print("\nFiltering complete. No stocks met the specified criteria.")

if __name__ == '__main__':
    run_stock_filter() 