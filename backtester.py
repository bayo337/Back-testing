import pandas as pd
import os
from datetime import date, timedelta

DATA_DIR = 'historical-DATA'

def get_previous_trading_day(current_date_str):
    """
    Finds the most recent previous day for which we have data.
    """
    current_date = date.fromisoformat(current_date_str)
    for i in range(1, 10): # Look back up to 10 days
        prev_date = current_date - timedelta(days=i)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        file_path = os.path.join(DATA_DIR, f'{prev_date_str}.csv')
        if os.path.exists(file_path):
            return prev_date_str
    return None

def load_day_data(date_str):
    """
    Loads all minute-level data for a given day.
    The date_str should be in 'YYYY-MM-DD' format.
    """
    file_path = os.path.join(DATA_DIR, f'{date_str}.csv')
    if not os.path.exists(file_path):
        print(f"Data file not found for date: {date_str}")
        return None
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # The 'window_start' is in nanoseconds. Let's convert it to a datetime object.
    df['timestamp'] = pd.to_datetime(df['window_start'])
    df = df.set_index('timestamp')
    
    # We can drop the original window_start column if we want
    df = df.drop(columns=['window_start'])
    
    print("Data loaded successfully.")
    return df

def run_gap_and_fade_strategy(today_data, ticker_symbol, prev_day_close, gap_threshold=0.01):
    """
    Runs the Gap and Fade strategy.
    - Shorts on a gap up.
    - Buys on a gap down.
    """
    print(f"\n--- Running Gap and Fade Strategy for {ticker_symbol} ---")
    
    stock_today = today_data[today_data['ticker'] == ticker_symbol].copy()
    if stock_today.empty:
        print(f"No data for {ticker_symbol} on the current day.")
        return

    # Get the opening price of the day
    opening_price = stock_today.iloc[0]['open']
    
    # Calculate the gap
    gap_percentage = (opening_price - prev_day_close) / prev_day_close
    
    print(f"Previous Day's Close: {prev_day_close:.2f}")
    print(f"Today's Open: {opening_price:.2f}")
    print(f"Gap: {gap_percentage:.2%}")

    position = 0  # -1 for short, 1 for long, 0 for no position
    
    if gap_percentage > gap_threshold:
        print("Gap Up detected. Fading the gap (shorting).")
        position = -1
    elif gap_percentage < -gap_threshold:
        print("Gap Down detected. Fading the gap (buying).")
        position = 1
    else:
        print("No significant gap detected. No trade.")
        return

    # --- Simulate the Trade ---
    # We enter the trade at the opening price.
    entry_price = opening_price
    
    # We exit the trade at the closing price of the day.
    exit_price = stock_today.iloc[-1]['close']
    
    # Calculate the result of the trade
    if position == 1: # Long position
        pnl = exit_price - entry_price
    elif position == -1: # Short position
        pnl = entry_price - exit_price
    else:
        pnl = 0

    print(f"Entered trade at: {entry_price:.2f}")
    print(f"Exited trade at: {exit_price:.2f}")
    print(f"Profit/Loss for the day: {pnl:.2f} per share")
    
    print("--- Strategy Simulation Finished ---")


if __name__ == '__main__':
    # --- Strategy Backtest Example ---
    
    # 1. Define the day we want to trade
    trade_date = '2020-08-14'
    
    # 2. Find the previous trading day to get the close
    prev_trade_date = get_previous_trading_day(trade_date)
    
    if prev_trade_date:
        print(f"Trade Date: {trade_date}, Previous Trading Day: {prev_trade_date}")
        
        # 3. Load data for both days
        prev_day_data = load_day_data(prev_trade_date)
        today_data = load_day_data(trade_date)
        
        if prev_day_data is not None and today_data is not None:
            # 4. Pick a stock and get its closing price from the previous day
            ticker_to_trade = 'AAPL'
            
            prev_day_stock = prev_day_data[prev_day_data['ticker'] == ticker_to_trade]
            if not prev_day_stock.empty:
                prev_day_close = prev_day_stock.iloc[-1]['close']
                
                # 5. Run the strategy
                run_gap_and_fade_strategy(today_data, ticker_to_trade, prev_day_close)

                # --- Example with another stock ---
                ticker_to_trade = 'TSLA'
                prev_day_stock_tsla = prev_day_data[prev_day_data['ticker'] == ticker_to_trade]
                if not prev_day_stock_tsla.empty:
                    prev_day_close_tsla = prev_day_stock_tsla.iloc[-1]['close']
                    run_gap_and_fade_strategy(today_data, ticker_to_trade, prev_day_close_tsla, gap_threshold=0.02)

            else:
                print(f"No data found for {ticker_to_trade} on {prev_trade_date}")
    else:
        print(f"Could not find a previous trading day with data for {trade_date}") 