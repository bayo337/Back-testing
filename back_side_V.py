import pandas as pd
import os
import numpy as np
from datetime import time, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import warnings

# --- Ignore harmless FutureWarning from pandas ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
FILTERED_STOCKS_FILE = 'back_side_V/refined_filtered_stocks_down.csv' # searching for stocks that have moved down
MINUTE_DATA_DIR = 'historical-DATA'
DAILY_DATA_DIR = 'daily-historical-DATA'
RISK_PER_TRADE = 100.0 
PROFIT_TARGET_R = 3.0

# --- Helper Functions ---
def get_previous_day_close(ticker, trade_date_str):
    """Fetches the previous day's closing price for a given ticker."""
    trade_date = pd.to_datetime(trade_date_str).date()
    for i in range(1, 6):
        prev_date = trade_date - timedelta(days=i)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        daily_file_path = os.path.join(DAILY_DATA_DIR, f'{prev_date_str}.csv')
        
        if os.path.exists(daily_file_path):
            try:
                daily_df = pd.read_csv(daily_file_path)
                stock_data = daily_df[daily_df['ticker'] == ticker]
                if not stock_data.empty:
                    return stock_data.iloc[0]['close']
            except Exception:
                return None
    return None

class Trade:
    """A class to manage the state and metrics of a single trade."""
    def __init__(self, ticker, date, entry_price, stop_loss, entry_attempt, full_position_size):
        self.ticker = ticker
        self.date = date
        self.entry_attempt = entry_attempt
        self.status = 'OPEN'
        self.initial_entry_price = entry_price
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        
        self.full_position_size = full_position_size
        self.position_size = max(1, int(self.full_position_size * 0.25))
        
        self.exit_price = None
        self.exit_time = None
        self.max_profit_per_share = 0.0
        
        self.total_dollar_risk = RISK_PER_TRADE
        self.max_rr_achieved = 0.0
        
        self.can_add_more = True
        self.chunks_added = 1

    def add_chunk(self, add_on_price):
        if not self.can_add_more or self.chunks_added >= 4:
            return

        current_total_shares = self.position_size
        current_total_cost = self.entry_price * current_total_shares
        
        add_on_shares = max(1, int(self.full_position_size * 0.25))
        new_total_shares = current_total_shares + add_on_shares
        
        if new_total_shares > self.full_position_size:
            new_total_shares = self.full_position_size
            add_on_shares = new_total_shares - current_total_shares
            if add_on_shares <= 0: return

        new_average_entry_price = (current_total_cost + (add_on_price * add_on_shares)) / new_total_shares
        
        self.entry_price = new_average_entry_price
        self.position_size = new_total_shares
        self.chunks_added += 1
        
        if self.position_size > 0:
            new_risk_per_share = self.total_dollar_risk / self.position_size
            self.stop_loss = self.entry_price - new_risk_per_share # Stop loss is below entry for a long trade

    def update_max_profit(self, current_high):
        potential_profit_per_share = current_high - self.entry_price # Profit for a long trade
        if potential_profit_per_share > self.max_profit_per_share:
            self.max_profit_per_share = potential_profit_per_share
        
        if self.total_dollar_risk > 0:
            potential_total_profit_dollars = self.max_profit_per_share * self.full_position_size
            current_rr = potential_total_profit_dollars / self.total_dollar_risk
            if current_rr > self.max_rr_achieved:
                self.max_rr_achieved = current_rr

    def close(self, exit_price, exit_time, status):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = status
        self.pnl_in_dollars = (self.exit_price - self.entry_price) * self.position_size # PnL for a long trade
        return self.to_dict()

    def to_dict(self):
        return {
            'ticker': self.ticker, 'date': self.date, 'status': self.status,
            'entry_attempt': self.entry_attempt, 'entry_price': self.entry_price,
            'stop_loss': self.stop_loss, 'total_dollar_risk': self.total_dollar_risk,
            'exit_price': self.exit_price, 'exit_time': self.exit_time,
            'pnl_in_dollars': getattr(self, 'pnl_in_dollars', 0.0),
            'max_profit_per_share': self.max_profit_per_share,
            'max_rr_achieved': self.max_rr_achieved
        }

def _execute_trade_logic(stock_df, prev_day_close, loss_threshold, ticker, date_str):
    trades_today = []
    state = 'SEARCHING_SETUP'
    entry_attempts = 0
    low_of_day = float('inf') 
    cumulative_volume = 0
    active_trade = None

    # Define market hours
    market_open_time = time(9, 30)
    market_close_time = time(16, 0)
    
    # Isolate pre-market and market data
    pre_market_df = stock_df.between_time(time(4, 0), time(9, 29))
    market_df = stock_df.between_time(market_open_time, market_close_time)

    # Consider the last 5 candles of pre-market, if available
    if not pre_market_df.empty:
        # If less than 5 candles, take all of them
        num_pre_market_candles = min(5, len(pre_market_df))
        relevant_pre_market_df = pre_market_df.tail(num_pre_market_candles)
        # Combine relevant pre-market with regular market hours data
        analysis_df = pd.concat([relevant_pre_market_df, market_df])
    else:
        analysis_df = market_df

    if analysis_df.empty:
        return []

    for index, current_candle in analysis_df.iterrows():
        # Update high of day and volume only during market hours
        current_time = index.time()
        if market_open_time <= current_time <= market_close_time:
            low_of_day = min(low_of_day, current_candle['low'])
            cumulative_volume += current_candle['volume']

        if state == 'SEARCHING_SETUP' and entry_attempts < 4:
            # Entry logic for reversal (long) strategy
            total_change = (current_candle['close'] - prev_day_close) / prev_day_close
            
            # --- Condition Check: Stock is down by the threshold amount ---
            if total_change <= loss_threshold and cumulative_volume >= 1_000_000 and current_candle['close'] >= 0.50:
                try:
                    prev_candle_index = analysis_df.index.get_loc(index) - 1
                    if prev_candle_index >= 2:
                        prev_candle = analysis_df.iloc[prev_candle_index]
                        prev_candle_range = prev_candle['high'] - prev_candle['low']
                        trigger_price = prev_candle['close'] + (prev_candle_range * 0.90) # Trigger above prev close
                        
                        stop_loss = low_of_day # Stop loss is the low of day
 
                        if current_candle['high'] > trigger_price: # Enter if price breaks out to the upside
                            risk_per_share = trigger_price - stop_loss
                            if risk_per_share <= 0: continue
                            
                            full_position_size = int(RISK_PER_TRADE / risk_per_share)
                            if full_position_size < 4: continue

                            entry_attempts += 1
                            active_trade = Trade(ticker, date_str, trigger_price, stop_loss, entry_attempts, full_position_size)
                            trades_today.append(active_trade)
                            state = 'IN_TRADE'
                except (KeyError, IndexError):
                    continue # Skip if we can't get previous candles
        
        elif state == 'IN_TRADE':
            # --- Trade Management Logic for a Long Position ---
            if active_trade.can_add_more and current_candle['low'] < active_trade.initial_entry_price:
                active_trade.can_add_more = False

            if active_trade.can_add_more and active_trade.chunks_added < 4:
                try:
                    prev_candle = analysis_df.iloc[analysis_df.index.get_loc(index) - 1]
                    if current_candle['high'] > prev_candle['high']: # Add on strength
                        active_trade.add_chunk(current_candle['high'])
                except (KeyError, IndexError):
                    pass # Ignore if there's no previous candle

            if active_trade.total_dollar_risk > 0:
                profit_target_in_dollars = active_trade.total_dollar_risk * PROFIT_TARGET_R
                current_profit_in_dollars = (current_candle['high'] - active_trade.entry_price) * active_trade.position_size
                if current_profit_in_dollars >= profit_target_in_dollars:
                    profit_per_share_target = profit_target_in_dollars / active_trade.position_size
                    exit_price = active_trade.entry_price + profit_per_share_target
                    active_trade.close(exit_price, index.time(), 'PROFIT_TARGET')
                    state = 'SEARCHING_SETUP'
                    continue 

            active_trade.update_max_profit(current_candle['high'])
            if current_candle['low'] < active_trade.stop_loss:
                active_trade.close(active_trade.stop_loss, index.time(), 'STOPPED_OUT')
                state = 'SEARCHING_SETUP'
                continue

    if active_trade and active_trade.status == 'OPEN':
        end_of_day_close = analysis_df.iloc[-1]['close']
        active_trade.update_max_profit(end_of_day_close)
        active_trade.close(end_of_day_close, analysis_df.index[-1].time(), 'CLOSED_EOD')

    return [t.to_dict() for t in trades_today]

def run_volatility_backtest(ticker, date_str, loss_threshold):
    prev_day_close = get_previous_day_close(ticker, date_str)
    if prev_day_close is None:
        return {'1min': [], '5min': []}

    minute_file = os.path.join(MINUTE_DATA_DIR, f'{date_str}.csv')
    if not os.path.exists(minute_file):
        return {'1min': [], '5min': []}

    try:
        minute_df = pd.read_csv(minute_file)
        stock_df = minute_df[minute_df['ticker'] == ticker].copy()
        if stock_df.empty:
            return {'1min': [], '5min': []}
    except Exception:
        return {'1min': [], '5min': []}

    stock_df['timestamp'] = pd.to_datetime(stock_df['window_start'], utc=True).dt.tz_convert('US/Eastern')
    stock_df = stock_df.set_index('timestamp').drop(columns=['window_start'])
    stock_df.sort_index(inplace=True)

    trades_1min = _execute_trade_logic(stock_df, prev_day_close, loss_threshold, ticker, date_str)
    
    stock_df_5min = stock_df.resample('5T').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    trades_5min = []
    if not stock_df_5min.empty:
        trades_5min = _execute_trade_logic(stock_df_5min, prev_day_close, loss_threshold, ticker, date_str)
        
    return {'1min': trades_1min, '5min': trades_5min}

def generate_performance_report(trades_df, loss_threshold, interval):
    if trades_df.empty:
        print(f"No trades for {int(abs(loss_threshold*100))}% {interval} to generate a report for.")
        return

    trades_df['pnl_in_dollars'] = trades_df['pnl_in_dollars'].apply(lambda pnl: pnl * 0.95 if pnl > 0 else pnl)
    trades_df['pnl_in_R'] = trades_df['pnl_in_dollars'] / RISK_PER_TRADE
    
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl_in_dollars'] > 0]
    losing_trades = trades_df[trades_df['pnl_in_dollars'] < 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit_dollars = winning_trades['pnl_in_dollars'].sum()
    gross_loss_dollars = losing_trades['pnl_in_dollars'].sum()
    net_profit_dollars = gross_profit_dollars + gross_loss_dollars
    profit_factor = abs(gross_profit_dollars / gross_loss_dollars) if gross_loss_dollars != 0 else float('inf')
    avg_win_dollars = winning_trades['pnl_in_dollars'].mean() if len(winning_trades) > 0 else 0
    avg_loss_dollars = abs(losing_trades['pnl_in_dollars'].mean()) if len(losing_trades) > 0 else 0
    avg_win_R = winning_trades['pnl_in_R'].mean() if len(winning_trades) > 0 else 0
    avg_loss_R = abs(losing_trades['pnl_in_R'].mean()) if len(losing_trades) > 0 else 0
    expectancy_dollars = ((win_rate / 100) * avg_win_dollars) - ((100 - win_rate) / 100 * avg_loss_dollars)
    expectancy_R = ((win_rate / 100) * avg_win_R) - ((100 - win_rate) / 100 * avg_loss_R)
    success_rate_3R = (trades_df['max_rr_achieved'] >= 3).sum() / total_trades * 100 if total_trades > 0 else 0
    success_rate_4R = (trades_df['max_rr_achieved'] >= 4).sum() / total_trades * 100 if total_trades > 0 else 0
    success_rate_5R = (trades_df['max_rr_achieved'] >= 5).sum() / total_trades * 100 if total_trades > 0 else 0
    success_rate_6R = (trades_df['max_rr_achieved'] >= 6).sum() / total_trades * 100 if total_trades > 0 else 0
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    yearly_stats = trades_df.groupby(trades_df['date'].dt.to_period('Y')).agg(net_profit_dollars=('pnl_in_dollars', 'sum'), num_trades=('ticker', 'count'))
    monthly_stats = trades_df.groupby(trades_df['date'].dt.to_period('M')).agg(net_profit_dollars=('pnl_in_dollars', 'sum'), num_trades=('ticker', 'count'))
    weekly_stats = trades_df.groupby(trades_df['date'].dt.to_period('W')).agg(net_profit_dollars=('pnl_in_dollars', 'sum'), num_trades=('ticker', 'count'))
    trades_df['equity_curve'] = trades_df['pnl_in_dollars'].cumsum()
    trades_df['running_max'] = trades_df['equity_curve'].cummax()
    trades_df['drawdown'] = trades_df['running_max'] - trades_df['equity_curve']
    max_drawdown_dollars = trades_df['drawdown'].max()
    wins = trades_df['pnl_in_dollars'] > 0
    losses = trades_df['pnl_in_dollars'] < 0
    max_consecutive_wins = (wins.groupby((wins != wins.shift()).cumsum()).cumsum()).max() if not wins.empty else 0
    max_consecutive_losses = (losses.groupby((losses != losses.shift()).cumsum()).cumsum()).max() if not losses.empty else 0
    reentry_stats = trades_df.groupby('entry_attempt').agg(
        num_trades=('ticker', 'count'),
        net_profit_dollars=('pnl_in_dollars', 'sum'),
        win_rate=('pnl_in_dollars', lambda pnl: (pnl > 0).sum() / pnl.count() * 100)
    ).rename(columns={'win_rate': 'win_rate (%)'})
    sharpe_ratio = (trades_df['pnl_in_R'].mean() / trades_df['pnl_in_R'].std()) * np.sqrt(252) if trades_df['pnl_in_R'].std() != 0 else 0
    avg_max_rr_achieved = trades_df['max_rr_achieved'].mean()
    reward_risk_ratio = avg_win_R / avg_loss_R if avg_loss_R != 0 else float('inf')
    loss_pct = int(abs(loss_threshold * 100))
    report = f"""
======================================================
     Reversal Strategy Report ({loss_pct}% Loss, {interval})
======================================================
NOTE: A 5% handicap has been applied to all gross profits.
      Position size is dynamic, based on a fixed risk of ${RISK_PER_TRADE:.2f} per trade.

--- Overall Performance ---
Total Trades: {total_trades}
Win Rate: {win_rate:.2f}%
Profit Factor: {profit_factor:.2f}
Reward/Risk Ratio: {reward_risk_ratio:.2f}:1
Sharpe Ratio (annualized): {sharpe_ratio:.2f}
Avg. Max R/R Achieved: {avg_max_rr_achieved:.2f}R
------------------------------------------------------
            |    Dollar Value    |      R-Value
------------------------------------------------------
Net Profit  | ${net_profit_dollars:15,.2f} | {net_profit_dollars / RISK_PER_TRADE:12.2f}R
Avg. Win    | ${avg_win_dollars:15,.2f} | {avg_win_R:12.2f}R
Avg. Loss   | ${avg_loss_dollars:15,.2f} | {avg_loss_R:12.2f}R
Expectancy  | ${expectancy_dollars:15,.2f} | {expectancy_R:12.2f}R
------------------------------------------------------

--- Key Metrics ---
Max Drawdown: ${max_drawdown_dollars:,.2f}
Max Consecutive Wins: {max_consecutive_wins}
Max Consecutive Losses: {max_consecutive_losses}

--- R/R Success Rates (% of all trades) ---
Achieved 1:3 R/R: {success_rate_3R:6.2f}%
Achieved 1:4 R/R: {success_rate_4R:6.2f}%
Achieved 1:5 R/R: {success_rate_5R:6.2f}%
Achieved 1:6 R/R: {success_rate_6R:6.2f}%

--- Re-entry Performance ---
{reentry_stats.to_string()}

--- Yearly Performance (Net Profit in $) ---
{yearly_stats.to_string()}

--- Monthly Performance (Net Profit in $) ---
{monthly_stats.to_string()}

--- Weekly Performance (Net Profit in $) ---
{weekly_stats.to_string()}

======================= End of Report =======================
"""
    os.makedirs(os.path.join('back_side_V', 'reports_txt'), exist_ok=True)
    report_filename = os.path.join('back_side_V', 'reports_txt', f'performance_report_{loss_pct}pct_loss_{interval}.txt')
    with open(report_filename, 'w') as f:
        f.write(report)
    print(f"  -> Report saved for {loss_pct}% {interval}")

def run_single_parameter_backtest(args):
    loss_pct, candidate_df, position = args
    loss_threshold = -(loss_pct / 100.0)
    
    # --- Create directories inside the parallel process ---
    os.makedirs(os.path.join('back_side_V', 'trade_logs_csv'), exist_ok=True)
    os.makedirs(os.path.join('back_side_V', 'reports_txt'), exist_ok=True)
    
    all_trades_1min, all_trades_5min = [], []
    
    for _, row in tqdm(candidate_df.iterrows(), total=candidate_df.shape[0], desc=f"{loss_pct}% Loss", position=position, leave=True):
        date_str = row['date'].strftime('%Y-%m-%d')
        results = run_volatility_backtest(row['ticker'], date_str, loss_threshold)
        if results['1min']: all_trades_1min.extend(results['1min'])
        if results['5min']: all_trades_5min.extend(results['5min'])
            
    if all_trades_1min:
        results_df_1min = pd.DataFrame(all_trades_1min)
        cols = ['date', 'ticker', 'status', 'entry_attempt', 'entry_price', 'stop_loss', 'total_dollar_risk', 'exit_time', 'exit_price', 'pnl_in_dollars', 'max_profit_per_share', 'max_rr_achieved']
        results_df_1min = results_df_1min[cols]
        filename_1min = os.path.join('back_side_V', 'trade_logs_csv', f'reversal_backtest_results_{loss_pct}pct_1min.csv')
        results_df_1min.to_csv(filename_1min, index=False)
        generate_performance_report(results_df_1min, loss_threshold, '1min')
    
    if all_trades_5min:
        results_df_5min = pd.DataFrame(all_trades_5min)
        cols = ['date', 'ticker', 'status', 'entry_attempt', 'entry_price', 'stop_loss', 'total_dollar_risk', 'exit_time', 'exit_price', 'pnl_in_dollars', 'max_profit_per_share', 'max_rr_achieved']
        results_df_5min = results_df_5min[cols]
        filename_5min = os.path.join('back_side_V', 'trade_logs_csv', f'reversal_backtest_results_{loss_pct}pct_5min.csv')
        results_df_5min.to_csv(filename_5min, index=False)
        generate_performance_report(results_df_5min, loss_threshold, '5min')
        
    return f"Completed {loss_pct}%"

if __name__ == '__main__':
    # Directories are created in the parallel function, but we can make them here too.
    os.makedirs(os.path.join('back_side_V', 'trade_logs_csv'), exist_ok=True)
    os.makedirs(os.path.join('back_side_V', 'reports_txt'), exist_ok=True)

    try:
        full_candidate_df = pd.read_csv(FILTERED_STOCKS_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{FILTERED_STOCKS_FILE}' was not found. Please run the filter script first.")
        exit()
    
    full_candidate_df['date'] = pd.to_datetime(full_candidate_df['date'])
    unique_candidates_df = full_candidate_df.sort_values(by='date').drop_duplicates(subset=['ticker', 'date']).reset_index(drop=True)
    
    # --- TEST RUN: Use first 100 candidates ---
    # test_candidate_df = unique_candidates_df.head(100)
        
    print(f"--- Preparing to run the backtest on all {len(unique_candidates_df)} candidates. ---")

    # --- FULL RUN: Test all loss thresholds ---
    loss_percentages = [10, 20, 30, 40, 50, 60]
    
    jobs = [(lp, unique_candidates_df, i) for i, lp in enumerate(loss_percentages)]
    
    print(f"--- Starting {len(jobs)} parallel backtest jobs. You will see a progress bar for each. ---")
    print("==========================================================================================")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(run_single_parameter_backtest, jobs)
        completed_tasks = list(results)

    print("\n\n--- All Parameter Backtests Complete! ---")
    for res in completed_tasks:
        print(f"  - {res}") 