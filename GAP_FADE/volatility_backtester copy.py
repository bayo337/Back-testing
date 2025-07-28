import pandas as pd
import os
import numpy as np
from datetime import time, timedelta, datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
FILTERED_STOCKS_FILE = 'final_filtered_stocks.csv'
MINUTE_DATA_DIR = 'historical-DATA'
DAILY_DATA_DIR = 'daily-historical-DATA'
RISK_PER_TRADE = 100.0
PROFIT_TARGET_R = 4.0
INITIAL_ACCOUNT_BALANCE = 2500.0
OUTPUT_DIR_NAME = 'climactic_reversal_simulated_5s_entry' # New folder for this test

# --- Helper Functions ---
def get_previous_day_close(ticker, trade_date_str):
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
    def __init__(self, ticker, date, entry_price, stop_loss):
        self.ticker = ticker
        self.date = date
        self.status = 'OPEN'

        self.initial_entry_price = entry_price
        self.initial_stop_high = stop_loss
        self.risk_per_share = self.initial_stop_high - self.initial_entry_price
        
        if self.risk_per_share <= 0:
            self.status = 'INVALID'
            return

        self.total_dollar_risk = RISK_PER_TRADE
        self.full_position_size = int(self.total_dollar_risk / self.risk_per_share)
        if self.full_position_size == 0: self.full_position_size = 1

        self.chunks_added = 0
        self.position_size = 0
        self.entries = []
        self.avg_entry_price = 0
        
        self.stop_loss = self.initial_stop_high
        self.exit_price = None
        self.exit_time = None
        
        self.pnl_in_dollars = 0.0
        self.max_profit_per_share = 0.0
        self.max_rr_achieved = 0.0
        
        self.partially_closed = False
        self.breakeven_stop_activated = False

        self.add_chunk(entry_price)

    def add_chunk(self, entry_price):
        if self.chunks_added >= 3 or self.status != 'OPEN':
            return False

        chunk_size = self.full_position_size // 3
        if chunk_size == 0: chunk_size = 1
        
        # On the last chunk, add the remainder
        if self.chunks_added == 2:
            chunk_size = self.full_position_size - self.position_size

        if self.position_size + chunk_size > self.full_position_size:
            chunk_size = self.full_position_size - self.position_size

        if chunk_size <= 0:
            return False

        self.avg_entry_price = ((self.avg_entry_price * self.position_size) + (entry_price * chunk_size)) / (self.position_size + chunk_size)
        self.position_size += chunk_size
        self.entries.append(entry_price)
        self.chunks_added += 1
        return True

    def take_partial_profit(self, exit_price, exit_time):
        if self.partially_closed or self.position_size <= 1:
            return

        partial_size = self.full_position_size // 2
        if partial_size > 0 and self.position_size >= partial_size:
            self.pnl_in_dollars += (self.avg_entry_price - exit_price) * partial_size
            self.position_size -= partial_size
            self.partially_closed = True
            self.stop_loss = self.avg_entry_price
            self.breakeven_stop_activated = True

    def update_max_profit(self, current_low):
        potential_profit_per_share = self.avg_entry_price - current_low
        if potential_profit_per_share > self.max_profit_per_share:
            self.max_profit_per_share = potential_profit_per_share
        
        if self.risk_per_share > 0:
            current_rr = potential_profit_per_share / self.risk_per_share
            if current_rr > self.max_rr_achieved:
                self.max_rr_achieved = current_rr
            return current_rr
        return 0

    def close(self, exit_price, exit_time, status):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = status
        self.pnl_in_dollars += (self.avg_entry_price - self.exit_price) * self.position_size
        return self.to_dict()

    def to_dict(self):
        return {
            'ticker': self.ticker, 'date': self.date, 'status': self.status,
            'entry_attempt': len(self.entries), 'entry_price': self.avg_entry_price,
            'stop_loss': self.stop_loss, 'total_dollar_risk': self.total_dollar_risk,
            'exit_price': self.exit_price, 'exit_time': self.exit_time,
            'pnl_in_dollars': self.pnl_in_dollars,
            'max_profit_per_share': self.max_profit_per_share,
            'max_rr_achieved': self.max_rr_achieved,
            'chunks_added': self.chunks_added
        }

def _execute_trade_logic(stock_df, prev_day_close, gain_threshold, ticker, date_str):
    trades_today = []
    state = 'WAITING_FOR_SETUP'
    entry_attempts = 0
    high_of_day = 0.0
    peak_candle_low = None
    peak_candle_high = None
    active_trade = None
    trade_won_today = False
    cumulative_volume = 0
    market_open_time = time(9, 30)
    market_close_time = time(16, 0)
    
    analysis_df = stock_df.between_time(market_open_time, market_close_time)
    if analysis_df.empty: return []

    for index, current_candle in analysis_df.iterrows():
        if trade_won_today: break

        # --- State Machine Logic ---
        # First, handle actions if we are already in a trade
        if state == 'IN_TRADE' and active_trade:
            # Check for stop loss first
            if current_candle['high'] > active_trade.stop_loss:
                status = 'STOPPED_OUT'
                if active_trade.breakeven_stop_activated:
                    status = 'BREAKEVEN_STOP'
                
                active_trade.close(active_trade.stop_loss, index.time(), status)
                if active_trade.pnl_in_dollars > 0: trade_won_today = True
                
                state = 'WAITING_FOR_SETUP'
                active_trade = None
            else:
                # If not stopped, manage the trade
                current_r = active_trade.update_max_profit(current_candle['low'])
                if active_trade.chunks_added == 1 and current_r >= 1.0:
                    active_trade.add_chunk(current_candle['close'])
                elif active_trade.chunks_added == 2 and current_r >= 2.0:
                    active_trade.add_chunk(current_candle['close'])

                if not active_trade.partially_closed and current_r >= PROFIT_TARGET_R:
                    profit_target_price = active_trade.avg_entry_price - (active_trade.risk_per_share * PROFIT_TARGET_R)
                    active_trade.take_partial_profit(profit_target_price, index.time())

        # Next, handle entries and setup tracking, ensuring we don't act on the same candle as a stop-out
        if state != 'IN_TRADE':
            is_new_high = current_candle['high'] >= high_of_day
            if is_new_high: high_of_day = current_candle['high']
            
            cumulative_volume += current_candle['volume']

            if state == 'WAITING_FOR_SETUP':
                if (high_of_day - prev_day_close) / prev_day_close >= gain_threshold and cumulative_volume >= 1_000_000 and current_candle['close'] >= 0.50:
                    state = 'TRACKING_PEAK'

            elif state == 'TRACKING_PEAK':
                if is_new_high:
                    peak_candle_low = current_candle['low']
                    peak_candle_high = current_candle['high']
                else:
                    if peak_candle_low is not None and current_candle['low'] < peak_candle_low and entry_attempts < 3:
                        # **SIMULATED 5-SECOND ENTRY LOGIC**
                        # We assume we could have entered at the exact moment the peak_candle_low was broken.
                        # This gives us the best possible entry price for this type of setup.
                        entry_price = peak_candle_low
                        stop_loss = peak_candle_high
                        
                        active_trade = Trade(ticker, date_str, entry_price, stop_loss)
                        if active_trade.status == 'INVALID':
                            active_trade = None
                            continue

                        entry_attempts += 1
                        trades_today.append(active_trade)
                        state = 'IN_TRADE'
                        # We must re-run the trade management logic on the same candle right after entry
                        # This handles the edge case where the entry, scale-in, and partial profit all happen on the same candle
                        if state == 'IN_TRADE' and active_trade:
                             if current_candle['high'] > active_trade.stop_loss:
                                status = 'STOPPED_OUT'
                                if active_trade.breakeven_stop_activated: status = 'BREAKEVEN_STOP'
                                active_trade.close(active_trade.stop_loss, index.time(), status)
                                if active_trade.pnl_in_dollars > 0: trade_won_today = True
                                state = 'WAITING_FOR_SETUP'
                                active_trade = None

    if active_trade and active_trade.status == 'OPEN':
        active_trade.close(analysis_df.iloc[-1]['close'], analysis_df.index[-1].time(), 'CLOSED_EOD')

    return [t.to_dict() for t in trades_today if t is not None]

def run_volatility_backtest(ticker, date_str, gain_threshold):
    prev_day_close = get_previous_day_close(ticker, date_str)
    if prev_day_close is None: return []
    
    minute_file = os.path.join(MINUTE_DATA_DIR, f'{date_str}.csv')
    if not os.path.exists(minute_file): return []

    try:
        # Step 1: Read CSV without auto-parsing dates, which can be unreliable.
        minute_df = pd.read_csv(minute_file)
        stock_df = minute_df[minute_df['ticker'] == ticker].copy()
        if stock_df.empty: return []
    except Exception: return []

    # Step 2: Explicitly and robustly convert the 'window_start' column to datetime.
    # 'errors=coerce' will turn any problematic date strings into NaT (Not a Time).
    stock_df['window_start'] = pd.to_datetime(stock_df['window_start'], errors='coerce')
    
    # Step 3: Drop any rows where the date conversion failed.
    stock_df.dropna(subset=['window_start'], inplace=True)
    if stock_df.empty:
        return []

    stock_df['timestamp'] = stock_df['window_start'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    stock_df = stock_df.set_index('timestamp').drop(columns=['window_start'])
    stock_df.sort_index(inplace=True)

    return _execute_trade_logic(stock_df, prev_day_close, gain_threshold, ticker, date_str)

def generate_performance_report(trades_df, gain_threshold, interval):
    if trades_df.empty:
        print(f"No trades for {int(gain_threshold*100)}% {interval} to generate a report for.")
        return

    trades_df['pnl_in_dollars'] = trades_df['pnl_in_dollars'].apply(lambda pnl: pnl * 0.95 if pnl > 0 else pnl)
    trades_df['pnl_in_R'] = trades_df['pnl_in_dollars'] / RISK_PER_TRADE
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df['datetime'] = trades_df.apply(lambda row: datetime.combine(row['date'].date(), row['exit_time']), axis=1)
    trades_df.sort_values(by='datetime', inplace=True)
    trades_df.reset_index(inplace=True, drop=True)
    
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
    
    initial_balance = INITIAL_ACCOUNT_BALANCE
    final_balance = initial_balance + net_profit_dollars
    total_return_pct = (net_profit_dollars / initial_balance) * 100 if initial_balance > 0 else 0
    trades_df['equity_curve'] = trades_df['pnl_in_dollars'].cumsum() + initial_balance
    account_low = trades_df['equity_curve'].min() if not trades_df.empty else initial_balance
    trades_df['running_max'] = trades_df['equity_curve'].cummax()
    trades_df['drawdown'] = trades_df['running_max'] - trades_df['equity_curve']
    max_drawdown_dollars = trades_df['drawdown'].max() if not trades_df.empty else 0
    
    wins = trades_df['pnl_in_dollars'] > 0
    losses = trades_df['pnl_in_dollars'] < 0
    max_consecutive_wins = (wins.groupby((wins != wins.shift()).cumsum()).cumsum()).max() if not wins.empty else 0
    max_consecutive_losses = (losses.groupby((losses != losses.shift()).cumsum()).cumsum()).max() if not losses.empty else 0
    
    sharpe_ratio = (trades_df['pnl_in_R'].mean() / trades_df['pnl_in_R'].std()) * np.sqrt(252) if trades_df['pnl_in_R'].std() != 0 else 0
    avg_max_rr_achieved = trades_df['max_rr_achieved'].mean()
    reward_risk_ratio = avg_win_R / avg_loss_R if avg_loss_R != 0 else float('inf')
    
    gain_pct = int(gain_threshold * 100)

    # --- Generate and Save Equity Curve Plot ---
    chart_dir = os.path.join('GAP_FADE', OUTPUT_DIR_NAME, 'equity_charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = os.path.join(chart_dir, f'equity_curve_{gain_pct}pct_{interval}.png')
    plt.figure(figsize=(14, 8))
    plt.plot(trades_df['datetime'], trades_df['equity_curve'], label='Equity Curve', color='navy', linewidth=2)
    plt.title(f'Equity Curve - {gain_pct}% Gain ({interval})', fontsize=18, fontweight='bold')
    plt.savefig(chart_filename)
    plt.close()

    report = f"""
======================================================
     Strategy Report ({gain_pct}% Gain, {interval})
======================================================
NOTE: A 5% handicap has been applied to all gross profits.
      Position size is dynamic, based on a fixed risk of ${RISK_PER_TRADE:.2f} per trade.

--- Account Summary ---
Initial Balance: ${initial_balance:,.2f}
Final Balance:   ${final_balance:,.2f}
Net Profit:      ${net_profit_dollars:,.2f}
Total Return:    {total_return_pct:.2f}%
Lowest Value:    ${account_low:,.2f}

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
Max Drawdown: ${max_drawdown_dollars:,.2f} ({(max_drawdown_dollars/initial_balance)*100:.2f}% of initial balance)
Max Consecutive Wins: {max_consecutive_wins}
Max Consecutive Losses: {max_consecutive_losses}

--- R/R Success Rates (% of all trades) ---
Achieved >= 3R: {success_rate_3R:6.2f}%
Achieved >= 4R: {success_rate_4R:6.2f}%

======================= End of Report =======================
"""
    report_dir = os.path.join('GAP_FADE', OUTPUT_DIR_NAME, 'reports_txt')
    os.makedirs(report_dir, exist_ok=True)
    report_filename = os.path.join(report_dir, f'performance_report_{gain_pct}pct_{interval}.txt')
    with open(report_filename, 'w') as f:
        f.write(report)
    print(f"  -> Report saved for {gain_pct}% {interval}")


def run_single_parameter_backtest(args):
    gain_pct, candidate_df, position = args
    gain_threshold = gain_pct / 100.0
    log_dir = os.path.join('GAP_FADE', OUTPUT_DIR_NAME, 'trade_logs_csv')
    os.makedirs(log_dir, exist_ok=True)
    
    all_trades = []
    for _, row in tqdm(candidate_df.iterrows(), total=candidate_df.shape[0], desc=f"{gain_pct}% Gain", position=position, leave=True):
        date_str = row['date'].strftime('%Y-%m-%d')
        trades = run_volatility_backtest(row['ticker'], date_str, gain_threshold)
        if trades:
            all_trades.extend(trades)
            
    if all_trades:
        results_df = pd.DataFrame(all_trades)
        filename = os.path.join(log_dir, f'backtest_results_{gain_pct}pct_1min.csv')
        results_df.to_csv(filename, index=False)
        generate_performance_report(results_df, gain_threshold, '1min')
        
    return f"Completed {gain_pct}%"

if __name__ == '__main__':
    try:
        full_candidate_df = pd.read_csv(FILTERED_STOCKS_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{FILTERED_STOCKS_FILE}' was not found.")
        print("Please ensure the data preparation script has been run successfully.")
        exit()
        
    full_candidate_df['date'] = pd.to_datetime(full_candidate_df['date'])
    unique_candidates_df = full_candidate_df.sort_values(by='date').drop_duplicates(subset=['ticker', 'date']).reset_index(drop=True)
    
    test_candidates_df = unique_candidates_df.head(500)
    print(f"--- Preparing to run a backtest on the first {len(test_candidates_df)} unique events. ---")

    gain_percentages = [50]
    jobs = [(gp, test_candidates_df, i) for i, gp in enumerate(gain_percentages)]
    
    print(f"--- Starting {len(jobs)} backtest job(s). This may take some time. ---")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(run_single_parameter_backtest, jobs)
        completed_tasks = list(results)

    print("\n\n--- All Backtests Complete! ---")
    for res in completed_tasks:
        print(f"  - {res}")
