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
RISK_PER_TRADE = 100.0  # Static risk per trade
PROFIT_TARGET_R = 4.0
INITIAL_ACCOUNT_BALANCE = 2500.0
OUTPUT_DIR_NAME = 'climactic_reversal_test'
# Note: RISK_PERCENTAGE from the other script is not used here to maintain static bet sizing.

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
    def __init__(self, ticker, date, entry_price, stop_loss, entry_attempt, full_position_size, total_dollar_risk):
        self.ticker = ticker
        self.date = date
        self.entry_attempt = entry_attempt
        self.status = 'OPEN'
        self.initial_entry_price = entry_price
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.initial_stop_high = stop_loss
        
        self.full_position_size = full_position_size
        self.position_size = max(1, int(self.full_position_size * 0.14))
        
        self.exit_price = None
        self.exit_time = None
        self.max_profit_per_share = 0.0
        
        self.total_dollar_risk = total_dollar_risk
        self.max_rr_achieved = 0.0
        
        self.can_add_more = True
        self.chunks_added = 1

        self.pnl_in_dollars = 0.0
        self.partially_closed = False
        self.breakeven_stop_activated = False
        self.took_profit_4R = False

    def add_chunk(self, add_on_price):
        if self.chunks_added >= 3:
            return

        current_total_shares = self.position_size
        current_total_cost = self.entry_price * current_total_shares
        
        if self.chunks_added == 1: # Second chunk
            add_on_percentage = 0.28
        elif self.chunks_added == 2: # Third chunk
            add_on_percentage = 0.58
        else:
            return

        add_on_shares = max(1, int(self.full_position_size * add_on_percentage))
        new_total_shares = current_total_shares + add_on_shares
        
        if new_total_shares > self.full_position_size:
            add_on_shares = self.full_position_size - current_total_shares
            if add_on_shares <= 0: return
            new_total_shares = self.full_position_size

        new_average_entry_price = (current_total_cost + (add_on_price * add_on_shares)) / new_total_shares
        
        self.entry_price = new_average_entry_price
        self.position_size = new_total_shares
        self.chunks_added += 1
        
        if self.position_size > 0:
            new_risk_per_share = self.total_dollar_risk / self.position_size
            self.stop_loss = self.entry_price + new_risk_per_share

    def take_profit(self, exit_price, exit_time, percentage_of_full_size_to_close):
        if self.position_size <= 1:
            return False

        shares_to_close = int(self.full_position_size * percentage_of_full_size_to_close)
        shares_to_close = min(shares_to_close, self.position_size)

        if shares_to_close > 0:
            self.pnl_in_dollars += (self.entry_price - exit_price) * shares_to_close
            self.position_size -= shares_to_close
            self.partially_closed = True
            self.breakeven_stop_activated = True
            self.stop_loss = self.entry_price
            return True
        return False

    def update_max_profit(self, current_low):
        potential_profit_per_share = self.entry_price - current_low
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
        self.pnl_in_dollars += (self.entry_price - self.exit_price) * self.position_size
        return self.to_dict()

    def to_dict(self):
        return {
            'ticker': self.ticker, 'date': self.date, 'status': self.status,
            'entry_attempt': self.entry_attempt, 'entry_price': self.entry_price,
            'stop_loss': self.stop_loss, 'total_dollar_risk': self.total_dollar_risk,
            'exit_price': self.exit_price, 'exit_time': self.exit_time,
            'pnl_in_dollars': self.pnl_in_dollars,
            'max_profit_per_share': self.max_profit_per_share,
            'max_rr_achieved': self.max_rr_achieved
        }

def _execute_trade_logic(stock_df, prev_day_close, gain_threshold, ticker, date_str, risk_per_trade_dollars):
    trades_today = []
    state = 'SEARCHING_SETUP'
    entry_attempts = 0
    high_of_day = 0.0 
    high_since_entry = 0.0
    cumulative_volume = 0
    active_trade = None
    trade_won_today = False
    
    market_open_time = time(9, 30)
    market_close_time = time(16, 0)
    
    pre_market_df = stock_df.between_time(time(4, 0), time(9, 29))
    market_df = stock_df.between_time(market_open_time, market_close_time)

    if not pre_market_df.empty:
        num_pre_market_candles = min(5, len(pre_market_df))
        relevant_pre_market_df = pre_market_df.tail(num_pre_market_candles)
        analysis_df = pd.concat([relevant_pre_market_df, market_df])
    else:
        analysis_df = market_df

    if analysis_df.empty:
        return []

    analysis_df = analysis_df.copy()
    analysis_df['ema_13'] = analysis_df['close'].ewm(span=13, adjust=False).mean()
    obv = (np.sign(analysis_df['close'].diff()) * analysis_df['volume']).fillna(0).cumsum()
    analysis_df['obv'] = obv

    for index, current_candle in analysis_df.iterrows():
        if trade_won_today:
            break
        
        current_time = index.time()
        if market_open_time <= current_time <= market_close_time:
            high_of_day = max(high_of_day, current_candle['high'])
            cumulative_volume += current_candle['volume']

        if state == 'SEARCHING_SETUP' and entry_attempts < 3:
            total_gain = (current_candle['close'] - prev_day_close) / prev_day_close
            
            if total_gain >= gain_threshold and cumulative_volume >= 1_000_000 and current_candle['close'] >= 0.50:
                try:
                    prev_candle_index = analysis_df.index.get_loc(index) - 1
                    if prev_candle_index >= 2:
                        prev_candle = analysis_df.iloc[prev_candle_index]
                        
                        prev_candle_range = prev_candle['high'] - prev_candle['low']
                        trigger_price = prev_candle['close'] - (prev_candle_range * 0.90)

                        if current_candle['low'] < trigger_price:
                            recent_highs = analysis_df.iloc[prev_candle_index-1 : prev_candle_index+1]['high']
                            stop_loss = recent_highs.max()

                            risk_per_share = stop_loss - trigger_price
                            if risk_per_share <= 0: continue
                            
                            full_position_size = int(risk_per_trade_dollars / risk_per_share)
                            if full_position_size < 4: continue

                            entry_attempts += 1
                            active_trade = Trade(ticker, date_str, trigger_price, stop_loss, entry_attempts, full_position_size, risk_per_trade_dollars)
                            trades_today.append(active_trade)
                            state = 'IN_TRADE'
                            high_since_entry = current_candle['high']
                except (KeyError, IndexError):
                    continue
        
        elif state == 'IN_TRADE':
            high_since_entry = max(high_since_entry, current_candle['high'])
            active_trade.update_max_profit(current_candle['low'])

            current_profit_dollars = (active_trade.entry_price - current_candle['low']) * active_trade.full_position_size
            current_r_profit = current_profit_dollars / active_trade.total_dollar_risk if active_trade.total_dollar_risk > 0 else 0

            if not active_trade.took_profit_4R and current_r_profit >= 4.0:
                profit_per_share_target = (active_trade.total_dollar_risk * 4.0) / active_trade.full_position_size
                exit_price = active_trade.entry_price - profit_per_share_target
                if active_trade.take_profit(exit_price, index.time(), 0.50):
                    active_trade.took_profit_4R = True
            
            if active_trade.can_add_more and current_candle['high'] > active_trade.initial_entry_price:
                active_trade.can_add_more = False

            if not active_trade.partially_closed and active_trade.can_add_more and active_trade.chunks_added < 3:
                if high_since_entry < active_trade.initial_stop_high:
                    try:
                        prev_candle = analysis_df.iloc[analysis_df.index.get_loc(index) - 1]
                        if current_candle['close'] < prev_candle['low']:
                            active_trade.add_chunk(current_candle['close'])
                    except (KeyError, IndexError):
                        pass 

            if current_candle['high'] > active_trade.stop_loss:
                status = 'STOPPED_OUT'
                if active_trade.partially_closed:
                    status = 'SCALED_OUT_STOP'
                elif active_trade.stop_loss < active_trade.initial_entry_price:
                    status = 'TRAILING_STOP_HIT'
                active_trade.close(active_trade.stop_loss, index.time(), status)
                if active_trade.pnl_in_dollars > 0:
                    trade_won_today = True
                state = 'SEARCHING_SETUP'
                continue

    if active_trade and active_trade.status == 'OPEN':
        end_of_day_close = analysis_df.iloc[-1]['close']
        active_trade.update_max_profit(end_of_day_close)
        active_trade.close(end_of_day_close, analysis_df.index[-1].time(), 'CLOSED_EOD')

    return [t.to_dict() for t in trades_today]

def run_volatility_backtest(ticker, date_str, gain_threshold, risk_per_trade_dollars, interval):
    prev_day_close = get_previous_day_close(ticker, date_str)
    if prev_day_close is None:
        return []

    minute_file = os.path.join(MINUTE_DATA_DIR, f'{date_str}.csv')
    if not os.path.exists(minute_file):
        return []

    try:
        minute_df = pd.read_csv(minute_file)
        stock_df = minute_df[minute_df['ticker'] == ticker].copy()
        if stock_df.empty:
            return []
    except Exception:
        return []

    stock_df['timestamp'] = pd.to_datetime(stock_df['window_start'], utc=True).dt.tz_convert('US/Eastern')
    stock_df = stock_df.set_index('timestamp').drop(columns=['window_start'])
    stock_df.sort_index(inplace=True)

    if interval != '1min':
        aggregation_rules = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }
        stock_df = stock_df.resample(interval).agg(aggregation_rules).dropna()
        if stock_df.empty:
            return []
    
    return _execute_trade_logic(stock_df, prev_day_close, gain_threshold, ticker, date_str, risk_per_trade_dollars)

def generate_performance_report(trades_df, gain_threshold, interval, initial_balance):
    if trades_df.empty:
        print(f"No trades for {int(gain_threshold*100)}% {interval} to generate a report for.")
        return

    trades_df['pnl_in_dollars'] = trades_df['pnl_in_dollars'].apply(lambda pnl: pnl * 0.95 if pnl > 0 else pnl)
    trades_df['pnl_in_R'] = trades_df.apply(lambda row: row['pnl_in_dollars'] / row['total_dollar_risk'] if row['total_dollar_risk'] != 0 else 0, axis=1)
    
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    # Handle cases where exit_time might be missing or not a valid time object
    trades_df['datetime'] = pd.to_datetime(trades_df['date'].dt.date.astype(str) + ' ' + trades_df['exit_time'].astype(str), errors='coerce')
    trades_df.dropna(subset=['datetime'], inplace=True)
    trades_df.sort_values(by='datetime', inplace=True)
    trades_df.reset_index(inplace=True, drop=True)

    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl_in_dollars'] > 0]
    losing_trades = trades_df[trades_df['pnl_in_dollars'] < 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit_dollars = winning_trades['pnl_in_dollars'].sum()
    gross_loss_dollars = losing_trades['pnl_in_dollars'].sum()
    net_profit_dollars = gross_profit_dollars + gross_loss_dollars
    final_balance = initial_balance + net_profit_dollars
    total_return_pct = (net_profit_dollars / initial_balance) * 100

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
    yearly_stats = trades_df.groupby(trades_df['date'].dt.to_period('Y')).agg(net_profit_dollars=('pnl_in_dollars', 'sum'), num_trades=('ticker', 'count'))
    monthly_stats = trades_df.groupby(trades_df['date'].dt.to_period('M')).agg(net_profit_dollars=('pnl_in_dollars', 'sum'), num_trades=('ticker', 'count'))
    weekly_stats = trades_df.groupby(trades_df['date'].dt.to_period('W')).agg(net_profit_dollars=('pnl_in_dollars', 'sum'), num_trades=('ticker', 'count'))
    
    trades_df['equity_curve'] = trades_df['pnl_in_dollars'].cumsum() + initial_balance
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
    total_pnl_in_R = trades_df['pnl_in_R'].sum()
    reward_risk_ratio = avg_win_R / avg_loss_R if avg_loss_R != 0 else float('inf')
    gain_pct = int(gain_threshold * 100)

    chart_dir = os.path.join('GAP_FADE', OUTPUT_DIR_NAME, 'equity_charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = os.path.join(chart_dir, f'equity_curve_{gain_pct}pct_{interval}.png')

    plt.figure(figsize=(14, 8))
    plt.plot(trades_df['datetime'], trades_df['equity_curve'], label='Equity Curve', color='navy', linewidth=2)
    plt.fill_between(trades_df['datetime'], trades_df['equity_curve'], initial_balance, where=trades_df['equity_curve'] > initial_balance, facecolor='green', alpha=0.3, interpolate=True)
    plt.fill_between(trades_df['datetime'], trades_df['equity_curve'], initial_balance, where=trades_df['equity_curve'] < initial_balance, facecolor='red', alpha=0.3, interpolate=True)
    plt.axhline(y=initial_balance, color='grey', linestyle='--', label=f'Initial Balance (${initial_balance:,.2f})')
    
    plt.title(f'Equity Curve - {gain_pct}% Gain ({interval})', fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Account Value ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    try:
        plt.savefig(chart_filename)
        print(f"  -> Equity chart saved for {gain_pct}% {interval}")
    except Exception as e:
        print(f"  -> Could not save chart for {gain_pct}% {interval}: {e}")
    finally:
        plt.close()
        
    report = f"""
======================================================
     Volatility Breakout Strategy Report ({gain_pct}% Gain, {interval})
======================================================
NOTE: A 5% commission/slippage handicap has been applied to all gross profits.
      Risk per trade is a static ${RISK_PER_TRADE:.2f}.

--- Account Summary ---
Initial Balance: ${initial_balance:,.2f}
Final Balance:   ${final_balance:,.2f}
Net Profit:      ${net_profit_dollars:,.2f}
Total Return:    {total_return_pct:.2f}%

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
Net Profit  | ${net_profit_dollars:15,.2f} | {total_pnl_in_R:12.2f}R
Avg. Win    | ${avg_win_dollars:15,.2f} | {avg_win_R:12.2f}R
Avg. Loss   | ${avg_loss_dollars:15,.2f} | {avg_loss_R:12.2f}R
Expectancy  | ${expectancy_dollars:15,.2f} | {expectancy_R:12.2f}R
------------------------------------------------------

--- Key Metrics ---
Max Drawdown: ${max_drawdown_dollars:,.2f} ({(max_drawdown_dollars/initial_balance)*100:.2f}% of initial balance)
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
    report_dir = os.path.join('GAP_FADE', OUTPUT_DIR_NAME, 'reports_txt')
    os.makedirs(report_dir, exist_ok=True)
    report_filename = os.path.join(report_dir, f'performance_report_{gain_pct}pct_{interval}.txt')
    with open(report_filename, 'w') as f:
        f.write(report)
    print(f"  -> Report saved for {gain_pct}% {interval}")


def run_single_parameter_backtest(args):
    gain_pct, interval, candidate_df, position = args
    gain_threshold = gain_pct / 100.0
    log_dir = os.path.join('GAP_FADE', OUTPUT_DIR_NAME, 'trade_logs_csv')
    os.makedirs(log_dir, exist_ok=True)
    all_trades = []
    
    desc = f"{gain_pct}% Gain ({interval})"
    for _, row in tqdm(candidate_df.iterrows(), total=candidate_df.shape[0], desc=desc, position=position, leave=False):
        date_str = row['date'].strftime('%Y-%m-%d')
        trades = run_volatility_backtest(row['ticker'], date_str, gain_threshold, RISK_PER_TRADE, interval)
        if trades:
            all_trades.extend(trades)

    if all_trades:
        results_df = pd.DataFrame(all_trades)
        filename = os.path.join(log_dir, f'backtest_results_{gain_pct}pct_{interval}.csv')
        results_df.to_csv(filename, index=False)
        generate_performance_report(results_df.copy(), gain_threshold, interval, INITIAL_ACCOUNT_BALANCE)
        
    return f"Completed {gain_pct}% ({interval})"

if __name__ == '__main__':
    try:
        full_candidate_df = pd.read_csv(FILTERED_STOCKS_FILE)
    except FileNotFoundError:
        print(f"Error: The file {FILTERED_STOCKS_FILE} was not found.")
        exit()
    full_candidate_df['date'] = pd.to_datetime(full_candidate_df['date'])
    unique_candidates_df = full_candidate_df.sort_values(by='date').drop_duplicates(subset=['ticker', 'date']).reset_index(drop=True)
    test_candidates_df = unique_candidates_df
    print(f"--- Preparing to run a backtest on all {len(test_candidates_df)} candidates. ---")
    gain_percentages = list(range(50, 151, 10))
    intervals_to_test = ['1min', '2min', '3min', '5min']
    jobs = []
    position_counter = 0
    for gp in gain_percentages:
        for interval in intervals_to_test:
            jobs.append((gp, interval, test_candidates_df, position_counter))
            position_counter += 1

    print(f"--- Starting {len(jobs)} backtest job(s). ---")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(run_single_parameter_backtest, jobs)
        completed_tasks = list(results)
    print("\n\n--- All Backtests Complete! ---")
    for res in completed_tasks:
        print(f"  - {res}")