import pandas as pd
import numpy as np
import os

def analyze_trade_log(file_path):
    """
    Analyzes a backtest trade log to diagnose performance issues.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    print(f"--- Analyzing Trade Log: {file_path} ---")

    df = pd.read_csv(file_path)

    if df.empty:
        print("The trade log is empty. No trades to analyze.")
        return

    # --- Analysis ---
    total_trades = len(df)
    
    # 1. Initial Risk Analysis
    # Risk per share is calculated from the initial entry and stop, before averaging in.
    # We need to recalculate it based on avg_entry_price for a more accurate picture of the realized risk.
    df['realized_risk_per_share'] = df['stop_loss'] - df['entry_price']
    avg_initial_risk_dollars = (df['realized_risk_per_share'] / df['entry_price'] * 100).mean()

    # 2. Exit Type Distribution
    exit_distribution = df['status'].value_counts(normalize=True) * 100

    # 3. Profitability by Number of Chunks Added
    profit_by_chunks = df.groupby('chunks_added')['pnl_in_dollars'].agg(['sum', 'mean', 'count'])
    
    # --- Print Report ---
    print("\n--- Performance Diagnosis Report ---")
    print("========================================")
    print(f"Total Trades Analyzed: {total_trades}")
    print(f"\n1. Average Initial Risk: {avg_initial_risk_dollars:.2f}%")
    print("   - This is the average percentage move required to hit the initial stop-loss.")
    print("   - A high value here (>5-7%) suggests the entry trigger is too slow, leading to poor entry prices.")

    print("\n2. Distribution of Trade Outcomes:")
    for status, percentage in exit_distribution.items():
        print(f"   - {status}: {percentage:.2f}%")
    print("   - A high percentage of 'STOPPED_OUT' trades confirms the poor entry quality.")

    print("\n3. Profitability by Position Size (Chunks Added):")
    print(profit_by_chunks)
    print("   - This shows if the strategy is more profitable when it scales in.")
    print("   - If trades with more chunks are less profitable, it means the trend isn't continuing as expected after entry.")
    
    print("\n--- Conclusion ---")
    if avg_initial_risk_dollars > 5:
        print("Primary issue appears to be a LATE ENTRY. The strategy waits too long to enter, resulting in a large initial stop and poor risk/reward.")
        print("Suggestion: Tighten the entry trigger. Consider entering *as soon as* the price breaks the low of the peak candle, rather than waiting for a close.")
    elif exit_distribution.get('STOPPED_OUT', 0) > 60:
         print("Primary issue appears to be POOR TRADE QUALITY. Most trades are stopping out.")
         print("Suggestion: The setup conditions might be too loose. The entry trigger itself could also be the problem.")
    else:
        print("Analysis suggests a mix of factors. The strategy might need a more fundamental rethink on its entry or trade management rules.")

if __name__ == '__main__':
    # Correct path to the new trade log file for the fast entry test
    log_file = os.path.join('GAP_FADE', 'climactic_reversal_fast_entry', 'trade_logs_csv', 'backtest_results_50pct_1min.csv')
    analyze_trade_log(log_file) 