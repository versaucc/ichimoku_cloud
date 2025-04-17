import pandas as pd

def find_fdp(line1, line2):
    """Find the last index where line1 and line2 differ before a crossover."""
    for i in range(len(line1) - 2, -1, -1):  
        if line1.iloc[i] != line2.iloc[i]:  
            return i
    return None  # No differing point found

def is_crossing(line1, line2, timestamps):
    """
    Check if line1 crosses above or below line2 based on the last differing point (FDP).

    A BUY signal occurs when the Tenkan-sen crosses above the Kijun-sen.
    A SELL signal occurs when the Kijun-sen crosses above the Tenkan-sen.
    """

    df = pd.DataFrame({"timestamp": timestamps, "line1": line1, "line2": line2}).dropna()

    if len(df) < 2:
        return None

    fdp = find_fdp(df["line1"], df["line2"])

    if fdp is None or fdp >= len(df) - 1:
        return None

    prev_time, prev_line1, prev_line2 = df.iloc[fdp]
    curr_time, curr_line1, curr_line2 = df.iloc[fdp + 1]

    crossing_up = prev_line1 < prev_line2 and curr_line1 > curr_line2  # Tenkan crosses above Kijun (BUY)
    crossing_down = prev_line1 > prev_line2 and curr_line1 < curr_line2  # Kijun crosses above Tenkan (SELL)

    if crossing_up:
        return "BUY", curr_time
    elif crossing_down:
        return "SELL", curr_time
    else:
        return None

def check_recent_crossovers(csv_file, lookback=60):
    """
    Reads a CSV file and checks for recent crossovers, while tracking P&L.
    :param csv_file: Path to CSV file.
    :param lookback: Number of periods to check.
    """

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        return
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "tenkan" not in df or "kijun" not in df or "high" not in df or "low" not in df:
        print("Missing required columns (tenkan, kijun, High, Low).")
        return

    df["price"] = (df["high"] + df["low"]) / 2  # Use midpoint price for P&L calculations

    recent_data = df.tail(lookback)

    position = None
    entry_price = 0
    total_pnl = 0

    for i in range(1, len(recent_data)):
        signal = is_crossing(
            recent_data["tenkan"].iloc[i-1:i+1], 
            recent_data["kijun"].iloc[i-1:i+1], 
            recent_data["timestamp"].iloc[i-1:i+1]
        )

        if signal:
            action, timestamp = signal
            price = recent_data["price"].iloc[i]

            if action == "BUY" and position is None:
                position = "LONG"
                entry_price = price
                print(f"BUY at {price} on {timestamp}")

            elif action == "SELL" and position == "LONG":
                pnl = price - entry_price
                total_pnl += pnl
                print(f"SELL at {price} on {timestamp} | P&L: {pnl:.2f}")
                position = None

    print(f"Total P&L: {total_pnl:.2f}")

# Run the script
csv_file = "data\ohlcv_data_1h_BTC.csv"  
check_recent_crossovers(csv_file)