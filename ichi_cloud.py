import dontshare as d
import utils as n

import eth_account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

import json, time, requests, schedule, pytz, os, sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv


symbol = 'BTC'
timeframe = '1h'
lookback_days = 60 #  used always
size = 1 # used always
target = 5 # Not used atm 
max_loss = -10 # Not used atm 
leverage = 1 # used always 
max_positions = 1 # Not used atm 
init = False 

load_dotenv()
secret = os.getenv("hl_secret")

if secret is None: 
    print("Error fetching hl secret key. Check your .env path!")


def initialize(timeframe): 
    
    # Convert current UTC time to Los Angeles time
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    la_time = utc_now.astimezone(pytz.timezone("America/Los_Angeles"))
    print(f"Current Time in Los Angeles: {la_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Intro screen 
    print('\n\n=======================================')
    print('Ichimoku Cloud Strategy Initializing')
    print(f'Parameters:\nSymbol: {symbol}\nTimeframe: {timeframe}\nLeverage: {leverage}')
    print('=======================================\n\n')

    #Initialize account object and adjust leverage
    account = LocalAccount = eth_account.Account.from_key(secret) 
    n.adjust_leverage_size_signal(symbol, leverage, account)

    # Select file to read and append ohlcv data 
    if timeframe == '1d':
        data_file = 'data\ohlcv_data_1d_BTC.csv'
    elif timeframe == '1h':
        data_file = 'data\ohlcv_data_1h_BTC.csv'
    elif timeframe == '30m':
        data_file = 'data\ohlcv_data_30m_BTC.csv'
    elif timeframe == '15m':
        data_file = 'data\ohlcv_data_15m_BTC.csv'

    return data_file, account

#Fetch open, high, low, close data for symbol and lookback days
def get_ohlcv2(symbol, interval, lookback_days):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000)
        }
    }

    # Historical data updates 
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        snapshot_data = response.json()
        return snapshot_data
    else:
        print(f"Error fetching data for {symbol}: {response.status_code}")
        return None


def process_data_to_df(snapshot_data):
    try: 
        if snapshot_data:

            # Build DataFrame out of ohlcv data
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            data = []
            for snapshot in snapshot_data:
                timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                open_price = snapshot['o']
                high_price = snapshot['h']
                low_price = snapshot['l']
                close_price = snapshot['c']
                volume = snapshot['v']
                data.append([timestamp, open_price, high_price, low_price, close_price, volume])

            df = pd.DataFrame(data, columns=columns)

            return df
        else:
            return pd.DataFrame()    
    except Exception as e: 
        print(f"[ERROR] building DataFrame from snapshot data: {e}")



#STRATEGY START: 
def ichimoku_tenkan(df, period=9):
    """Calculate the Tenkan-sen (Conversion Line) based on real timestamps."""
    
    if "high" not in df or "low" not in df:
        raise ValueError("DataFrame must contain 'high' and 'low' columns.")

    # Compute Tenkan-sen using rolling window
    df["tenkan"] = (df["high"].rolling(period).max() + df["low"].rolling(period).min()) / 2

    return df[["timestamp", "tenkan"]]  # Return timestamp + tenkan for accuracy

# Tenkan-sen (Conversion Line): This is calculated as the average of the highest high and the lowest low over the past 9 periods. It is a short-term indicator of market trend

def ichimoku_kijun(df, period=26):
    """Calculate the Kijun-sen line (Base Line) based on real timestamps."""
    
    if "high" not in df or "low" not in df:
        raise ValueError("DataFrame must contain 'high' and 'low' columns.")

    df["kijun"] = (df["high"].rolling(period).max() + df["low"].rolling(period).min()) / 2

    return df[["timestamp", "kijun"]]  # Return timestamp + kijun for accuracy

# Kijun-sen (Base Line): This is calculated as the average of the highest high and the lowest low over the past 26 periods. It serves as a medium-term indicator of market trend.

def find_fdp(line1, line2):
    """Find the last index where line1 and line2 differ before a crossover."""
    for i in range(len(line1) - 2, -1, -1):  # Start from the second last index
        if line1.iloc[i] != line2.iloc[i]:  # Use `.iloc[]` for safe indexing
            return i
    return None  # No differing point found

def is_crossing(line1, line2, timestamps, last_signal=None):
    """
    Check if line1 crosses above or below line2 based on the last differing point (FDP).

    A BUY signal occurs when the Tenkan-sen crosses above the Kijun-sen.
    A SELL signal occurs when the Kijun-sen crosses above the Tenkan-sen.
    """

    #  Ensure timestamps align with Tenkan & Kijun values
    df = pd.DataFrame({"timestamp": timestamps, "line1": line1, "line2": line2}).dropna()

    #  Ensure we have enough data
    if len(df) < 2:
        print(" Not enough data for crossover check.")
        return None

    # Find the last differing point (FDP)
    fdp = find_fdp(df["line1"], df["line2"])

    # Ensure FDP is valid
    if fdp is None or fdp >= len(df) - 1:
        return None  # No valid crossover detected

    # Extract values at the FDP and the next timestamp for crossover check
    prev_time, prev_line1, prev_line2 = df.iloc[fdp]
    curr_time, curr_line1, curr_line2 = df.iloc[fdp + 1]  # Next period

    # Check for crossover conditions
    crossing_up = prev_line1 < prev_line2 and curr_line1 > curr_line2  # Tenkan crosses above Kijun (BUY)
    crossing_down = prev_line1 > prev_line2 and curr_line1 < curr_line2  # Kijun crosses above Tenkan (SELL)

    # Allow valid signals, but ensure proper alternation
    if crossing_up:
        print('\n============================================================')
        print(f"BUY Signal - Tenkan crossed above Kijun at {curr_time}")
        print('============================================================\n')
        return "BUY"
    elif crossing_down:
        print('\n============================================================')
        print(f" SELL Signal - Kijun crossed above Tenkan at {curr_time}")
        print('============================================================\n')
        return "SELL"
    else:
        return None  # No crossover detected

# A buy signal is occuring when the Tenkan-sen crosses above the Kijun-sen. 
# A sell signal is occuring when the Kijun-sen crosses above the Tenkan-sen. 

#END STRATEGY

#For orders
def ask_bid(symbol):
    #returns ask and bid price 

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        'type': 'l2Book',
        'coin': symbol
        }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']

    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data

def get_sz_px_decimals(coin):
    """Returns size and price decimal precision required by Hyperliquid."""
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == coin), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
        else:
            print(f"Symbol {coin} not found.")
            return None, None  # Return None to avoid errors later

    else:
        print("Error getting symbol info")
        return None, None

    # Get Ask price and determine price decimals
    ask = ask_bid(coin)[0]

    ask_str = str(ask)
    px_decimals = len(ask_str.split('.')[1]) if '.' in ask_str else 0

    print(f"{coin} Price has {px_decimals} decimals, Size has {sz_decimals} decimals.")

    return sz_decimals, px_decimals

def limit_order(coin, is_buy, limit_px, sz, reduce_only, account):

    exchange = Exchange(account, constants.MAINNET_API_URL)  
    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz, rounding)
    print(f'coin: {coin}, type: {type(coin)}')
    print(f'sz: {sz}, type: {type(limit_px)}')
    print(f'reduce_only: {reduce_only}, type: {type(reduce_only)}')

    print(f'placing limit order for {sz} {coin} at {limit_px}\n\n')
    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": 'Gtc'}})

    if is_buy == True: 
        status = order_result['response']['data']['statuses'][0]  # Assign to a variable
        print('\n=============================================')
        print(f'limit BUY order placed, resting: {status}')
        print('=============================================\n')
    else: 
        status = order_result['response']['data']['statuses'][0]
        print('\n=============================================')
        print(f'limit SELL order place, resting: {status}\n\n')
        print('=============================================\n')
    return order_result


#Integration of the jap strat
def find_lines(data, position):

    data["high"] = pd.to_numeric(data["high"], errors="coerce")
    data["low"] = pd.to_numeric(data["low"], errors="coerce")

    tenkan_line = ichimoku_tenkan(data)
    kijun_line = ichimoku_kijun(data)

    print('Last 3 periods')
    print(pd.concat([tenkan_line.tail(3), kijun_line.tail(3)["kijun"]], axis=1))
       
    # Pass the timeframe (timestamp column) into is_crossing()
    signal = is_crossing(tenkan_line["tenkan"], kijun_line["kijun"], data["timestamp"])
    
    if position:
        if signal == 'SELL': # In position need to get out
            # self.position.close()
            print('selling...')
            return 'SELL'
        else:
            print('\nIn position, no sell signal\n')
            return None
    elif not position:
        if signal == 'BUY': # No position need to get in 
            # self.buy()
            print('buying...') 
            return 'BUY'
        else:
            print('\nNot in position, no buy signal\n')
            return None
    
        
def bot(init, timeframe):
    """Main trading loop that continuously fetches and processes OHLCV data."""
    if not init: 
        data_file, account = initialize(timeframe)
        init = True 
    
    active = True
    position = False
    latest_timestamp = None  # Track last known data point

    # Load previous data if available
    if os.path.exists(data_file):
        ohlcv_data = pd.read_csv(data_file)
        if not ohlcv_data.empty:
            ohlcv_data["timestamp"] = pd.to_datetime(ohlcv_data["timestamp"])
            latest_timestamp = ohlcv_data["timestamp"].max()
            print(f"Loaded backup data up to: {latest_timestamp}")
        else:
            ohlcv_data = pd.DataFrame()
    else:
        ohlcv_data = pd.DataFrame()

    while active:
        new_data = process_data_to_df(get_ohlcv2(symbol, timeframe, lookback_days))
        
        if not new_data.empty:
            new_data["timestamp"] = pd.to_datetime(new_data["timestamp"])

            # Ensure we are getting only new data
            if latest_timestamp is None or new_data["timestamp"].max() > latest_timestamp:
                latest_timestamp = new_data["timestamp"].max()  # Update latest timestamp

                # Append new data and remove duplicates
                ohlcv_data = pd.concat([ohlcv_data, new_data]).drop_duplicates(subset=["timestamp"])

                # Filter NaN 


                # Save updated data as backup
                ohlcv_data.to_csv(data_file, index=False)
                print(f"Data updated. Latest timestamp: {latest_timestamp}\n\n")

            else:
                print("No new data found. Waiting...\n\n")

            # Find lines
            signal = find_lines(ohlcv_data, position)
            
            if signal == "BUY" and not position:
                reduce_only = False
                ask, bid, l2 = ask_bid(symbol)
                is_buy = True
                limit_order(symbol, is_buy, bid, size, reduce_only, account)  # Adjust size as needed
                position = True  # Mark as in position

            elif signal == "SELL" and position:
                reduce_only = True
                ask, bid, l2 = ask_bid(symbol)
                is_buy = False
                limit_order(symbol, is_buy, ask, size, reduce_only, account)
                position = False  # Mark as out of position
            
            positions1, im_in_pos, mypos_size, pos_sym1, entry_px1, pnl_perc1, long1 = n.get_position(symbol, account)
            print(f'Current positions {positions1}\n\n')
            print('waiting\n')
            time.sleep(3600)

        else:
            print(" No data found. Retrying in 10 seconds...")
            time.sleep(10)


bot(init, timeframe)



    
    


