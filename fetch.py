from eth_account.signers.local import LocalAccount
import eth_account 
import json, time, ccxt, schedule, requests, os
from hyperliquid.info import Info 
from hyperliquid.exchange import Exchange 
from hyperliquid.utils import constants 
import pandas as pd 
import datetime 
from datetime import datetime, timedelta
import pandas_ta as ta
from dotenv import load_dotenv
import utils as ut 

load_dotenv()

secret_key = os.getenv("hl_secret")

# Rough draft
def fetch(symbol, timeframe, lookback_days): 
    output = ut.get_ohlcv2(symbol, timeframe, lookback_days)

    