## Automated trading bot and backtesting environment for the Ichomoku Cloud indicator on the Hyperliquid network

Inlcudes data fetch, backtesting, and live trading. 

Requirements: 
    -Packages (requirments.txt)
    -Hyperliquid API key

```
pip install -r reqiuirements.txt
```
```
cp .env.example .env
```

Replace with your keys 

## Usage:
To fetch data: 
Edit parameters in fetch.py to desired symbol/timeframe
```
python fetch.py
```

To backtest: 
Edit parameters in backtest.py to desired csv
```
python backtest.py
```

To run live: 
Edit parameters in ichi_cloud.py to desired timeframe, data file, symbol
```
python ichi_cloud.py
```