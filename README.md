# bvt-signals
---
bvt bot buy signals<br />

rs_signals_buy_dip.py  =  linear regression on 1h (+ cmo < -60 or wavetrend < -60 or macdh >0) ,15m,5m,1m ( cmo < -50 and wavetrend < -60 )

recommended settings

config.yml

```
CUSTOM_LIST: False
TIME_DIFFERENCE: 1
RECHECK_INTERVAL: 4
CHANGE_IN_PRICE: 100
STOP_LOSS: 100
TAKE_PROFIT: 3
USE_TRAILING_STOP_LOSS: True
TRAILING_STOP_LOSS: .8
TRAILING_TAKE_PROFIT: .1
TICKERS_LIST: 'tickers_all.txt'
SIGNALLING_MODULES:
- rs_signals_buy_dip
```

rs_signals_buy_dip.py
```
DEBUG = True
CMO_1h = True
WAVETREND_1h = True
MACD_1h = False
```
