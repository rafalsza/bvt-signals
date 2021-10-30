# bvt-signals
---
bvt bot buy signals<br />

rs_signals_buy_dip.py  =  linear regression on 1h (+ cmo < -50) ,15m,5m + ( cmo < -50 and wavetrend < -60 ) on 1m

recommended settings

config.yml

```
CUSTOM_LIST: False
TIME_DIFFERENCE: 1
RECHECK_INTERVAL: 3
CHANGE_IN_PRICE: 100
STOP_LOSS: 100
TAKE_PROFIT: 2
USE_TRAILING_STOP_LOSS: True
TRAILING_STOP_LOSS: .8
TRAILING_TAKE_PROFIT: .1
TICKERS_LIST: 'tickers_all.txt'
SIGNALLING_MODULES:
- rs_signals_buy_dip
```
