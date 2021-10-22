# bvt-signals
---
bvt bot buy signals<br />
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
