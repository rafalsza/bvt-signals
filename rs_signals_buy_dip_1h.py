"""
CUSTOM_LIST: False
"""
from binance.client import Client
import numpy as np
import threading
import os
import pandas_ta as pta
import pandas as pd
from datetime import datetime
import time

client = Client("", "")
TIME_TO_WAIT = 1  # Minutes to wait between analysis
DEBUG = False
TICKERS = 'tickers_all.txt'
SIGNAL_NAME = 'rs_signals_buy_dip'
SIGNAL_FILE_BUY = 'signals/' + SIGNAL_NAME + '.buy'


# for colourful logging to the console
class txcolors:
    BUY = '\033[92m'
    WARNING = '\033[93m'
    SELL_LOSS = '\033[91m'
    SELL_PROFIT = '\033[32m'
    DIM = '\033[2m\033[35m'
    DEFAULT = '\033[39m'


def get_symbols():
    response = client.get_ticker()
    PAIRS_WITH = 'USDT'
    ignore = ['UP', 'DOWN', 'AUD', 'BRL', 'BVND', 'BUSD', 'BCC', 'BCHABC', 'BCHSV', 'BEAR', 'BNBBEAR', 'BNBBULL',
              'BULL',
              'BKRW', 'DAI', 'ERD', 'EUR', 'USDS', 'HC', 'LEND', 'MCO', 'GBP', 'RUB', 'TRY', 'NPXS', 'PAX', 'STORM',
              'VEN', 'UAH', 'USDC', 'NGN', 'VAI', 'STRAT', 'SUSD', 'XZC', 'RAD']
    symbols = []

    for symbol in response:
        if PAIRS_WITH in symbol['symbol'] and all(item not in symbol['symbol'] for item in ignore):
            if symbol['symbol'][-len(PAIRS_WITH):] == PAIRS_WITH:
                symbols.append(symbol['symbol'])
            symbols.sort()
    # symbols = [sub[: -4] for sub in symbols]   # without USDT
    return symbols


filtered_pairs1 = []
filtered_pairs2 = []
selected_pair = []


def filter1(pairs):
    interval = '1h'
    symbol = pairs
    klines = client.get_klines(symbol=symbol, interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    low = [float(entry[3]) for entry in klines]
    high = [float(entry[2]) for entry in klines]
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)

    n1 = 10
    n2 = 21
    ap = pta.hlc3(high_series, low_series, close_series)
    esa = pta.ema(ap, n1)
    d = pta.ema(abs(ap - esa), n1)
    ci = (ap - esa) / (0.015 * d)
    wt1 = pta.ema(ci, n2)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = np.poly1d(np.polyfit(y, x + np.std(x), 1))(y)
    best_fit_line3 = np.poly1d(np.polyfit(y, x - np.std(x), 1))(y)

    if wt1.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and \
            best_fit_line1[0] <= best_fit_line1[-1] | wt1.iat[-1] < -60 \
            and x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
        filtered_pairs1.append(symbol)
        if DEBUG:
            print('found')
            print("on 1h timeframe " + symbol)
            print(f'wt1: {wt1.iat[-1]}')

    return filtered_pairs1


def filter2(filtered_pairs):
    interval = '5m'
    symbol = filtered_pairs
    klines = client.get_klines(symbol=symbol, interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

    if x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]:
        filtered_pairs2.append(symbol)
        if DEBUG:
            print("on 5min timeframe " + symbol)

    elif x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
        filtered_pairs2.append(symbol)
        if DEBUG:
            print("on 5min timeframe " + symbol)

    return filtered_pairs2


def momentum(filtered_pairs):
    interval = '1m'
    symbol = filtered_pairs
    # klines = client.get_klines(symbol=symbol, interval=interval)
    # open_time = [int(entry[0]) for entry in klines]
    # close = [float(entry[4]) for entry in klines]
    # close_array = pd.Series(close)
    # real = pta.cmo(close_array, talib=False)

    start_str = '12 hours ago UTC'
    end_str = f'{datetime.now()}'
    # print(f"Fetching new bars for {datetime.now().isoformat()}")
    df = pd.DataFrame(client.get_historical_klines(symbol, interval, start_str, end_str)[:-1]).astype(float)
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index, unit='ms')
    # CMO
    real = pta.cmo(df.close, talib=False)
    # WaveTrend
    n1 = 10
    n2 = 21
    ap = pta.hlc3(df.high, df.low, df.close)
    esa = pta.ema(ap, n1)
    d = pta.ema(abs(ap - esa), n1)
    ci = (ap - esa) / (0.015 * d)
    wt1 = pta.ema(ci, n2)
    #
    print("on 1h timeframe " + symbol)
    print(f'cmo: {real.iat[-1]}')
    print(f'wt1: {wt1.iat[-1]}')

    if real.iat[-1] < -50 and wt1.iat[-1] < -60:
        print('oversold dip found')
        selected_pair.append(symbol)

    return selected_pair


def analyze(trading_pairs):
    signal_coins = {}
    filtered_pairs1.clear()
    filtered_pairs2.clear()
    selected_pair.clear()

    if os.path.exists(SIGNAL_FILE_BUY):
        os.remove(SIGNAL_FILE_BUY)

    for i in trading_pairs:  # 1h
        output = filter1(i)
        # print(filtered_pairs1)

    for i in filtered_pairs1:  # 15m
        output = filter2(i)
        if DEBUG:
            print(output)

    for i in filtered_pairs2:  # 1m
        output = momentum(i)
        print(output)

    for pair in selected_pair:
        signal_coins[pair] = pair
        with open(SIGNAL_FILE_BUY, 'a+') as f:
            f.writelines(pair + '\n')

    if selected_pair:
        print(f'{txcolors.BUY}{SIGNAL_NAME}: {selected_pair} - Buy Signal Detected{txcolors.DEFAULT}')
    else:
        print(f'{txcolors.DEFAULT}{SIGNAL_NAME}: - not enough signal to buy')
    return signal_coins


def do_work():
    while True:
        try:
            if not os.path.exists(TICKERS):
                time.sleep((TIME_TO_WAIT * 60))
                continue

            signal_coins = {}
            pairs = {}
            with open(TICKERS) as f:
                pairs = f.read().splitlines()

            # pairs = get_symbols()

            if not threading.main_thread().is_alive():
                exit()
            print(f'{SIGNAL_NAME}: Analyzing {len(pairs)} coins')
            signal_coins = analyze(pairs)
            print(
                f'{SIGNAL_NAME}: {len(signal_coins)} '
                f'coins with Buy Signals. Waiting {TIME_TO_WAIT} minutes for next analysis.')

            time.sleep((TIME_TO_WAIT * 60))
        except Exception as e:
            print(f'{SIGNAL_NAME}: Exception do_work() 1: {e}')
            continue
        except KeyboardInterrupt as ki:
            continue
