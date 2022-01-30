from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime
import pandas_ta as pta
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import os, sys
from loguru import logger

client = Client("", "")


class Trader:
    def __init__(self, file):
        self.connect(file)

    """ Creates Binance client """

    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    """ Gets all account balances """

    def getBalances(self):
        prices = self.client.get_withdraw_history()
        return prices


filename = 'config/config.py'
trader = Trader(filename)
TICKERS = 'tickers_all.txt'
DEBUG = True
WAVETREND_1h = False
CMO_1h = False
MACD_1h = False


def get_symbols():
    response = requests.get('https://api.binance.com/api/v3/ticker/price')
    PAIRS_WITH = 'USDT'
    ignore = ['UP', 'DOWN', 'AUD', 'BRL', 'BVND', 'BCC', 'BCHABC', 'BCHSV', 'BEAR', 'BNBBEAR', 'BNBBULL',
              'BULL', 'BUSD', 'TUSD', 'BKRW', 'DAI', 'ERD', 'EUR', 'USDS', 'USDC', 'HC', 'LEND', 'MCO', 'GBP', 'RUB',
              'TRY', 'NPXS', 'PAX', 'STORM', 'USDP', 'VEN', 'UAH', 'NGN', 'VAI', 'STRAT', 'SUSD', 'XZC']
    symbols = []

    for symbol in response.json():
        if PAIRS_WITH in symbol['symbol'] and all(item not in symbol['symbol'] for item in ignore):
            if symbol['symbol'][-len(PAIRS_WITH):] == PAIRS_WITH:
                symbols.append(symbol['symbol'])
            symbols.sort()
    return symbols


with open(TICKERS) as f:
    trading_pairs = f.read().splitlines()

# trading_pairs = get_symbols()

filtered_pairs0 = []
filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
selected_pair = []
selected_pairCMO = []


# def filter(pair):
#     interval = '1d'
#     symbol = pair
#     start_str = '3 months ago UTC'
#     end_str = f'{datetime.now()}'
#     # print(f"Fetching new bars for {datetime.now().isoformat()}")
#     df = pd.DataFrame(client.get_historical_klines(symbol, interval, start_str, end_str)[:-1]).astype(float)
#     df = df.iloc[:, :6]
#     df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
#     df = df.set_index('timestamp')
#     df.index = pd.to_datetime(df.index, unit='ms')
#     n1 = 10
#     n2 = 21
#     ap = pta.hlc3(df.high, df.low, df.close)
#     esa = pta.ema(ap, n1)
#     d = pta.ema(abs(ap - esa), n1)
#     ci = (ap - esa) / (0.015 * d)
#     wt1 = pta.ema(ci, n2)
#     print("on 1d timeframe " + symbol)
#     print(f'wt1: {wt1.iat[-1]}')
#
#     if wt1.iat[-1] < -60:
#         filtered_pairs0.append(symbol)
#         print('found')
#     else:
#         print('searching')

@logger.catch
def filter1(pair):
    interval = '1h'
    symbol = pair
    # klines = trader.client.get_klines(symbol=symbol, interval=interval)
    klines = client.get_klines(symbol=symbol, interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    low = [float(entry[3]) for entry in klines]
    high = [float(entry[2]) for entry in klines]
    open = [float(entry[1]) for entry in klines]
    close_array = np.asarray(close)
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
    cmo = pta.cmo(close_series, talib=False)
    macdh = pta.macd(close_series)['MACDh_12_26_9']

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99
    best_fit_line2 = np.poly1d(np.polyfit(y, x + np.std(x), 1))(y)
    best_fit_line3 = np.poly1d(np.polyfit(y, x - np.std(x), 1))(y)

    # print("on 1h timeframe " + symbol)

    if CMO_1h and not WAVETREND_1h and not MACD_1h:
        if (cmo.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]) | \
                (cmo.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')
            print(f'cmo: {cmo.iat[-2]}')

            plt.figure(figsize=(8, 6))
            plt.grid(True)
            plt.plot(x)
            plt.title(label=f'{symbol}', color="green")
            plt.plot(best_fit_line1, '--', color='r')
            plt.plot(best_fit_line2, '--', color='r')
            plt.plot(best_fit_line3, '--', color='green')
            plt.show(block=False)
            plt.pause(6)
            plt.close()

    elif CMO_1h and WAVETREND_1h and not MACD_1h:
        if cmo.iat[-1] < -60 and wt1.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= \
                best_fit_line1[-1]:
            filtered_pairs1.append(symbol)
            if DEBUG:
                print('found')
                print(f'on {interval} timeframe {symbol}')
                print(f'cmo: {cmo.iat[-1]}')
                print(f'wt1: {wt1.iat[-1]}')

        elif cmo.iat[-1] < -60 and wt1.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= \
                best_fit_line1[-1]:
            filtered_pairs1.append(symbol)
            if DEBUG:
                print('found')
                print(f'on {interval} timeframe {symbol}')
                print(f'cmo: {cmo.iat[-1]}')
                print(f'wt1: {wt1.iat[-1]}')

    elif CMO_1h and WAVETREND_1h and MACD_1h:
        if cmo.iat[-1] < -60 and wt1.iat[-1] < -60 and macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and \
                best_fit_line1[0] <= best_fit_line1[-1]:
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')
            print(f'cmo: {cmo.iat[-1]}')
            print(f'wt1: {wt1.iat[-1]}')
            print(f'macdh: {macdh.iat[-1]}')
        elif cmo.iat[-1] < -60 and wt1.iat[-1] < -60 and macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and \
                best_fit_line1[0] >= best_fit_line1[-1]:
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')
            print(f'cmo: {cmo.iat[-1]}')
            print(f'wt1: {wt1.iat[-1]}')
            print(f'macdh: {macdh.iat[-1]}')

    elif not CMO_1h and WAVETREND_1h and not MACD_1h:
        if wt1.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]:
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')
            print(f'wt1: {wt1.iat[-1]}')
        elif wt1.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]:
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')
            print(f'wt1: {wt1.iat[-1]}')

    elif CMO_1h and MACD_1h and not WAVETREND_1h:
        if cmo.iat[-1] < -60 and macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= \
                best_fit_line1[-1]:
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')
            print(f'wt1: {wt1.iat[-1]}')
        elif cmo.iat[-1] < -60 and macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= \
                best_fit_line1[-1]:
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')
            print(f'wt1: {wt1.iat[-1]}')
    else:
        if (x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]) | \
                (x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')

            plt.figure(figsize=(8, 6))
            plt.grid(True)
            plt.plot(x)
            plt.title(label=f'{symbol}', color="green")
            plt.plot(best_fit_line1, '--', color='r')
            plt.plot(best_fit_line2, '--', color='r')
            plt.plot(best_fit_line3, '--', color='green')
            plt.show(block=True)


def filter2(filtered_pairs1):
    interval = '15m'
    symbol = filtered_pairs1
    # klines = trader.client.get_klines(symbol=symbol, interval=interval)
    klines = client.get_klines(symbol=symbol, interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    print("on 15min timeframe " + symbol)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

    if (x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]) | \
            (x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]):
        filtered_pairs2.append(symbol)
        print('found')
        print("on 15min timeframe " + symbol)

        plt.figure(figsize=(8, 6))
        plt.grid(True)
        plt.plot(close)
        plt.title(label=f'{symbol}', color="green")
        plt.plot(best_fit_line1, '--', color='r')
        plt.plot(best_fit_line2, '--', color='r')
        plt.plot(best_fit_line3, '--', color='r')
        plt.show(block=False)
        plt.pause(6)
        plt.close()


def filter3(filtered_pairs2):
    interval = '5m'
    symbol = filtered_pairs2
    # klines = trader.client.get_klines(symbol=symbol, interval=interval)
    klines = client.get_klines(symbol=symbol, interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)
    close_series = pd.Series(close)

    print("on 5m timeframe " + symbol)

    # min = ta.MIN(close_array, timeperiod=30)
    max = close_series.rolling(30).max()
    min = close_series.rolling(30).min()
    # max = ta.MAX(close_array, timeperiod=30)

    # real = ta.HT_TRENDLINE(close_array)
    # wcl = ta.WCLPRICE(max, min, close_array)

    # print(close[-1])
    # print(real[-1])

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

    plt.figure(figsize=(8, 6))
    plt.title(symbol)
    plt.grid(True)
    plt.plot(close)
    plt.title(label=f'{symbol} on {interval}', color="green")
    plt.plot(best_fit_line1, '--', color='r')
    plt.plot(best_fit_line2, '--', color='r')
    plt.plot(best_fit_line3, '--', color='r')
    plt.plot(close)
    plt.plot(min)
    plt.plot(max)
    # plt.plot(real)
    plt.show(block=False)
    plt.pause(10)
    plt.close()

    if (x[-1] < best_fit_line3[-1] and best_fit_line1[0] > best_fit_line1[-1]) | \
            (x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]):
        filtered_pairs3.append(symbol)
        print('found')

        plt.figure(figsize=(8, 6))
        plt.title(symbol)
        plt.grid(True)
        plt.plot(close)
        plt.title(label=f'{symbol} on {interval}', color="green")
        plt.plot(best_fit_line1, '--', color='r')
        plt.plot(best_fit_line2, '--', color='r')
        plt.plot(best_fit_line3, '--', color='r')
        plt.plot(close)
        plt.plot(min)
        plt.plot(max)
        # plt.plot(real)
        plt.show(block=False)
        plt.pause(10)
        plt.close()

    else:
        print('searching')


def momentum(filtered_pairs3):
    interval = '1m'
    symbol = filtered_pairs3
    # klines = client.get_klines(symbol=symbol, interval=interval)
    # open_time = [int(entry[0]) for entry in klines]
    # close = [float(entry[4]) for entry in klines]
    # close_array = pd.Series(close)
    # real = pta.cmo(close_array, talib=False)

    start_str = '24 hours ago UTC'
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
    print("on 1m timeframe " + symbol)
    print(f'cmo: {real.iat[-1]}')
    print(f'wt1: {wt1.iat[-1]}')

    if real.iat[-1] < -50 and wt1.iat[-1] < -60:
        print('oversold dip found')
        selected_pair.append(symbol)

    return selected_pair


# for i in trading_pairs:
#     output = filter(i)
#     print(filtered_pairs0)

for i in trading_pairs:
    output = filter1(i)
    # print(filtered_pairs1)

for i in filtered_pairs1:
    output = filter2(i)
    print(filtered_pairs2)

for i in filtered_pairs2:
    output = filter3(i)
    print(filtered_pairs3)

for i in filtered_pairs3:
    output = momentum(i)
    print(selected_pair)

if len(selected_pair) > 1:
    print('dips are more then 1 oversold')
    print(selected_pair)
    print(selected_pairCMO)

    if min(selected_pairCMO) in selected_pairCMO:
        print(selected_pairCMO.index(min(selected_pairCMO)))
        position = selected_pairCMO.index(min(selected_pairCMO))

    for id, value in enumerate(selected_pair):
        if id == position:
            print(selected_pair[id])
    sys.exit()

elif len(selected_pair) == 1:
    print('1 dip found')
    print(selected_pair)
    sys.exit()

else:
    print('no oversold dips for the moment, restart script...')
    print(selected_pair)
    print(selected_pairCMO)
    time.sleep(60)
    os.execl(sys.executable, sys.executable, *sys.argv)

# sys.exit(0)
# sys.exit() if KeyboardInterrupt else exit()
# exit()
