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
from loguru import logger
import matplotlib.pyplot as plt

client = Client("", "")
TIME_TO_WAIT = 1  # Minutes to wait between analysis
DEBUG = False
TICKERS = 'tickers_all.txt'
SIGNAL_NAME = 'rs_signals_buy_dip'
SIGNAL_FILE_BUY = 'signals/' + SIGNAL_NAME + '.buy'

CMO_1h = True
WAVETREND_1h = True
MACD_1h = False


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
filtered_pairs3 = []
selected_pair = []
selected_pairCMO = []


@logger.catch
def filter1(pair):
    interval = '1h'
    symbol = pair
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
    if symbol == 'BTCUSDT':
        n1 = 23
        n2 = 17
    elif symbol == '1INCHUSDT':
        n1 = 12
        n2 = 29
    elif symbol == 'AAVEUSDT':
        n1 = 22
        n2 = 17
    elif symbol == 'ACMUSDT':
        n1 = 20
        n2 = 19
    elif symbol == 'ADAUSDT':
        n1 = 11
        n2 = 11
    elif symbol == 'AIONUSDT':
        n1 = 12
        n2 = 16
    elif symbol == 'AKROUSDT':
        n1 = 11
        n2 = 20
    elif symbol == 'ALGOUSDT':
        n1 = 21
        n2 = 12
    elif symbol == 'ALICEUSDT':
        n1 = 13
        n2 = 11
    elif symbol == 'ALPHAUSDT':
        n1 = 18
        n2 = 39
    elif symbol == 'ANKRUSDT':
        n1 = 37
        n2 = 39
    elif symbol == 'ARUSDT':
        n1 = 13
        n2 = 13
    elif symbol == 'ARDRUSDT':
        n1 = 15
        n2 = 18
    elif symbol == 'ARPAUSDT':
        n1 = 22
        n2 = 39
    elif symbol == 'ASRUSDT':
        n1 = 18
        n2 = 36
    elif symbol == 'ATAUSDT':
        n1 = 13
        n2 = 16
    elif symbol == 'ATMUSDT':
        n1 = 13
        n2 = 35
    elif symbol == 'ATOMUSDT':
        n1 = 39
        n2 = 24
    elif symbol == 'AUDIOUSDT':
        n1 = 33
        n2 = 31
    elif symbol == 'AUTOUSDT':
        n1 = 27
        n2 = 37
    elif symbol == 'AVAUSDT':
        n1 = 10
        n2 = 10
    elif symbol == 'AVAXUSDT':
        n1 = 20
        n2 = 30
    elif symbol == 'AXSUSDT':
        n1 = 10
        n2 = 18
    elif symbol == 'BADGERUSDT':
        n1 = 20
        n2 = 23
    elif symbol == 'BAKEUSDT':
        n1 = 16
        n2 = 34
    elif symbol == 'BALUSDT':
        n1 = 10
        n2 = 18
    elif symbol == 'BANDUSDT':
        n1 = 21
        n2 = 36
    elif symbol == 'BARUSDT':
        n1 = 10
        n2 = 15
    elif symbol == 'BATUSDT':
        n1 = 12
        n2 = 23
    elif symbol == 'BCHUSDT':
        n1 = 10
        n2 = 23
    elif symbol == 'BEAMUSDT':
        n1 = 11
        n2 = 24
    elif symbol == 'BELUSDT':
        n1 = 15
        n2 = 20
    elif symbol == 'BLZUSDT':
        n1 = 27
        n2 = 22
    elif symbol == 'BNBUSDT':
        n1 = 15
        n2 = 31
    elif symbol == 'BTGUSDT':
        n1 = 13
        n2 = 12
    elif symbol == 'BURGERUSDT':
        n1 = 16
        n2 = 37
    elif symbol == 'C98USDT':
        n1 = 14
        n2 = 19
    elif symbol == 'CAKEUSDT':
        n1 = 14
        n2 = 32
    elif symbol == 'CELOUSDT':
        n1 = 31
        n2 = 22
    elif symbol == 'CELRUSDT':
        n1 = 24
        n2 = 18
    elif symbol == 'CFXUSDT':
        n1 = 10
        n2 = 14
    elif symbol == 'CHZUSDT':
        n1 = 12
        n2 = 35
    elif symbol == 'CHRUSDT':
        n1 = 28
        n2 = 12
    elif symbol == 'COMPUSDT':
        n1 = 15
        n2 = 13
    elif symbol == 'CTKUSDT':
        n1 = 10
        n2 = 28
    elif symbol == 'DASHUSDT':
        n1 = 17
        n2 = 37
    elif symbol == 'DOGEUSDT':
        n1 = 14
        n2 = 37
    elif symbol == 'DOTUSDT':
        n1 = 17
        n2 = 29
    elif symbol == 'DCRUSDT':
        n1 = 18
        n2 = 37
    elif symbol == 'EGLDUSDT':
        n1 = 10
        n2 = 15
    elif symbol == 'ENJUSDT':
        n1 = 11
        n2 = 11
    elif symbol == 'EOSUSDT':
        n1 = 17
        n2 = 39
    elif symbol == 'ETCUSDT':
        n1 = 20
        n2 = 24
    elif symbol == 'ETHUSDT':
        n1 = 23
        n2 = 13
    elif symbol == 'FILUSDT':
        n1 = 12
        n2 = 31
    elif symbol == 'FLOWUSDT':
        n1 = 25
        n2 = 37
    elif symbol == 'FTTUSDT':
        n1 = 10
        n2 = 27
    elif symbol == 'FTMUSDT':
        n1 = 15
        n2 = 32
    elif symbol == 'FXSUSDT':
        n1 = 13
        n2 = 38
    elif symbol == 'GRTUSDT':
        n1 = 26
        n2 = 28
    elif symbol == 'HBARUSDT':
        n1 = 17
        n2 = 11
    elif symbol == 'HNTUSDT':
        n1 = 12
        n2 = 10
    elif symbol == 'HOTUSDT':
        n1 = 15
        n2 = 27
    elif symbol == 'ICPUSDT':
        n1 = 10
        n2 = 27
    elif symbol == 'IOTAUSDT':
        n1 = 18
        n2 = 32
    elif symbol == 'KAVAUSDT':
        n1 = 20
        n2 = 19
    elif symbol == 'KEYUSDT':
        n1 = 19
        n2 = 14
    elif symbol == 'KNCUSDT':
        n1 = 24
        n2 = 10
    elif symbol == 'KSMUSDT':
        n1 = 18
        n2 = 39
    elif symbol == 'LINAUSDT':
        n1 = 12
        n2 = 29
    elif symbol == 'LINKUSDT':
        n1 = 22
        n2 = 39
    elif symbol == 'LSKUSDT':
        n1 = 12
        n2 = 24
    elif symbol == 'LTCUSDT':
        n1 = 11
        n2 = 15
    elif symbol == 'LTOUSDT':
        n1 = 11
        n2 = 10
    elif symbol == 'LRCUSDT':
        n1 = 28
        n2 = 39
    elif symbol == 'KLAYUSDT':
        n1 = 11
        n2 = 10
    elif symbol == 'LUNAUSDT':
        n1 = 29
        n2 = 29
    elif symbol == 'MANAUSDT':
        n1 = 16
        n2 = 11
    elif symbol == 'MATICUSDT':
        n1 = 25
        n2 = 35
    elif symbol == 'MINAUSDT':
        n1 = 21
        n2 = 20
    elif symbol == 'MKRUSDT':
        n1 = 15
        n2 = 19
    elif symbol == 'MTLUSDT':
        n1 = 11
        n2 = 20
    elif symbol == 'NEARUSDT':
        n1 = 38
        n2 = 23
    elif symbol == 'ONEUSDT':
        n1 = 18
        n2 = 37
    elif symbol == 'ONGUSDT':
        n1 = 35
        n2 = 29
    elif symbol == 'PERPUSDT':
        n1 = 29
        n2 = 26
    elif symbol == 'QNTUSDT':
        n1 = 15
        n2 = 12
    elif symbol == 'RIFUSDT':
        n1 = 15
        n2 = 33
    elif symbol == 'ROSEUSDT':
        n1 = 18
        n2 = 38
    elif symbol == 'RUNEUSDT':
        n1 = 16
        n2 = 27
    elif symbol == 'SANDUSDT':
        n1 = 36
        n2 = 26
    elif symbol == 'SHIBUSDT':
        n1 = 21
        n2 = 38
    elif symbol == 'SNXUSDT':
        n1 = 16
        n2 = 23
    elif symbol == 'SOLUSDT':
        n1 = 12
        n2 = 12
    elif symbol == 'STXUSDT':
        n1 = 10
        n2 = 17
    elif symbol == 'TFUELUSDT':
        n1 = 12
        n2 = 11
    elif symbol == 'THETAUSDT':
        n1 = 17
        n2 = 13
    elif symbol == 'TRXUSDT':
        n1 = 15
        n2 = 39
    elif symbol == 'UNIUSDT':
        n1 = 25
        n2 = 28
    elif symbol == 'WAVESUSDT':
        n1 = 13
        n2 = 10
    elif symbol == 'XECUSDT':
        n1 = 19
        n2 = 37
    elif symbol == 'XEMUSDT':
        n1 = 12
        n2 = 23
    elif symbol == 'XLMUSDT':
        n1 = 21
        n2 = 18
    elif symbol == 'XMRUSDT':
        n1 = 23
        n2 = 13
    elif symbol == 'XTZUSDT':
        n1 = 17
        n2 = 26
    elif symbol == 'XRPUSDT':
        n1 = 13
        n2 = 17
    elif symbol == 'ZECUSDT':
        n1 = 16
        n2 = 28
    elif symbol == 'ZILUSDT':
        n1 = 16
        n2 = 38
    else:
        n1 = 10
        n2 = 21

    ap = pta.hlc3(high_series, low_series, close_series)
    esa = pta.ema(ap, n1)
    d = pta.ema(abs(ap - esa), n1)
    ci = (ap - esa) / (0.015 * d)
    wt1 = pta.ema(ci, n2)
    wt2 = pta.sma(wt1, 4)
    cmo = pta.cmo(close_series, talib=False)
    macdh = pta.macd(close_series)['MACDh_12_26_9']
    # print(f'{symbol} : {wt1.iat[-1]}')

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = np.poly1d(np.polyfit(y, x + np.std(x), 1))(y)
    best_fit_line3 = np.poly1d(np.polyfit(y, x - np.std(x), 1))(y)

    if CMO_1h and not WAVETREND_1h and not MACD_1h:  # cmo=true,wavetrend=false,macdh=false
        if (cmo.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]) | \
                (cmo.iat[-1] < -60 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print('found')
                print("on 1h timeframe " + symbol)
                print(f'cmo: {cmo.iat[-2]}')

            # plt.figure(figsize=(8, 6))
            # plt.grid(True)
            # plt.plot(x)
            # plt.title(label=f'{symbol}', color="green")
            # plt.plot(best_fit_line1, '--', color='r')
            # plt.plot(best_fit_line2, '--', color='r')
            # plt.plot(best_fit_line3, '--', color='green')
            # plt.show(block=False)
            # plt.pause(6)
            # plt.close()

    elif CMO_1h and WAVETREND_1h and not MACD_1h:  # cmo=true,wavetrend=true,macd=false
        if (cmo.iat[-1] < -60 and wt1.iat[-1] < -75 and x[-1] < best_fit_line3[-1] and
            best_fit_line1[0] <= best_fit_line1[-1]) | (cmo.iat[-1] < -60 and
                                                        wt1.iat[-2] < -75 and
                                                        x[-1] < best_fit_line3[-1]
                                                        and best_fit_line1[0] >= best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print('found')
                print("on 1h timeframe " + symbol)
                print(f'cmo: {cmo.iat[-2]}')
                print(f'wt1: {wt1.iat[-2]}')

    elif CMO_1h and WAVETREND_1h and MACD_1h:  # cmo=true,wavetrend=true,macdh=true
        if (cmo.iat[-1] < -60 and wt1.iat[-1] < -75 and macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and
            best_fit_line1[0] <= best_fit_line1[-1]) | (
                cmo.iat[-1] < -60 and wt1.iat[-1] < -75 and macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and
                best_fit_line1[0] <= best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print('found')
                print("on 1h timeframe " + symbol)
                print(f'cmo: {cmo.iat[-2]}')
                print(f'wt1: {wt1.iat[-2]}')
                print(f'macdh: {macdh.iat[-2]}')

    elif WAVETREND_1h and not CMO_1h and not MACD_1h:  # cmo=false,wavetrend=true,macdh=false
        if (wt1.iat[-1] < -75 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]) | \
                (wt1.iat[-1] < -75 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print('found')
                print("on 1h timeframe " + symbol)
                print(f'wt1: {wt1.iat[-2]}')

    elif CMO_1h and MACD_1h and not WAVETREND_1h:  # cmo=true,wavetrend=false,macdh=true
        if (cmo.iat[-1] < -60 and macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and
            best_fit_line1[0] <= best_fit_line1[-1]) | (cmo.iat[-1] < -60 and
                                                        macdh.iat[-1] > 0 and
                                                        x[-1] < best_fit_line3[-1] and
                                                        best_fit_line1[0] > best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print('found')
                print("on 1h timeframe " + symbol)
                print(f'cmo: {cmo.iat[-1]}')
                print(f'macdh: {macdh.iat[-1]}')

    elif MACD_1h and not CMO_1h and not WAVETREND_1h:  # cmo=false,wavetrend=false,macdh=true
        if (macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]) | \
                (macdh.iat[-1] > 0 and x[-1] < best_fit_line3[-1] and best_fit_line1[0] > best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print('found')
                print("on 1h timeframe " + symbol)
                print(f'macdh: {macdh.iat[-1]}')

    else:
        if (x[-1] < best_fit_line3[-1] and best_fit_line1[0] <= best_fit_line1[-1]) | \
                (x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]):
            filtered_pairs1.append(symbol)
            print('found')
            print(f'on {interval} timeframe {symbol}')

    return filtered_pairs1


def filter2(filtered_pairs1):
    interval = '15m'
    symbol = filtered_pairs1
    klines = client.get_klines(symbol=symbol, interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    close_array = np.asarray(close)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

    if (x[-1] < best_fit_line3[-1] and best_fit_line1[0] < best_fit_line1[-1]) | \
            (x[-1] < best_fit_line3[-1] and best_fit_line1[0] > best_fit_line1[-1]):
        filtered_pairs2.append(symbol)
        if DEBUG:
            print("on 15min timeframe " + symbol)

    return filtered_pairs2


def filter3(filtered_pairs2):
    interval = '5m'
    symbol = filtered_pairs2
    klines = client.get_klines(symbol=symbol, interval=interval)
    open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

    if x[-1] < best_fit_line3[-1] and best_fit_line1[0] >= best_fit_line1[-1]:
        filtered_pairs3.append(symbol)
        if DEBUG:
            print("on 5min timeframe " + symbol)

    elif x[-1] < best_fit_line3[-1] and best_fit_line1[0] < best_fit_line1[-1]:
        filtered_pairs3.append(symbol)
        if DEBUG:
            print("on 5min timeframe " + symbol)

    return filtered_pairs3


def momentum(filtered_pairs3):
    interval = '1m'
    symbol = filtered_pairs3

    start_str = '12 hours ago UTC'
    end_str = f'{datetime.now()}'
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


def analyze(trading_pairs):
    signal_coins = {}
    filtered_pairs1.clear()
    filtered_pairs2.clear()
    filtered_pairs3.clear()
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

    for i in filtered_pairs2:  # 5m
        output = filter3(i)
        if DEBUG:
            print(output)

    for i in filtered_pairs3:  # 1m
        output = momentum(i)
        print(output)

    for pair in selected_pair:
        signal_coins[pair] = pair
        with open(SIGNAL_FILE_BUY, 'a+') as f:
            f.writelines(pair + '\n')
            # timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
        # with open(SIGNAL_NAME + '.log', 'a+') as f:
        #     f.write(timestamp + ' ' + pair + '\n')
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
            print(f'CMO_1h: {CMO_1h} | WAVETREND_1h: {WAVETREND_1h} | MACD_1h: {MACD_1h}')
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

