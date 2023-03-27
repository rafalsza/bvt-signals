from binance.client import Client
import numpy as np
import threading
import os
import pandas_ta as pta
import pandas as pd
import warnings
import time
from loguru import logger

warnings.filterwarnings("ignore")
client = Client("", "")
TIME_TO_WAIT = 1  # Minutes to wait between analysis
DEBUG = False
TICKERS = "tickerlists/tickers_binance_USDT.txt"
SIGNAL_NAME = "rs_signals_wavetrend"
SIGNAL_FILE_BUY = "signals/" + SIGNAL_NAME + ".buy"
SIGNAL_FILE_SELL = "signals/" + SIGNAL_NAME + ".sell"

CMO_1h = True
WAVETREND_1h = True
MACD_1h = False


# for colourful logging to the console
class TxColors:
    BUY = "\033[92m"
    WARNING = "\033[93m"
    SELL_LOSS = "\033[91m"
    SELL_PROFIT = "\033[32m"
    DIM = "\033[2m\033[35m"
    DEFAULT = "\033[39m"
    RED = "\033[91m"


filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
filtered_pairs_sell = []
selected_pair_buy = []
selected_pair_sell = []


def importdata(symbol, interval, limit):
    df = pd.DataFrame(client.get_historical_klines(symbol, interval, limit=limit)).astype(float)
    df = df.iloc[:, :6]
    df.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms")
    return df


def wavetrend(df, n1, n2):
    ap = pta.hlc3(df.High, df.Low, df.Close)
    esa = pta.ema(ap, n1)
    d = pta.ema(abs(ap - esa), n1)
    ci = (ap - esa) / (0.015 * d)
    wt1 = pta.ema(ci, n2)
    wt2 = pta.sma(wt1, 4)
    return wt1, wt2


@logger.catch
def filter1(pair):
    interval = "1h"
    symbol = pair
    df = importdata(symbol, interval, limit=500)

    if symbol == "BTCUSDT":
        n1 = 13
        n2 = 16
    elif symbol == "1INCHUSDT":
        n1 = 26
        n2 = 30
    elif symbol == "AAVEUSDT":
        n1 = 10
        n2 = 12
    elif symbol == "ACAUSDT":
        n1 = 10
        n2 = 32
    elif symbol == "ADAUSDT":
        n1 = 15
        n2 = 31
    elif symbol == "ALGOUSDT":
        n1 = 10
        n2 = 26
    elif symbol == "ALPHAUSDT":
        n1 = 10
        n2 = 10
    elif symbol == "AMPUSDT":
        n1 = 10
        n2 = 38
    elif symbol == "ANKRUSDT":
        n1 = 16
        n2 = 22
    elif symbol == "ANTUSDT":
        n1 = 10
        n2 = 16
    elif symbol == "APEUSDT":
        n1 = 23
        n2 = 30
    elif symbol == "API3USDT":
        n1 = 23
        n2 = 10
    elif symbol == "APTUSDT":
        n1 = 12
        n2 = 29
    elif symbol == "ARDRUSDT":
        n1 = 11
        n2 = 23
    elif symbol == "ARUSDT":
        n1 = 13
        n2 = 13
    elif symbol == "ASTRUSDT":
        n1 = 12
        n2 = 10
    elif symbol == "ATOMUSDT":
        n1 = 39
        n2 = 24
    elif symbol == "AUDIOUSDT":
        n1 = 11
        n2 = 16
    elif symbol == "AVAXUSDT":
        n1 = 12
        n2 = 22
    elif symbol == "AXSUSDT":
        n1 = 12
        n2 = 34
    elif symbol == "BALUSDT":
        n1 = 14
        n2 = 19
    elif symbol == "BANDUSDT":
        n1 = 11
        n2 = 38
    elif symbol == "BATUSDT":
        n1 = 10
        n2 = 13
    elif symbol == "BCHUSDT":
        n1 = 10
        n2 = 33
    elif symbol == "BICOUSDT":
        n1 = 10
        n2 = 26
    elif symbol == "BNBUSDT":
        n1 = 11
        n2 = 23
    elif symbol == "BNXUSDT":
        n1 = 10
        n2 = 34
    elif symbol == "BSWUSDT":
        n1 = 11
        n2 = 22
    elif symbol == "CAKEUSDT":
        n1 = 14
        n2 = 32
    elif symbol == "CELOUSDT":
        n1 = 20
        n2 = 38
    elif symbol == "CELRUSDT":
        n1 = 24
        n2 = 18
    elif symbol == "CHZUSDT":
        n1 = 13
        n2 = 16
    elif symbol == "CHRUSDT":
        n1 = 10
        n2 = 39
    elif symbol == "CKBUSDT":
        n1 = 10
        n2 = 10
    elif symbol == "COMPUSDT":
        n1 = 18
        n2 = 14
    elif symbol == "COTIUSDT":
        n1 = 12
        n2 = 17
    elif symbol == "CRVUSDT":
        n1 = 15
        n2 = 27
    elif symbol == "CTKUSDT":
        n1 = 24
        n2 = 10
    elif symbol == "CTSIUSDT":
        n1 = 14
        n2 = 11
    elif symbol == "CVXUSDT":
        n1 = 10
        n2 = 39
    elif symbol == "DASHUSDT":
        n1 = 11
        n2 = 30
    elif symbol == "DENTUSDT":
        n1 = 19
        n2 = 37
    elif symbol == "DOGEUSDT":
        n1 = 10
        n2 = 36
    elif symbol == "DOTUSDT":
        n1 = 10
        n2 = 20
    elif symbol == "DYDXUSDT":
        n1 = 32
        n2 = 39
    elif symbol == "DCRUSDT":
        n1 = 17
        n2 = 38
    elif symbol == "DEXEUSDT":
        n1 = 32
        n2 = 13
    elif symbol == "DGBUSDT":
        n1 = 20
        n2 = 23
    elif symbol == "EGLDUSDT":
        n1 = 10
        n2 = 15
    elif symbol == "ELFUSDT":
        n1 = 24
        n2 = 16
    elif symbol == "ENJUSDT":
        n1 = 10
        n2 = 12
    elif symbol == "ENSUSDT":
        n1 = 10
        n2 = 29
    elif symbol == "EOSUSDT":
        n1 = 10
        n2 = 19
    elif symbol == "ETCUSDT":
        n1 = 10
        n2 = 10
    elif symbol == "ETHUSDT":
        n1 = 10
        n2 = 19
    elif symbol == "FETUSDT":
        n1 = 11
        n2 = 15
    elif symbol == "FILUSDT":
        n1 = 11
        n2 = 33
    elif symbol == "FLOWUSDT":
        n1 = 10
        n2 = 28
    elif symbol == "FLUXUSDT":
        n1 = 12
        n2 = 35
    elif symbol == "FTMUSDT":
        n1 = 17
        n2 = 39
    elif symbol == "FUNUSDT":
        n1 = 11
        n2 = 27
    elif symbol == "FXSUSDT":
        n1 = 11
        n2 = 37
    elif symbol == "GNOUSDT":
        n1 = 11
        n2 = 26
    elif symbol == "GRTUSDT":
        n1 = 15
        n2 = 23
    elif symbol == "GTCUSDT":
        n1 = 21
        n2 = 25
    elif symbol == "HBARUSDT":
        n1 = 16
        n2 = 13
    elif symbol == "HIVEUSDT":
        n1 = 10
        n2 = 16
    elif symbol == "HOTUSDT":
        n1 = 11
        n2 = 21
    elif symbol == "ICPUSDT":
        n1 = 15
        n2 = 23
    elif symbol == "ICXUSDT":
        n1 = 10
        n2 = 27
    elif symbol == "IMXUSDT":
        n1 = 18
        n2 = 32
    elif symbol == "INJUSDT":
        n1 = 14
        n2 = 11
    elif symbol == "IOSTUSDT":
        n1 = 12
        n2 = 11
    elif symbol == "IOTAUSDT":
        n1 = 20
        n2 = 25
    elif symbol == "IOTXUSDT":
        n1 = 16
        n2 = 19
    elif symbol == "JSTUSDT":
        n1 = 14
        n2 = 16
    elif symbol == "KAVAUSDT":
        n1 = 21
        n2 = 21
    elif symbol == "KDAUSDT":
        n1 = 10
        n2 = 37
    elif symbol == "KLAYUSDT":
        n1 = 12
        n2 = 35
    elif symbol == "KNCUSDT":
        n1 = 34
        n2 = 23
    elif symbol == "KSMUSDT":
        n1 = 18
        n2 = 39
    elif symbol == "LINKUSDT":
        n1 = 13
        n2 = 16
    elif symbol == "LPTUSDT":
        n1 = 11
        n2 = 15
    elif symbol == "LSKUSDT":
        n1 = 11
        n2 = 12
    elif symbol == "LTCUSDT":
        n1 = 13
        n2 = 18
    elif symbol == "LRCUSDT":
        n1 = 10
        n2 = 22
    elif symbol == "MANAUSDT":
        n1 = 11
        n2 = 13
    elif symbol == "MASKUSDT":
        n1 = 19
        n2 = 25
    elif symbol == "MATICUSDT":
        n1 = 21
        n2 = 38
    elif symbol == "MBOXUSDT":
        n1 = 10
        n2 = 30
    elif symbol == "MDXUSDT":
        n1 = 11
        n2 = 38
    elif symbol == "MINAUSDT":
        n1 = 18
        n2 = 12
    elif symbol == "MKRUSDT":
        n1 = 11
        n2 = 17
    elif symbol == "NEARUSDT":
        n1 = 10
        n2 = 31
    elif symbol == "NEOUSDT":
        n1 = 11
        n2 = 16
    elif symbol == "NMRUSDT":
        n1 = 29
        n2 = 10
    elif symbol == "OCEANUSDT":
        n1 = 24
        n2 = 28
    elif symbol == "OMGUSDT":
        n1 = 16
        n2 = 30
    elif symbol == "ONEUSDT":
        n1 = 19
        n2 = 39
    elif symbol == "ONTUSDT":
        n1 = 18
        n2 = 16
    elif symbol == "PEOPLEUSDT":
        n1 = 13
        n2 = 30
    elif symbol == "PONDUSDT":
        n1 = 10
        n2 = 10
    elif symbol == "PUNDIXUSDT":
        n1 = 10
        n2 = 16
    elif symbol == "PYRUSDT":
        n1 = 20
        n2 = 35
    elif symbol == "QNTUSDT":
        n1 = 14
        n2 = 11
    elif symbol == "QTUMUSDT":
        n1 = 10
        n2 = 23
    elif symbol == "RADUSDT":
        n1 = 16
        n2 = 39
    elif symbol == "RENUSDT":
        n1 = 13
        n2 = 34
    elif symbol == "REQUSDT":
        n1 = 11
        n2 = 24
    elif symbol == "RLCUSDT":
        n1 = 16
        n2 = 13
    elif symbol == "RNDRUSDT":
        n1 = 18
        n2 = 36
    elif symbol == "ROSEUSDT":
        n1 = 15
        n2 = 13
    elif symbol == "RSRUSDT":
        n1 = 11
        n2 = 34
    elif symbol == "RUNEUSDT":
        n1 = 12
        n2 = 24
    elif symbol == "RVNUSDT":
        n1 = 10
        n2 = 27
    elif symbol == "SANDUSDT":
        n1 = 10
        n2 = 30
    elif symbol == "SFPUSDT":
        n1 = 18
        n2 = 28
    elif symbol == "SHIBUSDT":
        n1 = 21
        n2 = 38
    elif symbol == "SKLUSDT":
        n1 = 16
        n2 = 23
    elif symbol == "SNXUSDT":
        n1 = 17
        n2 = 38
    elif symbol == "SOLUSDT":
        n1 = 12
        n2 = 37
    elif symbol == "STXUSDT":
        n1 = 10
        n2 = 34
    elif symbol == "SXPUSDT":
        n1 = 29
        n2 = 30
    elif symbol == "SYSUSDT":
        n1 = 19
        n2 = 39
    elif symbol == "TFUELUSDT":
        n1 = 10
        n2 = 39
    elif symbol == "THETAUSDT":
        n1 = 20
        n2 = 26
    elif symbol == "TRXUSDT":
        n1 = 10
        n2 = 17
    elif symbol == "TWTUSDT":
        n1 = 23
        n2 = 16
    elif symbol == "UMAUSDT":
        n1 = 24
        n2 = 10
    elif symbol == "UNIUSDT":
        n1 = 20
        n2 = 27
    elif symbol == "VETUSDT":
        n1 = 13
        n2 = 15
    elif symbol == "VGXUSDT":
        n1 = 13
        n2 = 32
    elif symbol == "WAXPUSDT":
        n1 = 10
        n2 = 39
    elif symbol == "WINUSDT":
        n1 = 13
        n2 = 39
    elif symbol == "WOOUSDT":
        n1 = 19
        n2 = 24
    elif symbol == "WRXUSDT":
        n1 = 12
        n2 = 15
    elif symbol == "XECUSDT":
        n1 = 14
        n2 = 37
    elif symbol == "XEMUSDT":
        n1 = 28
        n2 = 30
    elif symbol == "XLMUSDT":
        n1 = 18
        n2 = 18
    elif symbol == "XMRUSDT":
        n1 = 19
        n2 = 25
    elif symbol == "XNOUSDT":
        n1 = 16
        n2 = 38
    elif symbol == "XTZUSDT":
        n1 = 17
        n2 = 26
    elif symbol == "XRPUSDT":
        n1 = 12
        n2 = 16
    elif symbol == "YFIUSDT":
        n1 = 10
        n2 = 16
    elif symbol == "ZECUSDT":
        n1 = 19
        n2 = 28
    elif symbol == "ZENUSDT":
        n1 = 16
        n2 = 38
    elif symbol == "ZILUSDT":
        n1 = 17
        n2 = 33
    elif symbol == "ZRXUSDT":
        n1 = 12
        n2 = 29
    else:
        n1 = 10
        n2 = 21

    wt1, wt2 = wavetrend(df, n1, n2)
    ema_200 = pta.ema(df.Close, 200)

    if wt1[-1] < -75 and df.Close[-1] < ema_200[-1]:
        filtered_pairs1.append(symbol)
        if DEBUG:
            print("found")
            print("on 1h timeframe " + symbol)
            print(f"cmo: {wt1.iat[-1]}")

    if wt1.iat[-1] > 60:
        selected_pair_sell.append(symbol)
        if DEBUG:
            print("found sell signal")
            print("on 1h timeframe " + symbol)
            print(f"wt1: {wt1.iat[-1]}")

    return filtered_pairs1, selected_pair_sell


def filter2(filtered_pairs1):
    interval = "15m"
    symbol = filtered_pairs1
    klines = client.get_klines(symbol=symbol, interval=interval)
    # open_time = [int(entry[0]) for entry in klines]
    close = [float(entry[4]) for entry in klines]
    # close_array = np.asarray(close)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 1.01
    best_fit_line3 = (np.poly1d(np.polyfit(y, x, 1))(y)) * 0.99

    if (x[-1] < best_fit_line3[-1] and best_fit_line1[0] < best_fit_line1[-1]) | (
        x[-1] < best_fit_line3[-1] and best_fit_line1[0] > best_fit_line1[-1]
    ):
        filtered_pairs2.append(symbol)
        if DEBUG:
            print("on 15min timeframe " + symbol)

    return filtered_pairs2


def filter3(filtered_pairs2):
    interval = "5m"
    symbol = filtered_pairs2
    klines = client.get_klines(symbol=symbol, interval=interval)
    # open_time = [int(entry[0]) for entry in klines]
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
    interval = "1m"
    symbol = filtered_pairs3
    df = importdata(symbol, interval, limit=1000)
    # CMO
    real = pta.cmo(df.Close, talib=False)
    # WaveTrend
    n1 = 10
    n2 = 21
    wt1, wt2 = wavetrend(df, n1, n2)
    #
    print("on 1m timeframe " + symbol)
    print(f"cmo: {real.iat[-1]}")
    print(f"wt1: {wt1.iat[-1]}")

    if real.iat[-1] < -50 and wt1.iat[-1] < -60:
        print("oversold dip found")
        selected_pair_buy.append(symbol)

    return selected_pair_buy


def analyze(trading_pairs):
    signal_coins = {}
    signal_coins_sell = {}
    filtered_pairs_sell.clear()
    filtered_pairs1.clear()
    filtered_pairs2.clear()
    filtered_pairs3.clear()
    selected_pair_buy.clear()
    selected_pair_sell.clear()

    if os.path.exists(SIGNAL_FILE_BUY):
        os.remove(SIGNAL_FILE_BUY)

    if os.path.exists(SIGNAL_FILE_SELL):
        os.remove(SIGNAL_FILE_SELL)

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

    for pair in selected_pair_buy:
        signal_coins[pair] = pair
        with open(SIGNAL_FILE_BUY, "a+") as f:
            f.writelines(pair + "\n")
            # timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
        # with open(SIGNAL_NAME + '.log', 'a+') as f:
        #     f.write(timestamp + ' ' + pair + '\n')

    for pair in selected_pair_sell:
        signal_coins_sell[pair] = pair
        with open(SIGNAL_FILE_SELL, "a+") as f:
            f.writelines(pair + "\n")

    if selected_pair_buy:
        print(f"{TxColors.BUY}{SIGNAL_NAME}: {selected_pair_buy} - Buy Signal Detected{TxColors.DEFAULT}")
    if selected_pair_sell:
        print(f"{TxColors.RED}{SIGNAL_NAME}: {selected_pair_sell} - Sell Signal Detected{TxColors.RED}")
    else:
        print(f"{TxColors.DEFAULT}{SIGNAL_NAME}: - not enough signal to buy")
    return signal_coins, signal_coins_sell


def do_work():
    while True:
        try:
            if not os.path.exists(TICKERS):
                time.sleep((TIME_TO_WAIT * 60))
                continue

            signal_coins = {}
            signal_coins_sell = {}
            pairs = {}

            with open(TICKERS) as f:
                pairs = f.read().splitlines()

            if not threading.main_thread().is_alive():
                exit()
            print(f"{SIGNAL_NAME}: Analyzing {len(pairs)} coins")
            print(f"CMO_1h: {CMO_1h} | WAVETREND_1h: {WAVETREND_1h} | MACD_1h: {MACD_1h}")
            signal_coins, signal_coins_sell = analyze(pairs)
            print(
                f"{SIGNAL_NAME}: {len(signal_coins)} "
                f"coins with Buy Signals. Waiting {TIME_TO_WAIT} minutes for next analysis."
            )
            print(
                f"{SIGNAL_NAME}: {len(signal_coins_sell)} "
                f"coins with Sell Signals. Waiting {TIME_TO_WAIT} minutes for next analysis."
            )

            time.sleep((TIME_TO_WAIT * 60))
        except Exception as e:
            print(f"{SIGNAL_NAME}: Exception do_work() 1: {e}")
            continue
        except KeyboardInterrupt as ki:
            exit()
