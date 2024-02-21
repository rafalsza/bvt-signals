import os
import threading

# from datetime import datetime
import time

import numpy as np
import pandas as pd
import pandas_ta as pta
from finta import TA
from binance.client import Client
from loguru import logger

# from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from w_params import wavetrend_parameters

client = Client("", "")
TIME_TO_WAIT = 1  # Minutes to wait between analysis
DEBUG = False
TICKERS = "tickerlists/tickers_all_USDT.txt"
SIGNAL_NAME = "rs_signals_buy_dip"
SIGNAL_FILE_BUY = "signals/" + SIGNAL_NAME + ".buy"

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


filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
selected_pair = []


def importdata(symbol, interval, limit):
    df = pd.DataFrame(
        client.get_historical_klines(symbol, interval, limit=limit)
    ).astype(float)
    df = df.iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "Low", "close", "Volume"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms")
    return df


def regression_channel(data):
    # Create the linear regression channel
    y = data["close"].values
    X = range(len(y))
    X = np.array(X).reshape(-1, 1)  # Reshape X to be a 2D array
    model = LinearRegression()
    model.fit(X, y)
    linear_regression = model.predict(X)
    # Calculate the standard deviation of the residuals
    residuals = y - model.predict(X)
    std = np.std(residuals)
    linear_upper = linear_regression + 2 * std
    linear_lower = linear_regression - 2 * std

    return (
        linear_regression,
        linear_lower,
        linear_upper,
    )


@logger.catch
def filter1(pair):
    interval = "1h"
    symbol = pair
    df = importdata(symbol, interval, limit=500)
    linear_regression, linear_lower, linear_upper = regression_channel(df)
    ema_200 = pta.ema(df.close, 200)

    n1, n2 = wavetrend_parameters.get(symbol, (10, 21))

    wt1 = TA.WTO(df, n1, n2)["WT1."]
    cmo = pta.cmo(df.close, talib=False)
    macdh = pta.macd(df.close)["MACDh_12_26_9"]

    if CMO_1h and not WAVETREND_1h and not MACD_1h:
        if (
            cmo.iloc[-1] < -60
            and df.close[-1] < ema_200.iloc[-1]
            and df.close[-1] < linear_lower[-1]
            and linear_regression[0] <= linear_regression[-1]
        ) | (
            cmo.iloc[-1] < -60
            and df.close[-1] < linear_lower.iloc[-1]
            and linear_regression[0] >= linear_regression[-1]
        ):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print("found")
                print("on 1h timeframe " + symbol)
                print(f"cmo: {cmo.iat[-2]}")

    elif CMO_1h and WAVETREND_1h and not MACD_1h:  # cmo=true,wavetrend=true,macd=false
        if (
            cmo.iloc[-1] < -60
            and wt1.iloc[-1] < -75
            and df.close.iloc[-1] < ema_200.iloc[-1]
            and df.close.iloc[-1] <= linear_lower[-1]
        ):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print("found")
                print("on 1h timeframe " + symbol)
                print(f"cmo: {cmo.iloc[-1]}")
                print(f"wt1: {wt1.iloc[-1]}")

            # plt.figure(figsize=(8, 6))
            # plt.grid(True)
            # plt.plot(list(df.close))
            # plt.title(label=f'{symbol}', color="green")
            # plt.plot(linear_regression, '--', color='r')
            # plt.plot(linear_upper, '--', color='r')
            # plt.plot(linear_lower, '--', color='green')
            # plt.show(block=False)
            # plt.pause(15)
            # plt.close()

    elif CMO_1h and WAVETREND_1h and MACD_1h:  # cmo=true,wavetrend=true,macdh=true
        if (
            cmo.iloc[-1] < -60
            and wt1.iloc[-1] < -75
            and macdh.iloc[-1] > 0
            and df.close.iloc[-1] < linear_lower[-1]
            and linear_regression[0] <= linear_regression[-1]
        ) | (
            cmo.iat[-1] < -60
            and wt1.iloc[-1] < -75
            and macdh.iat[-1] > 0
            and df.close.iloc[-1] < linear_lower[-1]
            and linear_regression[0] <= linear_regression[-1]
        ):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print("found")
                print("on 1h timeframe " + symbol)
                print(f"cmo: {cmo.iat[-2]}")
                print(f"wt1: {wt1.iloc[-2]}")
                print(f"macdh: {macdh.iat[-2]}")

    elif WAVETREND_1h and not CMO_1h and not MACD_1h:
        if (
            wt1.iat[-1] < -75
            and df.close[-1] < linear_lower[-1]
            and linear_regression[0] <= linear_regression[-1]
        ) | (
            wt1.iat[-1] < -75
            and df.close[-1] < linear_lower[-1]
            and linear_regression[0] >= linear_regression[-1]
        ):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print("found")
                print("on 1h timeframe " + symbol)
                print(f"wt1: {wt1.iat[-2]}")

    elif CMO_1h and MACD_1h and not WAVETREND_1h:  # cmo=true,wavetrend=false,macdh=true
        if (
            cmo.iat[-1] < -60
            and macdh.iat[-1] > 0
            and df.close[-1] < linear_lower[-1]
            and linear_regression[0] <= linear_regression[-1]
        ) | (
            cmo.iat[-1] < -60
            and macdh.iat[-1] > 0
            and df.close[-1] < linear_lower[-1]
            and linear_regression[0] > linear_regression[-1]
        ):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print("found")
                print("on 1h timeframe " + symbol)
                print(f"cmo: {cmo.iat[-1]}")
                print(f"macdh: {macdh.iat[-1]}")

    elif (
        MACD_1h and not CMO_1h and not WAVETREND_1h
    ):  # cmo=false,wavetrend=false,macdh=true
        if (
            macdh.iloc[-1] > 0
            and df.close[-1] < linear_lower[-1]
            and linear_regression[0] <= linear_regression[-1]
        ) | (
            macdh.iloc[-1] > 0
            and df.close[-1] < linear_lower[-1]
            and linear_regression[0] > linear_regression[-1]
        ):
            filtered_pairs1.append(symbol)
            if DEBUG:
                print("found")
                print("on 1h timeframe " + symbol)
                print(f"macdh: {macdh.iloc[-1]}")

    return filtered_pairs1


def filter2(filtered_pairs1):
    interval = "15m"
    symbol = filtered_pairs1
    df = importdata(symbol, interval, limit=500)
    linear_regression, linear_lower, linear_upper = regression_channel(df)

    if df.close.iloc[-1] < linear_lower[-1]:
        filtered_pairs2.append(symbol)
        if DEBUG:
            print("on 15min timeframe " + symbol)
    return filtered_pairs2


def filter3(filtered_pairs2):
    interval = "5m"
    symbol = filtered_pairs2
    klines = client.get_klines(symbol=symbol, interval=interval)
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
    real = pta.cmo(df.close, talib=False)
    # WaveTrend
    wt1 = TA.WTO(df)["WT1."]
    #
    print("on 1m timeframe " + symbol)
    print(f"cmo: {real.iloc[-1]}")
    print(f"wt1: {wt1.iloc[-1]}")

    if real.iloc[-1] < -50 and wt1.iloc[-1] < -60:
        print("oversold dip found")
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
        with open(SIGNAL_FILE_BUY, "a+") as f:
            f.writelines(pair + "\n")
            # timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
        # with open(SIGNAL_NAME + '.log', 'a+') as f:
        #     f.write(timestamp + ' ' + pair + '\n')
    if selected_pair:
        print(
            f"{TxColors.BUY}{SIGNAL_NAME}: {selected_pair} - Buy Signal Detected{TxColors.DEFAULT}"
        )
    else:
        print(f"{TxColors.DEFAULT}{SIGNAL_NAME}: - not enough signal to buy")
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
            print(f"{SIGNAL_NAME}: Analyzing {len(pairs)} coins")
            print(
                f"CMO_1h: {CMO_1h} | WAVETREND_1h: {WAVETREND_1h} | MACD_1h: {MACD_1h}"
            )
            signal_coins = analyze(pairs)
            print(
                f"{SIGNAL_NAME}: {len(signal_coins)} "
                f"coins with Buy Signals. Waiting {TIME_TO_WAIT} minutes for next analysis."
            )

            time.sleep((TIME_TO_WAIT * 60))
        except Exception as e:
            print(f"{SIGNAL_NAME}: Exception do_work() 1: {e}")
            continue
        except KeyboardInterrupt as ki:
            print(ki)
            continue
