"""
Linear/Polynomial Regression Channel
"""
import os
import threading
import time
import warnings

import numpy as np
import pandas as pd
import pandas_ta as pta
from binance.client import Client
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
client = Client("", "")
TIME_TO_WAIT = 1  # Minutes to wait between analysis
DEBUG = False
TICKERS = "tickerlists/tickers_binance_USDT.txt"
SIGNAL_NAME = "rs_signals_polynomial"
SIGNAL_FILE_BUY = "signals/" + SIGNAL_NAME + ".buy"
SIGNAL_FILE_SELL = "signals/" + SIGNAL_NAME + ".sell"


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
filtered_pairs_5m = []
filtered_pairs_1h_buy = []
selected_pair_buy = []
selected_pair_sell = []


def importdata(symbol, interval, limit):
    client = Client()
    df = pd.DataFrame(
        client.get_historical_klines(symbol, interval, limit=limit)
    ).astype(float)
    df = df.iloc[:, :6]
    df.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms")

    return df


def regression_channel(data):
    # Create the linear regression channel
    y = data["Close"].values
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

    # Create the polynomial regression channel
    degree = 9
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    data["Polynomial Regression"] = model.predict(X)

    # Calculate the standard deviation of the residuals
    residuals = y - model.predict(X)
    std = np.std(residuals)

    # Calculate the upper and lower channels
    polynomial_upper = data["Polynomial Regression"] + 2 * std
    polynomial_lower = data["Polynomial Regression"] - 2 * std

    return (
        polynomial_upper,
        polynomial_lower,
        linear_regression,
        linear_lower,
        linear_upper,
    )


@logger.catch
def filter_1h(pair):
    interval = "1h"
    symbol = pair
    df = importdata(symbol, interval, limit=500)
    ema_200 = pta.ema(df.Close, length=200)

    (
        polynomial_upper,
        polynomial_lower,
        linear_regression,
        linear_lower,
        linear_upper,
    ) = regression_channel(df)

    if (
        polynomial_lower[-1] > df.Close[-1] >= ema_200[-1]
        and polynomial_lower[0] < polynomial_lower[-1]
    ):
        filtered_pairs_1h_buy.append(symbol)
        if DEBUG:
            print("found buy signal")
            print("on 1h timeframe " + symbol)
    elif df.Close[-1] > polynomial_upper[-1]:
        selected_pair_sell.append(symbol)
        if DEBUG:
            print("found sell signal")
            print("on 1h timeframe " + symbol)

    return filtered_pairs_1h_buy, selected_pair_sell


def filter_5m(coins):
    interval = "5m"
    symbol = coins
    df = importdata(symbol, interval, limit=500)
    (
        polynomial_upper,
        polynomial_lower,
        linear_regression,
        linear_lower,
        linear_upper,
    ) = regression_channel(df)

    if (
        df.Close[-1] < linear_lower[-1]
        and linear_regression[0] >= linear_regression[-1]
    ):
        filtered_pairs_5m.append(symbol)
        if DEBUG:
            print("on 5m timeframe " + symbol)

    elif (
        df.Close[-1] < linear_lower[-1] and linear_regression[0] < linear_regression[-1]
    ):
        filtered_pairs_5m.append(symbol)
        if DEBUG:
            print("on 5m timeframe " + symbol)

    return filtered_pairs_5m


def momentum(coins):
    interval = "1m"
    symbol = coins
    df = importdata(symbol, interval, limit=1000)
    # CMO
    cmo = pta.cmo(df.Close, talib=False)
    # WaveTrend
    n1 = 10
    n2 = 21
    ap = pta.hlc3(df.High, df.Low, df.Close)
    esa = pta.ema(ap, n1)
    d = pta.ema(abs(ap - esa), n1)
    ci = (ap - esa) / (0.015 * d)
    wt1 = pta.ema(ci, n2)
    #
    print("on 1m timeframe " + symbol)
    print(f"cmo: {cmo.iat[-1]}")
    print(f"wt1: {wt1.iat[-1]}")

    if cmo.iat[-1] < -50 and wt1.iat[-1] < -60:
        print("oversold dip found")
        selected_pair_buy.append(symbol)

    return selected_pair_buy


def analyze(trading_pairs):
    """
    Analyze trading pairs and generate buy/sell signals.

    Args:
        trading_pairs (list): List of trading pair symbols.

    Returns:
        tuple: A tuple containing the trading pairs with buy signals and sell signals.
    """
    signal_coins = {}
    signal_coins_sell = {}
    filtered_pairs_1h_buy.clear()
    filtered_pairs_5m.clear()
    selected_pair_buy.clear()
    selected_pair_sell.clear()

    if os.path.exists(SIGNAL_FILE_BUY):
        os.remove(SIGNAL_FILE_BUY)

    if os.path.exists(SIGNAL_FILE_SELL):
        os.remove(SIGNAL_FILE_SELL)

    for i in trading_pairs:  # 1h
        output = filter_1h(i)

    for i in filtered_pairs_1h_buy:  # 5m
        output = filter_5m(i)
        if DEBUG:
            print(output)

    for i in filtered_pairs_5m:  # 1m
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
        print(
            f"{TxColors.BUY}{SIGNAL_NAME}: {selected_pair_buy} - Buy Signal Detected{TxColors.DEFAULT}"
        )
    if selected_pair_sell:
        print(
            f"{TxColors.RED}{SIGNAL_NAME}: {selected_pair_sell} - Sell Signal Detected{TxColors.RED}"
        )
    else:
        print(f"{TxColors.DEFAULT}{SIGNAL_NAME}: - not enough signal to buy")
    return signal_coins, signal_coins_sell


def do_work():
    """
    Main function for performing the analysis.
    """
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

            # pairs = get_symbols()

            if not threading.main_thread().is_alive():
                exit()
            print(f"{SIGNAL_NAME}: Analyzing {len(pairs)} coins")
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
            continue
