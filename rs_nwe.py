"""
CUSTOM_LIST: False
"""
from binance.client import Client
import numpy as np
import threading
import os
import pandas_ta as ta
import pandas as pd
from datetime import datetime
import time
import math
from loguru import logger

client = Client("", "")
TIME_TO_WAIT = 1  # Minutes to wait between analysis
DEBUG = True
TICKERS = 'tickers_all_withUSDT.txt'
SIGNAL_NAME = 'rs_signals_nwe'
SIGNAL_FILE_BUY = 'signals/' + SIGNAL_NAME + '.buy'
SIGNAL_FILE_SELL = 'signals/' + SIGNAL_NAME + '.sell'


# for colourful logging to the console
class TxColors:
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


selected_pair_buy = []
selected_pair_sell = []


def nadarayawatsonenvelope(dtloc, source='close', bandwidth=8, window=500, mult=3):
    """
    // This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
     https://creativecommons.org/licenses/by-nc-sa/4.0/
    // Nadaraya-Watson Envelope [LUX]
      https://www.tradingview.com/script/Iko0E2kL-Nadaraya-Watson-Envelope-LUX/
     :return: up and down
     translated for freqtrade: viksal1982  viktors.s@gmail.com
    """
    dtNWE = dtloc.copy()
    dtNWE['nwe_up'] = np.nan
    dtNWE['nwe_down'] = np.nan
    wn = np.zeros((window, window))
    for i in range(window):
        for j in range(window):
            wn[i, j] = math.exp(-(math.pow(i - j, 2) / (bandwidth * bandwidth * 2)))
    sumSCW = wn.sum(axis=1)

    def calc_nwa(dfr, init=0):
        global calc_src_value
        if init == 1:
            calc_src_value = list()
            return
        calc_src_value.append(dfr[source])
        mae = 0.0
        y2_val = 0.0
        y2_val_up = np.nan
        y2_val_down = np.nan
        if len(calc_src_value) > window:
            calc_src_value.pop(0)
        if len(calc_src_value) >= window:
            src = np.array(calc_src_value)
            sumSC = src * wn
            sumSCS = sumSC.sum(axis=1)
            y2 = sumSCS / sumSCW
            sum_e = np.absolute(src - y2)
            mae = sum_e.sum() / window * mult
            y2_val = y2[-1]
            y2_val_up = y2_val + mae
            y2_val_down = y2_val - mae
        return y2_val_up, y2_val_down

    calc_nwa(None, init=1)
    dtNWE[['nwe_up', 'nwe_down']] = dtNWE.apply(calc_nwa, axis=1, result_type='expand')
    return dtNWE[['nwe_up', 'nwe_down']]


@logger.catch
def filter1(pairs):
    interval = '1h'
    symbol = pairs
    start_str = '21 days ago UTC'
    end_str = f'{datetime.now()}'
    df = pd.DataFrame(client.get_historical_klines(symbol, interval, start_str, end_str)[:-1]).astype(float)
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index, unit='ms')

    ss = nadarayawatsonenvelope(df, source='close', bandwidth=8, window=500, mult=3)

    if ss.nwe_down.iat[-1] > df.close.iat[-1]:
        selected_pair_buy.append(symbol)
        if DEBUG:
            print('found buy')
            print("on 1h timeframe " + symbol)

    elif ss.nwe_up.iat[-1] < df.close.iat[-1]:
        selected_pair_sell.append(symbol)
        if DEBUG:
            print('found sell')
            print("on 1h timeframe " + symbol)

    return selected_pair_buy, selected_pair_sell


def analyze(trading_pairs):
    signal_coins_buy = {}
    signal_coins_sell = {}
    selected_pair_buy.clear()
    selected_pair_sell.clear()

    if os.path.exists(SIGNAL_FILE_BUY):
        os.remove(SIGNAL_FILE_BUY)

    if os.path.exists(SIGNAL_FILE_SELL):
        os.remove(SIGNAL_FILE_SELL)

    for i in trading_pairs:  # 1h
        output = filter1(i)

    for pair in selected_pair_buy:
        signal_coins_buy[pair] = pair
        with open(SIGNAL_FILE_BUY, 'a+') as f:
            f.writelines(pair + '\n')

    for pair in selected_pair_sell:
        signal_coins_sell[pair] = pair
        with open(SIGNAL_FILE_SELL, 'a+') as f:
            f.writelines(pair + '\n')

    if selected_pair_buy:
        print(f'{TxColors.BUY}{SIGNAL_NAME}: {selected_pair_buy} - Buy Signal Detected{TxColors.DEFAULT}')
    elif selected_pair_sell:
        print(f'{TxColors.BUY}{SIGNAL_NAME}: {selected_pair_sell} - Sell Signal Detected{TxColors.DEFAULT}')
    else:
        print(f'{TxColors.DEFAULT}{SIGNAL_NAME}: - not enough signal to buy')
    return signal_coins_buy, signal_coins_sell


def do_work():
    while True:
        try:
            if not os.path.exists(TICKERS):
                time.sleep((TIME_TO_WAIT * 60))
                continue

            signal_coins_buy = {}
            signal_coins_sell = {}
            pairs = {}
            with open(TICKERS) as f:
                pairs = f.read().splitlines()

            # pairs = get_symbols()

            if not threading.main_thread().is_alive():
                exit()
            print(f'{SIGNAL_NAME}: Analyzing {len(pairs)} coins')
            signal_coins_buy, signal_coins_sell = analyze(pairs)
            print(
                f'{SIGNAL_NAME}: {len(signal_coins_buy)} '
                f'coins with Buy Signals. Waiting {TIME_TO_WAIT} minutes for next analysis.')
            print(
                f'{SIGNAL_NAME}: {len(signal_coins_sell)} '
                f'coins with Sell Signals. Waiting {TIME_TO_WAIT} minutes for next analysis.')

            time.sleep((TIME_TO_WAIT * 60))
        except Exception as e:
            print(f'{SIGNAL_NAME}: Exception do_work() 1: {e}')
            continue
        except KeyboardInterrupt as ki:
            continue
