import os
import threading
import time
import requests
from binance.client import Client
import yaml
from globals import user_data_path

client = Client("", "")

DEFAULT_CREDS_FILE = user_data_path + "creds.yml"
# load yml file to dictionary
keys = yaml.safe_load(open(DEFAULT_CREDS_FILE))
TICKERS = "tickerlists/tickers_binance_USDT.txt"
TIME_TO_WAIT = 60


def get_binance():
    try:
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
        dataj = response.json()["symbols"]

        PAIRS_WITH = "USDT"
        li = [
            item.get("symbol")
            for item in dataj
            if item["status"] == "TRADING"
            and item["quoteAsset"] == PAIRS_WITH
            and item["isSpotTradingAllowed"]
        ]
        ignore = [
            "UP",
            "DOWN",
            "USD",
            "AEUR",
            "EUR",
            "DAI",
            "BUSD",
            "TUSD",
            "FDUSD",
            "GBP",
            "WBTC",
            "PAX",
            "USTC",
            "1000SATS",
            "WBETH",
        ]
        filtered = [
            x for x in li if not (x.endswith("USD") | x.startswith(tuple(ignore)))
        ]

        # filtered = [sub[: -4] for sub in symbols]   # without USDT

        return filtered
    except requests.exceptions.RequestException as e:
        return None


def get_cryptorank():
    url = "https://api.cryptorank.io/v1/currencies"
    payload = {"api_key": keys["cryptorank"]["api_key"], "limit": 300}  # 148=100
    # 370=200
    req = requests.get(url, params=payload)
    dataj = req.json()["data"]
    li = [item.get("symbol") for item in dataj]
    ignore_usd = [x for x in li if not (x.endswith("USD") | x.startswith("USD"))]
    list1 = ["WBTC", "UST", "USDD", "DAI", "STETH", "CETH", "GBP", "PAX"]
    filtered = [x for x in ignore_usd if all(y not in x for y in list1)]
    filtered = [x + "USDT" for x in filtered]
    return filtered


def get_binance_tickerlist():
    ticker_list = list(set(get_cryptorank()) & set(get_binance()))
    ticker_list.sort()
    length = len(ticker_list)

    with open(f"{TICKERS}", "w") as output:
        for item in ticker_list:
            output.write(str(item) + "\n")
    return length


def do_work():
    while True:
        try:
            if not os.path.exists(TICKERS):
                with open(TICKERS, "w") as f:
                    f.write("")

            if not threading.main_thread().is_alive():
                exit()
            print("Importing binance tickerlist")
            get_binance_tickerlist()
            print(
                f"Imported {TICKERS}: {get_binance_tickerlist()} coins. Waiting {TIME_TO_WAIT} minutes for next import."
            )

            time.sleep((TIME_TO_WAIT * 60))
        except Exception as e:
            print(f"Exception do_work() import binance tickerlist: {e}")
            continue
        except KeyboardInterrupt as ki:
            print(ki)
            exit()
