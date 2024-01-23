import requests
from binance.client import Client
import yaml

# load yml file to dictionary
keys = yaml.safe_load(open("../config.yaml"))

client = Client("", "")
TICKERS = "tickerlists/tickers_all_USDT.txt"


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


list1 = list(set(get_cryptorank()) & set(get_binance()))
list1.sort()
print(list1)
print(len(list1))

with open(f"{TICKERS}", "w") as output:
    for item in list1:
        output.write(str(item) + "\n")
