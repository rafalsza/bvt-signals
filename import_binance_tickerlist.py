import requests
import yaml
from binance.client import Client

client = Client("", "")

TICKERS = "tickerlists/tickers_binance.txt"

# load yml file to dictionary
config = yaml.safe_load(open("./config.yaml"))


def get_all_market_pairs():
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
        ignore = config["coins"]["IGNORED"]

        filtered = [
            x for x in li if not (x.endswith("USD") | x.startswith(tuple(ignore)))
        ]
        filtered.sort()

        filtered = [sub[:-4] for sub in filtered]  # without USDT

        return filtered
    except requests.exceptions.RequestException as e:
        return None


print(get_all_market_pairs())
print("")
print(len(get_all_market_pairs()))

# get symbols from binance and write to txt

with open(f"{TICKERS}", "w") as output:
    for item in get_all_market_pairs():
        output.write(str(item) + "\n")

# with open(TICKERS, "r") as f:
#     lines = f.read().splitlines()
#     print(lines)

# output.write(str(get_symbols())+"\n")
