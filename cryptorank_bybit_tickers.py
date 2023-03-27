import requests
import ccxt
import yaml

# load yml file to dictionary
config = yaml.safe_load(open("./config.yaml"))

TICKERS = "tickerlists/tickers_bybit_USDT.txt"
TICKERS2 = "tickerlists/tickers_bybit.txt"

try:
    bybit = ccxt.bybit(
        {
            "apiKey": config["bybit"]["api_key"],
            "secret": config["bybit"]["secret_key"],
        }
    )
    markets = bybit.fetch_spot_markets(params={})
except ccxt.errors.AuthenticationError as e:
    print(f"Invalid API key: {e}")
    exit()

spot_markets = [market["symbol"] for market in markets if market["quote"] == "USDT" and market["active"] == True]


def get_cryptorank():
    url = "https://api.cryptorank.io/v1/currencies"
    payload = {"api_key": config["cryptorank"]["api_key"], "limit": 300}  # 148=100
    # 370=200
    req = requests.get(url, params=payload)
    print(req)
    dataj = req.json()["data"]
    li = [item.get("symbol") for item in dataj]
    ignore_usd = [x for x in li if not (x.endswith("USD") | x.startswith("USD"))]
    list1 = ["WBTC", "UST", "USDD", "DAI", "STETH", "CETH", "GBP", "PAX"]
    filtered = [x for x in ignore_usd if all(y not in x for y in list1)]
    filtered = [x + "/USDT" for x in filtered]
    return filtered


list1 = list(set(get_cryptorank()) & set(spot_markets))
list1.sort()
print(list1)
print(len(list1))

with open(f"{TICKERS}", "w") as output:
    for item in list1:
        output.write(str(item) + "\n")
