import requests
from binance.client import Client
import json

client = Client("", "")

TICKERS = 'tickers_all_withUSDT.txt'


def get_all_market_pairs():
    try:
        response = requests.get("https://api.binance.com/api/v1/exchangeInfo")
        res_json = json.loads(response.text)

        symbols = res_json.get('symbols', None)

        PAIRS_WITH = 'USDT'
        ignore = ['UP', 'DOWN', 'BEAR', 'BULL', 'BUSD', 'TUSD', 'AUDUSDT', 'BKRW', 'DAI', 'ERD', 'EUR', 'USDS',
                  'USDC', 'GBP', 'RUB', 'BTCST', 'PAX', 'USDP', 'SUSD']

        markets = []

        for i in symbols:
            if i['status'] == "TRADING" and i['quoteAsset'] == PAIRS_WITH and i['isSpotTradingAllowed']:
                if all(item not in i['symbol'] for item in ignore):
                    markets.append(i['symbol'])
                markets.sort()

        # markets = [sub[: -4] for sub in symbols]   # without USDT

        return markets
    except requests.exceptions.RequestException as e:
        return None


print(get_all_market_pairs())
print('')
print(len(get_all_market_pairs()))


# get symbols from binance and write to txt

with open(f'{TICKERS}', "w") as output:
    for item in get_all_market_pairs():
        output.write(str(item) + "\n")

# with open(TICKERS, "r") as f:
#     lines = f.read().splitlines()
#     print(lines)

# output.write(str(get_symbols())+"\n")
