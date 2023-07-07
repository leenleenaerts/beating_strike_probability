import numpy as np
import pandas_datareader.data as web
from datetime import datetime, timedelta
import yfinance as yfin
import requests


yfin.pdr_override()

# PARAMETERS TO SET
discount_rate = 0.0525
arithmic_div_yield = 0.005
stock = "BTC-USD"
expiration_date = "07/14/2023"
K = 32500


trials = 5000
today = datetime.today()
days_to_expiration = (datetime.strptime(expiration_date, "%m/%d/%Y") - today).days
start = today - timedelta(days = 365*5)
end = datetime.today()

# calculate true yield
cash_price = 1 - discount_rate * days_to_expiration/360
true_yield = 365/days_to_expiration*np.log(1+0.004375/cash_price)

# calculate continuous dividend yield
cont_div_yield = np.log(1 + arithmic_div_yield)

# calculate risk neutral drift
drift = true_yield - cont_div_yield

# read stock data
df = web.DataReader(stock, start, end)["Adj Close"]

# calculate standard deviation
std_dev = df.pct_change().std() * np.sqrt(252/3650000)

# MONTE CARLO SIMULATION
# assume normal distribution of returns
counter1 = 0
counter2 = 0
cur_price = df.iloc[-1]
for i in range(trials):
    price = cur_price
    daily_return = np.random.normal(drift / 3650000, std_dev, days_to_expiration*10000) + 1
    strike_met = False
    for r in daily_return:
        price *= r
        if price >= K and strike_met == False:
            counter1 += 1
            strike_met = True
    if price >= K:
        counter2 += 1

print(f"current price of {stock} = ${cur_price}.")
print(f"The probability of {stock} being above ${K} at any time before expiration date is {counter1/trials * 100} %")
print(f"The probability of {stock} being above ${K} on expiration date {counter2/trials * 100} %")
