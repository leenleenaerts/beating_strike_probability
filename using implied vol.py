import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from yahoo_fin import options, stock_info
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('display.max_columns', None)


# PARAMETERS TO SET
discount_rate = 0.0525
arithmic_div_yield = 0.005
stock = "AAPL"
expiration_date = "08/04/2023"
K = 200


today = datetime.today()
days_to_expiration = (datetime.strptime(expiration_date, "%m/%d/%Y") - today).days


# calculate true yield
cash_price = 1 - discount_rate * days_to_expiration/360
true_yield = 365/days_to_expiration*np.log(1+0.004375/cash_price)

# calculate continuous dividend yield
cont_div_yield = np.log(1 + arithmic_div_yield)

# calculate risk neutral drift
drift = true_yield - cont_div_yield

# get information on implied volatility from calls at Strike Price
chain = options.get_calls(stock, expiration_date)
mean = stock_info.get_live_price(stock) * np.exp(drift*days_to_expiration/365)
sd = float((chain[chain["Strike"] == 200.0].iloc[0]["Implied Volatility"])[:-1]) / 100 * np.sqrt(days_to_expiration/365) * mean

mu = np.log(mean ** 2.0 / (sd ** 2.0 + mean ** 2.0) ** 0.5)
sigma = (np.log(1.0 + sd ** 2.0 / mean ** 2.0)) ** 0.5

l = stats.lognorm(s=sigma, scale=np.exp(mu))

P = 1 - l.cdf(K)

print(f"The probability of {stock} trading above {K} on {expiration_date} is {P*100:0.1f}%")

