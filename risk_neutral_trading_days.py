# calculate standard deviation
std_dev = df.pct_change().std() * np.sqrt(365/252)

# MONTE CARLO SIMULATION
# assume normal distribution of returns
counter1 = 0
counter2 = 0
cur_price = df.iloc[-1]
for i in range(trials):
    price = cur_price
    daily_return = np.random.normal(drift / 365, std_dev, days_to_expiration) + 1
    strike_met = False
    for r in daily_return:
        price *= r
        if price >= strike_price and strike_met == False:
            counter1 += 1
            strike_met = True
    if price >= strike_price:
        counter2 += 1

print(f"The probability of being above strike at any time before expiration date is {counter1/trials * 100} %")
print(f"The probability of being above strike on expiration date {counter2/trials * 100} %")
