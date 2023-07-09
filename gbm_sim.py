

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import GeometricBrownianMotionGenerator, bsm_pricer, bsm_delta,\
    cont_rate_from_tbill

sns.set(rc={'figure.figsize': (11.7, 8.27)})


def plot_gbm_equity(log_filepath, out_filepath):

    data = pd.read_csv(log_filepath)
    fig, axes = plt.subplots()
    fig.suptitle('GBM Equity Paths')

    # plot price chart
    equity = data.set_index('Time Step')
    axes.plot(equity)
    axes.set(ylabel='Index Level')

    fig.savefig(out_filepath)


if __name__ == '__main__':

    # Inputs
    put_call = 'call'
    spot = 100
    strike = 115
    ttm = 365  # Actual days to expiry
    rfr = 0.0525  # Quoted T-Bill rate matching the days to expiry
    div_arith = 0.005  # Arithmetic dividend yield
    vol = 0.2  # Standard deviation of log returns

    # GBM configs
    n_paths = 10000
    dt = 1  # Number of days in each time step for the GBM simulation
    breach_strike = 115
    breach_direction = 'above'

    # Continuous calcs
    rf_cont = cont_rate_from_tbill(rfr, ttm)
    div_cont = np.log(1 + div_arith)

    # Price option using BSM and input vars
    calc_price = bsm_pricer(put_call, spot, strike, ttm / 365, rf_cont,
                            div_cont, vol)

    calc_delta = bsm_delta(put_call, spot, strike, ttm / 365, rf_cont,
                           div_cont, vol)

    print(f"BSM Price:              {round(calc_price, 4)}")
    print(f"BSM Delta:              {round(calc_delta, 4)}")

    # Create GBM Generator object using input vars
    equity_gen = GeometricBrownianMotionGenerator(
        drift_rate=rf_cont-div_cont,
        volatility=vol,
        index_level=spot)

    # Generate Paths
    sim_results = equity_gen.generate_path(proj_length=ttm/365, dt=1/365,
                                           n_paths=n_paths)

    sim_results.index.name = 'Time Step'

    # Plot a line plot of the simulated paths and save to CSV (optional)
    sim_results.to_csv('./output/sim_results.csv')
    plot_gbm_equity('./output/sim_results.csv', './output/gbm_paths.png')

    # Calculate the simulated option values at expiration
    if put_call == 'call':
        payoffs = np.maximum(sim_results[-1:] - strike, 0)
    elif put_call == 'put':
        payoffs = np.maximum(strike - sim_results[-1:], 0)
    else:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    # Discount the mean payoff to today
    sim_price = np.exp(-rf_cont * ttm / 365) * np.mean(payoffs)

    print(f"Simulated Price:        {round(sim_price, 4)}")

    # Count the total number of paths which breached the specified strike
    if breach_direction == 'above':
        path_extremes = sim_results.max()
        breaches = path_extremes > breach_strike
    elif breach_direction == 'below':
        path_extremes = sim_results.min()
        breaches = path_extremes < breach_strike
    else:
        raise ValueError("'breach_direction' must be 'above' or 'below'")

    n_breaches = breaches.sum()
    prob_breach = n_breaches / n_paths

    # Probability of breaching the strike threshold at any time during the path
    # A.K.A "Probability of Touch"
    print(f"Prob. of {breach_direction} {breach_strike}:    {prob_breach:.2%}")
