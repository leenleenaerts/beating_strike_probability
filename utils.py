

import math
import numpy as np
import pandas as pd
from scipy.stats import norm


def std_norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))


def cont_rate_from_tbill(quoted_yield, days_to_maturity):
    # NOTE: This is only necessary for "discount instruments" such as t-bills,
    # which are quoted on an ACTUAL/360 day basis
    # See here for further reading:
    # https://forum.bionicturtle.com/threads/continuous-compounding-for-t-bill.9056/

    # Determine the cash price of the t-bill
    cash_price = 100 - quoted_yield * 100 * days_to_maturity / 360

    # Calculate the cash interest payment
    cash_interest = 100 - cash_price

    # Calculate the continuously compounded yield on an ACTUAL/365 basis
    cont_yld = 365 / days_to_maturity * np.log(1 + cash_interest / cash_price)

    return cont_yld


def bsm_pricer(put_call, spot, strike, ttm, rfr, div_yield, sigma):
    """
    Implementation of the generalized Black Scholes Merton option
    pricing formula using continuous dividend yield.

    :param put_call: string of 'put' for put or 'call' for call
    :param spot: Spot price of the underlying
    :param strike: Strike price of the option
    :param ttm: Time to maturity (in year fractions)
    :param rfr: The continuously compounded annualized risk-free rate
    :param div_yield: The continuously compounded annualized dividend yield
    :param sigma: The implied volatility of the option
    :return: float of the option price
    """

    d1 = ((math.log(spot / strike)
          + (rfr - div_yield + 0.5 * sigma ** 2) * ttm)
          / (sigma * math.sqrt(ttm)))

    d2 = d1 - sigma * math.sqrt(ttm)

    if put_call == 'call':
        price = (spot * math.exp(-div_yield * ttm) * norm.cdf(d1)
                 - strike * math.exp(-rfr * ttm) * norm.cdf(d2))

    elif put_call == 'put':
        price = (strike * math.exp(-rfr * ttm) * norm.cdf(-d2)
                 - spot * math.exp(-div_yield * ttm) * norm.cdf(-d1))

    else:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    return price


def bsm_delta(put_call: str, spot: float, strike: float, ttm: float,
              rfr: float, div_yield: float, sigma: float) -> float:
    """
    Return Black-Scholes delta of an option.

    :param put_call: string of 'put' for put or 'call' for call
    :param spot: float of the underlying asset price
    :param strike: float of the strike price
    :param ttm: float of the time to expiration in years
    :param rfr: risk-free continuous (zero) interest rate
    :param div_yield: float of the continuous annualized dividend yield
    :param sigma: float of the implied volatility
    :return: float of the BSM option delta
    """

    d1 = ((math.log(spot / strike)
          + (rfr - div_yield + 0.5 * sigma ** 2) * ttm)
          / (sigma * math.sqrt(ttm)))

    if put_call == 'put':
        delta = math.exp(-div_yield * ttm) * std_norm_cdf(-d1) * - 1

    elif put_call == 'call':
        delta = math.exp(-div_yield * ttm) * std_norm_cdf(d1)

    else:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    return delta


class GeometricBrownianMotionGenerator(object):
    """
    Class for generating index levels and returns that follow a
    Geometric Brownian Motion.

    NOTE: Inputs for the drift rate are continuous rates!
    """

    def __init__(self, drift_rate, volatility,
                 index_level=100, seed=None):
        """
        :param drift_rate: float of the continuous equity drift rate
        :param volatility: float of the volatility (vol of log-returns)
        :param index_level: float of the initial index level
        :param seed: integer of the seed to use in the random number generator
        """

        self.drift = drift_rate
        self.vol = volatility
        self.index_level = index_level
        self.seed = seed

    def generate_path(self, proj_length, dt, n_paths):
        """
        Function to generate a DataFrame of equity returns.

        :param proj_length: int of the total projection periods in years
        :param dt: float of the year fractions to use as time steps
        :param n_paths: int of the number of paths to generate
        :return: DataFrame of the equity returns at each time step
        """

        n_steps = int(proj_length / dt)

        np.random.seed(self.seed)

        x = np.random.normal(
            (self.drift - 0.5 * self.vol**2) * dt,
            self.vol * math.sqrt(dt),
            size=(n_steps, n_paths))

        x_cumulative = np.cumsum(x, axis=0)
        levels = self.index_level * np.exp(x_cumulative)
        paths = pd.DataFrame(levels)
        starting_levels = pd.Series([self.index_level] * n_paths)
        starting_levels = pd.DataFrame(starting_levels).T
        paths = pd.concat([starting_levels, paths]).reset_index(drop=True)
        return paths
