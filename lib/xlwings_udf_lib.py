from scipy.stats import norm

import numpy as np
import pandas as pd
import xlwings as xw

@xw.func
def call_price(S, K, T, r, sigma):
    """
    Calculate the call option price using the Black-Scholes formula.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying stock

    Returns:
    float: Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


@xw.func
def double_sum(x, y):
    """Returns twice the sum of the two arguments"""
    return 2 * (x + y)