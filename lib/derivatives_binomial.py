import numpy as np
import pandas as pd

def calc_probability(rf, q, vol, T, steps):

    # up_probability
    return (np.exp((rf-q)*T/steps) - np.exp(-vol*np.sqrt(T/steps))) / (np.exp(vol*np.sqrt(T/steps)) - np.exp(-vol*np.sqrt(T/steps)))


def calc_spot_tree(S0, vol, T, steps):

    # allocate matrix for stock prices
    stock_prices = pd.DataFrame(np.zeros((steps+1, steps+1)))
    stock_prices.iloc[0, 0] = S0

    # calculate up and down factors
    up = np.exp(vol*np.sqrt(T/steps))
    down = 1/up

    # calculate stock prices
    for i in range(1, steps+1):
        stock_prices.iloc[0, i] = stock_prices.iloc[0, i-1] * up
        for j in range(1, i+1):
            stock_prices.iloc[j, i] = stock_prices.iloc[j-1, i-1] * down

    # replace zeros with NaN
    stock_prices.replace(0, np.nan, inplace=True)

    return stock_prices


def ccr_eur_call(S0, X, rf, q, vol, T, steps):

    # calculate up and down factors
    up = np.exp(vol*np.sqrt(T/steps))
    down = 1/up

    # calculate risk-neutral probabilities
    prop_up = calc_probability(rf, q, vol, T, steps)
    prop_down = 1 - prop_up

    # calculate stock prices
    stock_prices = calc_spot_tree(S0, vol, T, steps)

    # calculate option prices
    option_prices = pd.DataFrame(np.zeros((steps+1, steps+1)))
    option_prices.iloc[:, steps] = np.maximum(0, stock_prices.iloc[:, steps] - X)

    # calculate option prices at each node
    for i in range(steps-1, -1, -1):
        for j in range(i+1):
            option_prices.iloc[j, i] = np.exp(-rf*T/steps) * (prop_up * option_prices.iloc[j, i+1] + prop_down * option_prices.iloc[j+1, i+1])

    return option_prices


def ccr_amer_call(S0, X, rf, q, vol, T, steps):
    
    # calculate up and down factors
    up = np.exp(vol*np.sqrt(T/steps))
    down = 1/up

    # calculate risk-neutral probabilities
    prop_up = calc_probability(rf, q, vol, T, steps)
    prop_down = 1 - prop_up

    # calculate stock prices
    stock_prices = calc_spot_tree(S0, vol, T, steps)

    # calculate option prices
    option_prices = pd.DataFrame(np.zeros((steps+1, steps+1)))
    option_prices.iloc[:, steps] = np.maximum(0, stock_prices.iloc[:, steps] - X)

    # calculate option prices at each node
    for i in range(steps-1, -1, -1):
        for j in range(i+1):
            option_prices.iloc[j, i] = np.maximum(np.exp(-rf*T/steps) * (prop_up * option_prices.iloc[j, i+1] + prop_down * option_prices.iloc[j+1, i+1]), stock_prices.iloc[j, i] - X)

    return option_prices


def ccr_amer_put(S0, X, rf, q, vol, T, steps):
    
    # calculate up and down factors
    up = np.exp(vol*np.sqrt(T/steps))
    down = 1/up

    # calculate risk-neutral probabilities
    prop_up = calc_probability(rf, q, vol, T, steps)
    prop_down = 1 - prop_up

    # calculate stock prices
    stock_prices = calc_spot_tree(S0, vol, T, steps)

    # calculate option prices
    option_prices = pd.DataFrame(np.zeros((steps+1, steps+1)))
    option_prices.iloc[:, steps] = np.maximum(0, X - stock_prices.iloc[:, steps])

    # calculate option prices at each node
    for i in range(steps-1, -1, -1):
        for j in range(i+1):
            option_prices.iloc[j, i] = np.maximum(np.exp(-rf*T/steps) * (prop_up * option_prices.iloc[j, i+1] + prop_down * option_prices.iloc[j+1, i+1]), X - stock_prices.iloc[j, i])

    return option_prices