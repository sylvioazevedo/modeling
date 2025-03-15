from enum import Enum
from scipy import stats
from scipy.stats import norm

import numpy as np

# constants
lim_zero = 0.000001

# enumeration
class OptionType(Enum):
    CALL = 1
    PUT = 2

def calc_payoff(S, K, type: OptionType):

    if type == OptionType.CALL:
        return np.maximum(S - K, 0)
    
    elif type == OptionType.PUT:
        return np.maximum(K - S, 0)
    
    else:
        raise ValueError('Option type not valid.')
    
    # return np.maximum(K - S, 0) if type == OptionType.PUT else np.maximum(S - K, 0)
    
def calc_call_payoff(S, K):
    return calc_payoff(S, K, OptionType.CALL)

def calc_put_payoff(S, K):
    return calc_payoff(S, K, OptionType.PUT)

# binary (digital) option
def binary(S, K, type: OptionType):

    e = lim_zero

    if type == OptionType.CALL:
        return calc_payoff(S, K, OptionType.CALL) - calc_payoff(S, K + e, OptionType.CALL)
    
    elif type == OptionType.PUT:
        return calc_payoff(S, K, OptionType.PUT) - calc_payoff(S, K + e, OptionType.PUT)
    
    raise ValueError('Option type not valid.')


def bs_call_option(S, K, T, r, vol):
    '''
    Calculate the price of a European call option using the Black-Scholes formula.
    
    :param S: Spot price
    :param K: Strike price
    :param T: Time to maturity
    :param r: Risk-free rate
    :param vol: Volatility
    :return: Call option price
    '''
    d1 = (np.log(S/K) + (r + vol**2/2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_put_option(S, K, T, r, vol):
    '''
    Calculate the price of a European put option using the Black-Scholes formula.
    
    :param S: Spot price
    :param K: Strike price
    :param T: Time to maturity
    :param r: Risk-free rate
    :param vol: Volatility
    :return: Put option price
    '''
    d1 = (np.log(S/K) + (r + vol**2/2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_call_delta(S, K, T, r, vol):
    '''
    Calculate the delta of a European call option using the Black-Scholes formula.
    
    :param S: Spot price
    :param K: Strike price
    :param T: Time to maturity
    :param r: Risk-free rate
    :param vol: Volatility
    :return: Call option delta
    '''
    d1 = (np.log(S/K) + (r + vol**2/2)*T) / (vol*np.sqrt(T))

    return norm.cdf(d1)