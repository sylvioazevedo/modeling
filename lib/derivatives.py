from enum import Enum
from scipy import stats
from scipy.stats import norm
from scipy.optimize import fsolve

import math
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

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


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

def bs_strike_from_delta(S, delta, T, r, vol, type: OptionType = OptionType.CALL):
    '''
    Calculate the strike price of a European option given the delta using the Black-Scholes formula.
    
    :param S: Spot price
    :param delta: Option delta
    :param T: Time to maturity
    :param r: Risk-free rate
    :param vol: Volatility
    :param type: Option type
    :return: Strike price
    '''
    if type == OptionType.CALL:
        d1 = norm.ppf(delta)
        return S*np.exp(vol*np.sqrt(T)*d1 - (r + vol**2/2)*T)
    
    elif type == OptionType.PUT:
        d1 = norm.ppf(delta)
        return S*np.exp(vol*np.sqrt(T)*(d1 - 1) - (r + vol**2/2)*T)
    
    raise ValueError('Option type not valid.')

def calculate_strike(delta, S, r, sigma, T, option_type='call'):
    def equation(K):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return norm.cdf(d1) - delta
        elif option_type == 'put':
            return norm.cdf(d1) - 1 + delta
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    K_initial_guess = S  # Initial guess for the strike price
    K = fsolve(equation, K_initial_guess)[0]
    return K

def calculate_delta(K, S, r, T, deltas, a, sigma, option_type='call'):    
    
    max_iter = 1000

    while max_iter > 0:

        d1 = (np.log(S / K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))

        delta = norm.cdf(d1)

        if option_type == 'call':              

            _sigma = calculate_vol_by_delta(delta, deltas, a)

            if abs(_sigma - sigma) < 10**-8:
                return delta, _sigma

            sigma = _sigma

        elif option_type == 'put':
            
            sigma = calculate_vol_by_delta(delta - 1, deltas, a)

            if abs(_sigma - sigma) < 10**-8:
                return delta -1 , _sigma

            sigma = _sigma
        
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        max_iter -= 1

    raise ValueError("Max iterations reached")


    # def equation(sigma):        

        #     d1 = (np.log(S / K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        #     delta = norm.cdf(d1)

        #     if option_type == 'call':              

        #         _sigma = calculate_vol_by_delta(delta, deltas, a)

        #         return _sigma - sigma
            
        #     elif option_type == 'put':
                
        #         return calculate_vol_by_delta(delta - 1, deltas, a)
            
        #     else:
        #         raise ValueError("option_type must be 'call' or 'put'")
        
        # # Initial guess for delta
        # sigma = fsolve(equation, init_sigma)[0]
        
        # return (np.log(S / K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))

def find_best_sigma(strike, strikes, sigmas):

    for i in range(len(strikes) - 1):

        if strike >= strikes[i] and strike < strikes[i + 1]:

            return sigmas[i]

    return sigmas[-1]

def calculate_vol_by_delta(delta, deltas, a):

    if delta < deltas[1]:
        vol = a[0]*delta**3 + a[1]*delta**2 + a[2]*delta + a[3]

    elif delta < deltas[2]:
        vol = a[4]*delta**3 + a[5]*delta**2 + a[6]*delta + a[7]

    elif delta < deltas[3]:
        vol = a[8]*delta**3 + a[9]*delta**2 + a[10]*delta + a[11]

    else:
        vol = a[12]*delta**3 + a[13]*delta**2 + a[14]*delta + a[15]    

    return vol
    