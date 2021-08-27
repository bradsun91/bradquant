from pcm_tools.toolbox import google_data_close_price
from datetime import datetime
from scipy.stats import norm
from math import log
import numpy as np
import pandas as pd


# Calculate historical volatility

def annualized_hist_vol(ticker, start, end, rolling_window_span):
    df = google_data_close_price(ticker, start, end)
    ticker_close =df[ticker]
    ticker_pct = ticker_close.pct_change()
    rolling_std_daily = ticker_pct.rolling(window = rolling_window_span).std()
    rolling_std_annualized = rolling_std_daily*np.sqrt(252)
    return rolling_std_annualized


# Calculate implied volatility

n = norm.pdf
N = norm.cdf

def bs_price(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (log(S/K)+(r+v*v/2.)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if cp_flag == 'c':
        price = S*np.exp(-q*T)*N(d1)-K*np.exp(-r*T)*N(d2)
    else:
        price = K*np.exp(-r*T)*N(-d2)-S*np.exp(-q*T)*N(-d1)
    return price

def bs_vega(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (log(S/K)+(r+v*v/2.)*T)/(v*np.sqrt(T))
    return S * np.sqrt(T)*n(d1)


def find_vol(target_value, call_put, S, K, T, r):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5

    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_price(call_put, S, K, T, r, sigma)
        vega = bs_vega(call_put, S, K, T, r, sigma)

        price = price
        diff = target_value - price  # our root

        print (i, sigma, diff)

        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma












