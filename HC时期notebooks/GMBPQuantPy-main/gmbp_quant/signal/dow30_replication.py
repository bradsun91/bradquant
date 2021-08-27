# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:23:55 2021

@author: kaiyan
"""

#####
# Library required:
# 1. tabulate (pip3 install tabulate)
# 2. pytickersymbols (pip3 install pytickersymbols)
# 3. pandas (pip3 install pandas)
# 4. yfinance (pip3 install yfinance)
#####

####
# Utility related Libraries
####
# Library for pretty print the result
from tabulate import tabulate
# Library for working with dataframe
import pandas as pd
pd.options.mode.chained_assignment = None
####

####
# Financial related Libraries
####
# Library for getting components for stocks and indices
from pytickersymbols import PyTickerSymbols
# Library for fetching stock related information
import yfinance as yf
####

print("=> Starting ...")

# 1. Fetching tickers for components of DJI
print("=> Fetching tickers for components of DJI ...")
stock_data = PyTickerSymbols()
dji = stock_data.get_stocks_by_index('DOW JONES')
dji_tickers = list(map(lambda x: x['symbol'], list(dji)))

# 2. Fetching data for all tickers for components of DJI and DJI itself
# Given dow did not change its component since 9/1/2020, only test the replication starting from 9/1/2020
print("=> Fetching data for all tickers for components of DJI and DJI itself ...")
all_dji_tickers_data = yf.download(" ".join(dji_tickers), start="2021-02-19", end="2021-02-26")
all_dji_data = yf.download("DJI", start="2020-09-01", end="2021-01-31")

# 3. Extracting required columns for ease of computation later
print("=> Extracting required columns for ease of computation later ...")
dji_tickers_data_close = all_dji_tickers_data['Close']
dji_data_close = all_dji_data['Close']

# 4. Trying to replicate divisor for DJI and calculate absolute difference
print("=> Trying to replicate divisor for DJI and calculate absolute difference ...")
dji_tickers_data_close['Sum'] = dji_tickers_data_close.sum(axis=1)
dji_data_close['Divisor'] = dji_tickers_data_close['Sum'] / dji_data_close
dji_data_close['Error'] = dji_data_close['Divisor'].subtract(0.15198707565833).abs() #0.15198707565833 is the divisor value from wiki

# 5. Print out the results
print("\n/**********")
print("/* The absolute difference between replicated divisor and wikipedia provided divisor: ")
print("/**********")
print(tabulate(dji_data_close['Error'].to_frame('Absolute Difference'), headers='keys', tablefmt='psql')) 