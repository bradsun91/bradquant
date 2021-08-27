from .toolbox import counting_bdays
from .toolbox import fred_3mon_ir
from fredapi import Fred 
import numpy as np


def sharpe_ratio(pct_changes, start, end):
    first = pct_changes[0]
    last = pct_changes[-1]
    total_return = ((last - first)/first)
    annual_trading_days = 252
    actual_trading_days = counting_bdays(start, end)
    annual_rf = fred_3mon_ir(start, end).fillna(method='ffill')/100 # annual_rf is series, so we need to add the [-1] behind it
    daily_rf = (annual_rf/annual_trading_days) # because the number from fred does not include percentage sign, so needing to be divided by 100
    risk_free_rate = daily_rf
    excess_daily_return = pct_changes - risk_free_rate
    expected_return = np.mean(excess_daily_return)
    daily_std = np.std(pct_changes)
    stddev = np.std(excess_daily_return)
    return_distribution = pct_changes.describe(percentiles=[.10, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    sharpe_ratio = expected_return/stddev*np.sqrt(annual_trading_days)
    # pct_changes.hist(by=None, bins=150, figsize = (20,8))
    
    return sharpe_ratio
    # print ("Sharpe Ratio: {}".format(sharpe_ratio))
    # print ("Return Distribution: {}".format(return_distribution))
    # print ("Daily std is {}".format(daily_std))

#pct_chgs = qqq_spy.pct_change()
#sharpe_ratio(pct_chgs, datetime(2016, 1, 1), datetime(2017, 7, 21))