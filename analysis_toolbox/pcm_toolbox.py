"""
Contents:

1. Open CSV file
2. Count business days
3. Create multiple graphs
4. Import multi-tickers' close price
5. Stats Arb Pair's Z-score plotting
6. Any single time series's sharpe ratio
7. Fred's economic data import

"""

#------------------------------------------------------------------------------------------------------------------------------ 


# Open CSV file:
import csv
def open_with_csv(filename):
    data = []
    with open(filename, encoding = 'utf-8') as tsvin:
        tie_reader = csv.reader(tsvin, delimiter = d) # just in case we have different types of delimiter
        for line in tie_reader:
            data.append(line)
    # use UTF-8 encoding, just in case there are any special characters that may be an apostrophe or accent mark
    return data


#------------------------------------------------------------------------------------------------------------------------------

# Counting business days:
import numpy as np
import datetime as dt

def counting_bdays(start, end):
    days = np.busday_count(start, end)    
    return days

# counting_bdays(dt.date(2016, 1, 1), dt.date(2017,7,15))


#------------------------------------------------------------------------------------------------------------------------------ 

# Creating multiple graphs on one single chart:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_datareader.data as wb
import seaborn as sns


def two_separate_graphs(series1, series2):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(25,20), sharex=True)
    
    series1.plot(ax = ax1, color = 'blue', lw=1)
    series2.plot(ax = ax2, color = 'red', lw=1)
    
def two_graphs_twinx(series1, series2):
    fig = plt.figure(figsize = (30, 15))
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    
    series1.plot(ax = ax1, color = 'blue', lw = 1)
    series2.plot(ax = ax2, color = 'red', lw = 1)

    ax1.axhline(0, color = 'blue', linestyle='--', lw=1)
    ax2.axhline(0, color = 'red', linestyle='--', lw=1)
    

def three_graphs(series1, series2, series3):

    fig, ax1 = plt.subplots(1,1, figsize=(25,10), sharex=True)
    
    series1.plot(ax = ax1, color = 'blue', lw=1)
    series2.plot(ax = ax1, color = 'red', lw=1)
    series3.plot(ax = ax1, color = 'green', lw = 1)
 
    
def three_graphs_twinx(series1, series2, series3):
    
    fig = plt.figure(figsize = (30, 15))
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    
    series1.plot(ax = ax1, color = 'blue', lw = 1)
    series2.plot(ax = ax2, color = 'red', lw =1)
    series3.plot(ax = ax3, color = 'green', lw = 1)

    # ax1.axhline(-1.0, color='g', linestyle='--', lw=1)
    ax2.axhline(0, color = 'red', linestyle='--', lw=1)
    ax3.axhline(0, color = 'green', linestyle='--', lw=1)

def five_graphs_twinx(series1, series2, series3, series4, series5):
    
    fig = plt.figure(figsize = (30, 15))
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    ax5 = ax1.twinx()
    
    series1.plot(ax = ax1, color = 'blue', lw = 1)
    series2.plot(ax = ax2, color = 'red', lw =1)
    series3.plot(ax = ax3, color = 'green', lw = 1)
    series4.plot(ax = ax4, color = 'black', lw = 1)
    series5.plot(ax = ax5, color = 'purple', lw = 1)
    

    
def four_graphs(series1, series2, series3, series4):

    fig, ax1 = plt.subplots(1,1, figsize=(25,10), sharex=True)
    
    series1.plot(ax = ax1, color = 'blue', lw=1)
    series2.plot(ax = ax1, color = 'red', lw=1)
    series3.plot(ax = ax1, color = 'green', lw = 1)
    series4.plot(ax = ax1, color = 'purple', lw = 1)


def five_graphs(series1, series2, series3, series4, series5):

    fig, ax1 = plt.subplots(1,1, figsize=(25,10), sharex=True)
    
    series1.plot(ax = ax1, color = 'blue', lw=1)
    series2.plot(ax = ax1, color = 'red', lw=1)
    series3.plot(ax = ax1, color = 'green', lw = 1)
    series4.plot(ax = ax1, color = 'purple', lw = 1)
    series5.plot(ax = ax1, color = 'black', lw = 1)
    
    
# ax3.axhline(-1.0, color='g', linestyle='--', lw=1)
# ax3.axhline(1.0, color='red', linestyle='--', lw=1)
# ax3.axhline(0, color = 'black', linestyle='--', lw=1)

#------------------------------------------------------------------------------------------------------------------------------ 
#def multiple_graphs2(series1, series2):

#     fig, ax1 = plt.subplots(figsize = (30,15))
    
#     series1.plot(ax = ax1, color = 'blue', lw=1)
#     series2.plot(ax = ax1, color = 'red', lw=1)


#------------------------------------------------------------------------------------------------------------------------------ 

def google_data_close_price(symbol_list, start, end): 
    
    #decide which variables and how the function is gonna give you the result.
    import_ = wb.DataReader(symbol_list, data_source='google', start=start, end=end) 
    close = import_['Close']
    df = import_.to_frame().unstack()
#     return df
    return close

def google_data_oc_price(symbol_list, start, end):
    import_ = wb.DataReader(symbol_list, data_source='google', start=start, end=end).drop(['Volume', 'High', 'Low'])
    df_ = import_.to_frame().stack()
    df = df_.unstack(level = 1)
    
    idx = []
    for ts, t in df.index:
        if t == 'Open':
            ts_ = ts.replace(hour=9, minute=30, second=0)
        elif t == 'Close':
            ts_ = ts.replace(hour=15, minute=59, second=59)

        idx.append(ts_)
    
    df.index = pd.DatetimeIndex(idx)
    return df


# Update this fixed package on 9-18-2017:
import fix_yahoo_finance as yf
def get_yahoo_data(symbol_list, start_str, end_str):
    """
    Documentation: start/end_str is of the format of, e.g. "2017-09-15"
    """
    import_ = yf.download(symbol_list, start = start_str, end = end_str)
    df = import_.to_frame().unstack()
    return df

def get_yahoo_data_single(symbol, start_str, end_str):
    """
    Documentation: start/end_str is of the format of, e.g. "2017-09-15"
    """
    import_ = yf.download(symbol, start = start_str, end = end_str)
    return import_
#------------------------------------------------------------------------------------------------------------------------------ 


# Create z-score 

from collections import deque

def data_stream(symbol_list, start=datetime(2000,1,1), end=datetime(2015,12,31), source='google'):
    if isinstance(symbol_list, str): symbol_list = [symbol_list]
        
    data = wb.DataReader(symbol_list, data_source=source, start=start, end=end)
    data = data.to_frame().unstack(level=1).dropna()
    
    for dt, tickers in data.iterrows():
        output = tickers.unstack().to_dict(orient='dict')
        output['timestamp'] = dt
        yield output

def cal_beta(X, y):
	"""can be applied to cal beta betwene two securities
	"""
	X_ = np.asarray(X).reshape(-1,1)
	y_ = np.asarray(y).reshape(-1,1)
	X_ = np.concatenate([np.ones_like(X_), X_], axis=1)
	return np.linalg.pinv(X_.T.dot(X_)).dot(X_.T).dot(y_)[1][0]

	
def cal_residual(x, y, beta):
    return y - beta * x
        
def run_pairs(window, pair=['VXX', 'VXZ'], start=None, end=None, source='google', use_log=False):
    s1, s2 = deque(maxlen=window), deque(maxlen=window)  # data store
    beta, signal, signal2 = [], [], []  # signal store
    residual = deque(maxlen=window)  # store the residual
    ts = []  # time stamp

    for tickers in data_stream(pair, start=start, end=end, source=source):  # iter every market ticks
        ts.append(tickers['timestamp'])  # store this day's timestamp
        s1.append(tickers[pair[0]]['Close'])  # store VXX
        s2.append(tickers[pair[1]]['Close'])  # store VXZ

        # now we have enough data to calculate signal
        if len(s1)>=window and len(s2)>=window:
            # getting the beta
            if use_log:
                X = np.log(s1)
                y = np.log(s2)
            else:
                X, y = s1, s2
            beta_ = cal_beta(X, y)
            beta.append(beta_)

            # calculate residual
            res_ = cal_residual(x=X[-1], y=y[-1], beta=beta_)
            residual.append(res_)

            if len(residual)>=window:
                mean = np.mean(residual)
                std = np.std(residual)
                signal_ = (res_ - mean) / std
                signal2_ = res_
            else:
                signal_ = None
                signal2_ = None

            signal.append(signal_)
            signal2.append(signal2_)
        else:
            beta.append(None)
            signal.append(None)
            signal2.append(None)
		
    return pd.Series(beta, index=ts), pd.Series(signal, index=ts), pd.Series(signal2, index=ts)

# beta, signal, signal2 = run_pairs(10, ['QQQ','SPY'], start = datetime(2016,8,24))


def plot_zscore(signal, signal2): 

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(18,12))
    signal2.plot(ax=ax1)
    signal.plot(ax=ax2)

    ax1.set_title('No Z-score Signal')
    ax2.set_title('Z-score signal')

    plt.axhline(signal.mean(), color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.legend(["Spread z-score - Ryan", 'Mean', '+1', '-1'], loc = 'best')
    


from fredapi import Fred     
from datetime import datetime

USE_API = False

# generate a series of interest rate
def fred_3mon_ir(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('DTB3', observation_start=start, observation_end=end)
    return s

# generate the most updated interest rate
def fred_3mon_ir_today(end = datetime.now()):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('DTB3', observation_end=end)
    return s[-1]/100

def fred_1r_ir_today(end = datetime.now()):
    if USE_API:
        fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
        s = fred.get_series('DGS1', observation_end=end)
        return s[-1]/100
    else:
        return 1.2/100

# fred_data('DTB3', '2012-09-02', '2014-09-05')

def ten_yr_rate(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('DGS10', observation_start=start, observation_end=end)
    return s

def three_fin_rate(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('DCPF3M', observation_start=start, observation_end=end)
    return s

def three_nonfin_rate(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('CPN3M', observation_start=start, observation_end=end)
    return s

def three_rate(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('TB3MS', observation_start=start, observation_end=end)
    return s

def fed_total_asset(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('WALCL', observation_start=start, observation_end=end)
    return s

def high_yield_rate(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('BAMLH0A0HYM2EY', observation_start=start, observation_end=end)
    return s

def spx(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('SP500', observation_start=start, observation_end=end)
    return s

def unemployment(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('UNRATE', observation_start=start, observation_end=end)
    return s

def cpi(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('CPIAUCSL', observation_start=start, observation_end=end)
    return s

def effective_fed_rate(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('FEDFUNDS', observation_start=start, observation_end=end)
    return s

# industrial production index
def ipi(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('INDPRO', observation_start=start, observation_end=end)
    return s

def m2(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('M2', observation_start=start, observation_end=end)
    return s

def ppi(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('PCUOMFGOMFG', observation_start=start, observation_end=end)
    return s

def gdp(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('A191RL1Q225SBEA', observation_start=start, observation_end=end)
    return s

def debt_to_equity(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('TOTDTEUSQ163N', observation_start=start, observation_end=end)
    return s

import urllib.request, urllib.error, urllib.parse

def put_call(start, end):
    urlStr = 'http://www.cboe.com/publish/ScheduledTask/MktData/datahouse/totalpc.csv'
    df_ = pd.read_csv(urllib.request.urlopen(urlStr), header = 2, index_col=0,parse_dates=True)
    df = df_['P/C Ratio']
    return df[start:end]

def ten_yr_3_month_rate_sprd(start, end):
    fred = Fred(api_key='3de60b3b483033f57252b59f497000cf')
    s = fred.get_series('T10Y3M', observation_start=start, observation_end=end)
    return s
















    



