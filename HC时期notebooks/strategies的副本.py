import pandas as pd, numpy as np
np.set_printoptions(suppress=True)# 关掉科学计数法
import glob
import os
import csv
# 一次性merge多个pct_chg
import yfinance as yf
from functools import reduce
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels import regression
import time, urllib

"""
MACD 系列
"""

# 定义每个筛选模型
class PRICE_VOL_INDICATORS():
	# 1. MACD
	def MACD(df, price_col, n_fast, n_slow, n_ema): # n_fast = 12, n_slow = 26
	    """
	    http://stockcharts.com/docs/doku.php?id=scans:indicators
	    MACD, MACD Signal and MACD difference, rationale CHECKED, code CHECKED, updated
	    # Conventional look-back window for calculating MACDsign is 9
	    """
	    EMAfast = df[price_col].ewm(span = n_fast, min_periods = n_fast - 1).mean()
	    EMAslow = df[price_col].ewm(span = n_slow, min_periods = n_slow - 1).mean()
	    diff = pd.Series(EMAfast - EMAslow)
	    dea = diff.ewm(span = n_ema, min_periods = n_ema-1).mean()
	    macd = (pd.Series(diff - dea))*2
	    df["DIFF"] = diff
	    df["DEA"] = dea
	    df["MACD"] = macd
	    return df


class SIGNALS():
	# 在MACD模型下的开平仓算法
	def macd_updown_signals(df, signal_spread_col, date_col, ticker_col):
	    ticker = df[ticker_col].values[-1]
	    last_date = df[date_col].values[-1]
	    df.dropna(inplace = True)
	    df.reset_index(inplace = True)
	    del df['index']
	    listLongShort = []
	    macd = df[signal_spread_col].values
	    
	    for i in range(1, len(df)):
	        last_date = df[date_col][i]
	        
	        if macd[i]>macd[i-1] and macd[i-1]<macd[i-2]:
	            
	            listLongShort.append("BUY")
	        #                          # The other way around
	        elif macd[i]<macd[i-1] and macd[i-1]>macd[i-2]:
	            listLongShort.append("SELL")
	        #                          # Do nothing if not satisfied
	        elif macd[i]<macd[i-1]:
	            listLongShort.append("HOLD SHORT")
	            
	        elif macd[i]>macd[i-1]:
	            listLongShort.append("HOLD LONG")        
	            
	    return ticker, last_date, listLongShort[-1]
	#     df['Advice'] = listLongShort
	    # The advice column means "Buy/Sell/Hold" at the end of this day or
	    #  at the beginning of the next day, since the market will be closed


	def macd_cross_signals(df, signal_spread_col, date_col, ticker_col):
	    ticker = df[ticker_col].values[-1]
	    last_date = df[date_col].values[-1]
	    listLongShort = []
	    listDate = []
	    macd = df[signal_spread_col].values
	    
	    for i in range(1, len(df)):
	        last_date = df[date_col][i]
	        #                          # If the MACD crosses the signal line upward
	        if macd[i] >0 and macd[i - 1] <0:
	            listLongShort.append("BUY")
	            listDate.append(last_date)
	        #                          # The other way around
	        elif macd[i] < 0 and macd[i - 1] >0:
	            listLongShort.append("SELL")
	            listDate.append(last_date)
	        #                          # Do nothing if not crossed
	        else: # 还要改，再增加点条件
	            listLongShort.append("HOLD")
	            listDate.append(last_date)
	#     print("Ticker: ", ticker)
	#     print("Last Date", listDate[-1])
	#     print("Last Signal", listLongShort[-1])
	    return ticker, last_date, listLongShort[-1]
	#     df['Advice'] = listLongShort
	    # The advice column means "Buy/Sell/Hold" at the end of this day or
	    #  at the beginning of the next day, since the market will be closed

	def macd_updown_ma_slope(df, signal_spread_col, date_col, ticker_col, price_col):
		ticker = df[ticker_col].values[-1]
		last_date = df[date_col].values[-1]
		listLongShort = []
		# listDate = []
		macd = df[signal_spread_col].values
		# Calculate MA5:
		ma5 = pd.Series(df[price_col]).rolling(window=5).mean()
		# Calculate MA10:
		ma10 = pd.Series(df[price_col]).rolling(window=10).mean()

		for i in range(1, len(df)):
			current_date = df[date_col][i]
			last_close = df[price_col][i]
			last_ma5 = ma5[i]
			last_ma10 = ma10[i]
			sec_last_ma10 = ma10[i-1]

			if macd[i]>macd[i-1] and macd[i-1]<macd[i-2] and last_close>last_ma5 and last_ma10>sec_last_ma10:
	            
				listLongShort.append("买入")
	        #                          # The other way around
			elif macd[i]<macd[i-1] and macd[i-1]>macd[i-2] and last_close<last_ma5:
				listLongShort.append("卖出")
	        #                          # Do nothing if not satisfied
			elif macd[i]<macd[i-1] and last_close<last_ma5:
				listLongShort.append("空头持有")
	            
			elif macd[i]>macd[i-1] and last_close>last_ma5:
				listLongShort.append("多头持有")   
			else:
				listLongShort.append("其他状态")

		return ticker, last_date, listLongShort[-1]


	def cross_MA(df, date_col, ticker_col, price_col, MA_para):
		ticker = df[ticker_col].values[-1]
		last_date = df[date_col].values[-1]
		listLongShort = []		
		ma = pd.Series(df[price_col]).rolling(window=MA_para).mean()

		for i in range(1, len(df)):
			last_close = df[price_col][i]
			if last_close > ma[i] and last_close < ma[i-1]:
				listLongShort.append("上穿MA{}，买入")
			elif last_close < ma[i] and last_close > ma[i-1]:
				listLongShort.append("下穿MA{}，卖出")
			else:
				listLongShort.append("价格压在均线上，无操作")
		return ticker, last_date, listLongShort[-1]





class GENERATE_SIGNALS():

	def generate_signals_macd_updown(data_list, model_name, date_col, price_col, ticker_col, model_freq):
	    stock_list = []
	    date_list = []
	    signal_list = []
	    signal_df = pd.DataFrame()
	    signal_count = 1

	    for data in data_list:
	        try:
	            TA_df = PRICE_VOL_INDICATORS.MACD(data, "Adj Close", 12, 26, 9)
	            ticker, last_date, last_signal = SIGNALS.macd_updown_signals(TA_df, "MACD", date_col, ticker_col)
	            stock_list.append(ticker)
	            date_list.append(last_date)
	            signal_list.append(last_signal)
	            print ("Signals({}) Prepared for No.{} : {}".format(model_freq, signal_count, ticker))
	        except Exception as e:
	            print(e)
	        signal_count+=1

	    signal_df['Ticker'] = stock_list
	    signal_df['Last_Date'] = date_list
	    signal_df['Signal'] = signal_list
	    signal_df['model_name'] = model_name
	    signal_df['model_freq'] = model_freq
	    print("-------------------------")
	    print("All Done!")
	    return signal_df

	def generate_signals_macd_updown_ma_slope(data_list, model_name, date_col, price_col, ticker_col, model_freq):
	    stock_list = []
	    date_list = []
	    signal_list = []
	    signal_df = pd.DataFrame()
	    signal_count = 1

	    for data in data_list:
	        try:
	            TA_df = PRICE_VOL_INDICATORS.MACD(data, "Adj Close", 12, 26, 9)
	            ticker, last_date, last_signal = SIGNALS.macd_updown_ma_slope(TA_df, "MACD", date_col, ticker_col, price_col)
	            stock_list.append(ticker)
	            date_list.append(last_date)
	            signal_list.append(last_signal)
	            print ("Signals({}) Prepared for No.{} : {}".format(model_freq, signal_count, ticker))
	        except Exception as e:
	            print(e)
	        signal_count+=1

	    signal_df['Ticker'] = stock_list
	    signal_df['Last_Date'] = date_list
	    signal_df['Signal'] = signal_list
	    signal_df['model_name'] = model_name
	    signal_df['model_freq'] = model_freq
	    print("-------------------------")
	    print("All Done!")
	    return signal_df
# class SIGNAL_DF_GENERATION():
