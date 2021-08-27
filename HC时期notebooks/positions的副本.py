import pandas as pd, numpy as np
np.set_printoptions(suppress=True)# 关掉科学计数法
import glob
import os
import csv
# 一次性merge多个pct_chg
from functools import reduce
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels import regression

# import tushare as ts
import time, urllib
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)


class RiskParity(object):

    def __init__(self, stocks_path, tickers, 
                 date_col, code_col, price_col, 
                 ticker_type, asset_name, draw_pie_graph):

        self.path = stocks_path
        self.tickers = tickers
        self.date_col = date_col
        self.code_col = code_col
        self.price_col = price_col
        self.ticker_type = ticker_type
        self.asset_name = asset_name
        self.draw_pie_graph = draw_pie_graph
        self.ticker_df_list = self.get_date_price_code_return_list()
        self.tgt_returns = self.ticker_df_list
        self.tgt_merged_returns = self.merge_dfs_by_ticker(self.tgt_returns, 
                                                           self.date_col)
        self.wts, self.risk = self.get_smart_weight(self.tgt_merged_returns, 
                                                    method='risk parity', 
                                                    cov_adjusted=False, 
                                                    wts_adjusted=False)
        self.df_wts, self.risk_parity_tickers, self.weights = self.get_df_wts()


    # Get date_col, price_col, code_col, pct_chg_col
    def get_date_price_code_return_list(self):
        # for etf data cols are 'date', 'close', 'code'
        ticker_df_list = []
        for ticker in self.tickers:
            try:
                ticker_df = pd.read_csv(self.path+ticker+".csv")
                ticker_df = ticker_df.sort_values(self.date_col)
                ticker_df = ticker_df[[self.date_col, 
                                       self.price_col, 
                                       self.code_col]]
                ticker_df['pct_chg'] = ticker_df[self.price_col].pct_change()
                ticker_df = ticker_df[[self.date_col, 'pct_chg']].dropna()
                ticker_df.columns = [self.date_col, ticker]
                ticker_df_list.append(ticker_df)
            except Exception as e:
                print(e)
        return ticker_df_list


    def merge_dfs_by_ticker(self, ticker_df_list, date_col):
        merged_all = reduce(lambda left, right: pd.merge(left, right, on=date_col), ticker_df_list)
    #         merged_all = reduce(merge_df_for_reduce, ticker_df_list)
        merged_all.set_index(self.date_col, inplace=True)
        merged_all.dropna(how="all", axis = 1, inplace = True)
        merged_all.fillna(method="ffill", inplace = True)
        return merged_all


    def get_smart_weight(self, pct, method, cov_adjusted, wts_adjusted):
        if cov_adjusted == False:
            #协方差矩阵
            cov_mat = pct.cov()
        else:
            #调整后的半衰协方差矩阵
            cov_mat = pct.iloc[:len(pct)/4].cov()*(1/10.) + pct.iloc[len(pct)/4+1:len(pct)/2].cov()*(2/10.) +\
                pct.iloc[len(pct)/2+1:len(pct)/4*3].cov()*(3/10.) + pct.iloc[len(pct)/4*3+1:].cov()*(4/10.)
        if not isinstance(cov_mat, pd.DataFrame):
            raise ValueError('cov_mat should be pandas DataFrame！')

        omega = np.matrix(cov_mat.values)  # 协方差矩阵

        a, b = np.linalg.eig(np.array(cov_mat)) #a为特征值,b为特征向量
        a = np.matrix(a)
        b = np.matrix(b)
        # 定义目标函数

        def fun1(x):
            tmp = (omega * np.matrix(x).T).A1
            risk = x * tmp/ np.sqrt(np.matrix(x) * omega * np.matrix(x).T).A1[0]
            delta_risk = [sum((i - risk)**2) for i in risk]
            return sum(delta_risk)

        def fun2(x):
            tmp = (b**(-1) * omega * np.matrix(x).T).A1
            risk = (b**(-1)*np.matrix(x).T).A1 * tmp/ np.sqrt(np.matrix(x) * omega * np.matrix(x).T).A1[0]
            delta_risk = [sum((i - risk)**2) for i in risk]
            return sum(delta_risk)

        # 初始值 + 约束条件 
        x0 = np.ones(omega.shape[0]) / omega.shape[0]  
        bnds = tuple((0,None) for x in x0)
        cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
        options={'disp':False, 'maxiter':1000, 'ftol':1e-20}


        #------------------问题出在这里------------------
        if method == 'risk parity':
            res = minimize(fun1, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)        
        elif method == 'pc risk parity':
            res = minimize(fun2, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
        #------------------------------------

        else:
            raise ValueError('method error！！！')

        # 权重调整
        if res['success'] == False:
            # print res['message']
            pass
        wts = pd.Series(index=cov_mat.index, data=res['x'])

        if wts_adjusted == True:
            wts[wts < 0.0001]=0.0
            wts = wts / wts.sum()
        elif wts_adjusted == False:
            wts = wts / wts.sum()
        else:
            raise ValueError('wts_adjusted should be True/False！')

        risk = pd.Series(wts * (omega * np.matrix(wts).T).A1 / np.sqrt(np.matrix(wts) * omega * np.matrix(wts).T).A1[0],index = cov_mat.index)
        risk[risk<0.0] = 0.0
        return wts,risk


    def get_df_wts(self):
        df_wts = pd.DataFrame(self.wts)
        df_wts.reset_index(inplace = True)
        df_wts.columns = [self.asset_name, 'weight']
        risk_parity_tickers = list(df_wts[self.asset_name])
        weights = list(df_wts['weight'])
        if self.draw_pie_graph == True:
            # 保证圆形
            plt.figure(1, figsize = (7, 7))
            plt.axes(aspect=1)
            plt.pie(x=weights, labels=risk_parity_tickers, autopct='%3.1f %%')
            plt.title("Risk-Parity Allocation", fontsize = 15)
            plt.show()
            
        else:
            pass
        return df_wts, risk_parity_tickers, weights


class LIVE_TRADING_POSITIONS():

	def target_HK_shares(ticker, last_price, target_pos_pct, available_dollar):

	#     last_price = 13.3
	#     target_pos_pct = 0.5
	#     stop_loss = 27.9518.2
	#     available_dollar = 100000
	    target_pos_dollar = target_pos_pct*available_dollar
	    USD_HKD_exchange_rate = 7.5
	    target_pos_HKD = target_pos_dollar*USD_HKD_exchange_rate
	    target_HK_shares = round(target_pos_HKD/last_price)
	    print("Ticker: {}".format(ticker))
	    print("Target_Shares: ", target_HK_shares)
	    return target_HK_shares, target_pos_HKD


	def HK_expected_loss(target_HK_shares, target_pos_HKD, ex_price_w_cost, stop_loss_price, portfolio_value):
	    USD_HKD_exchange_rate = 7.5
	    single_trade_expected_loss_pct = stop_loss_price/ex_price_w_cost-1
	    single_trade_expected_loss_USD = single_trade_expected_loss_pct*target_pos_HKD/USD_HKD_exchange_rate
	    port_expected_loss_pct = single_trade_expected_loss_USD/portfolio_value
	    
	    print("单笔交易止损损失%: ", round(single_trade_expected_loss_pct*100, 3), "%")
	    print("账户产生的止损损失%: ", round(port_expected_loss_pct*100, 3), "%")
	    print("账户止损损失美元价值: ", "$",round(single_trade_expected_loss_USD, 2))
	    print("Breakeven Price w commissions:", round(ex_price_w_cost*1.0015, 2))
	    print("1% Profits w commissions:", round(ex_price_w_cost*1.01, 2))
	    print("2% Profits w commissions:", round(ex_price_w_cost*1.02, 2))


# class CORRELATION():

# def stock_list_to_calculate_corr(all_csvs, date_col, price_col, ticker_col):
# 	len_ = 0
# 	for fname in glob.glob(all_csvs)[:]:
# 		try:
# 			stock = pd.read_csv(fname)
# 			stock = stock.sort_values(date_col)
# 			ticker = stock[ticker_col].values[-1]
# 			print(ticker)
# 			stock = stock[[date_col,price_col]]
# 			stock['pct_chg'] = stock[price_col].pct_change()
# 			stock.columns = [date_col, price_col, ticker]
# 			stock = stock[[date_col, ticker]].dropna()
# 			stock[date_col] = pd.to_datetime(stock[date_col])
# 			#     stock.set_index('date', inplace=True)
# 			stock_list.append(stock)
# 			print ("Length of {}: {}".format(ticker, len(stock)))
# 			len_ = len_+len(stock)
# 			print ("Total length:{}".format(len_))
# 			print ("===========")
# 		except:
# 			print("Error on: {}".format(fname))
# 	return stock_list

def merge_df(df1, df2):
	df1.sort_values("Date", inplace = True)
	merged = df1.merge(df2, on = "Date", how = 'outer')
	merged.sort_values("Date", inplace = True)
	return merged

def plot_heatmap_corrleation(data, stock_list):
    plt.figure(figsize=(10, 10))
    sns.heatmap(data[stock_list].corr(), annot=True, cmap = 'Blues', vmax = 1.0, vmin = -1.0)
    plt.xlabel('stocks')
    plt.ylabel('stocks')
    plt.show()


def stock_list_to_calculate_corr(all_csvs, date_col, price_col, ticker_col):
	len_ = 0
	for fname in glob.glob(all_csvs)[:]:
		try:
			stock = pd.read_csv(fname)
			stock = stock.sort_values(date_col)
			ticker = stock[ticker_col].values[-1]
			print(ticker)
			stock = stock[[date_col,price_col]]
			stock['pct_chg'] = stock[price_col].pct_change()
			stock.columns = [date_col, price_col, ticker]
			stock = stock[[date_col, ticker]].dropna()
			stock[date_col] = pd.to_datetime(stock[date_col])
			#     stock.set_index('date', inplace=True)
			stock_list.append(stock)
			print ("Length of {}: {}".format(ticker, len(stock)))
			len_ = len_+len(stock)
			print ("Total length:{}".format(len_))
			print ("===========")
		except Exception as e:
			print("Error on: {}:{}".format(fname, e))
	return stock_list