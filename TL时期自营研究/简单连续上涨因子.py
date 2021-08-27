#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:15:22 2019

@author: hurenjie
"""
# 简单连续涨跌因子：连续涨/跌N（3、4、5、6）天之后，下一天涨/跌的概率
#%% 
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import tushare as ts

#token='2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67'
#pro = ts.pro_api(token)
#df = ts.get_hist_data('sh').sort_index()

#%%
def simple_predict_up_or_down(df, N): # N代表我们认为的连续多少天持续的涨或者跌
    df.reset_index(inplace = True)
    df['close_diff'] = df['close'].diff()
    df['signal'] = 0
    for i in df.index:
        if df['close_diff'][i] < 0:
            df['signal'][i] = -1
        elif df['close_diff'][i] > 0:
            df['signal'][i] = 1
    
    df['signal_sum'] = df['signal'].rolling(N).sum()
    df_up_Ndays = df[df['signal_sum'] == N]
    next_day_is_up = 0
    
    df_down_Ndays = df[df['signal_sum'] == -N]
    next_day_is_down = 0
    
    for i in df_up_Ndays.index:
        if i!=df.index[-1]:
            if df['close_diff'][i+1] > 0:
                next_day_is_up += 1
    
    for i in df_down_Ndays.index:
        if i!=df.index[-1]:
            if df['close_diff'][i+1] < 0:
                next_day_is_down += 1
    df.drop(columns = ['close_diff', 'signal', 'signal_sum'], inplace = True)
    df.set_index('date', inplace = True)
    
    prob_up = next_day_is_up/np.shape(df_up_Ndays)[0]
    prob_down = next_day_is_down/np.shape(df_down_Ndays)[0]
    
    return {'连续'+ str(N) +'天上涨后第二天上涨的概率': prob_up, '连续'+ str(N) +'天下跌后第二天下跌的概率':prob_down}

            
            
            
            










































