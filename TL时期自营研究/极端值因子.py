#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:31:49 2019

@author: hurenjie
"""

# 极端值因子：当大盘(或者其他股票)日涨幅/跌幅绝对值超过2%，下一天上涨或者下跌的概率/历史上的次数有多少
#%%
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import tushare as ts

token='2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67'
pro = ts.pro_api(token)
df = ts.get_hist_data('sh').sort_index()

#%%
def extreme_value(df, threshold): # threshold 在此应该定义为一个百分数,代表我们认为的涨跌幅度阈值
    df['close_pct_ch'] = df['close'].pct_change().shift(-1)
    df['close_diff'] = df['close'].diff().shift(-2)
    df_up = df[df['close_diff'] > 0]
    df_down = df[df['close_diff'] < 0]
    
    up = df_up[df_up['close_pct_ch'] > threshold]
    down = df_down[df_down['close_pct_ch'] < -threshold]
    
    prob_up = np.shape(up)[0]/np.shape(df_up)[0]
    prob_down = np.shape(down)[0]/np.shape(df_down)[0]
        
    if df['close_diff'].iloc[-3] > 0:
        result = prob_up # 当天价格上涨，利用极端值因子估计法估计出第二天继续上涨的概率
        print('当天价格上涨,该函数计算的是第二天价格继续上涨的概率')
    else:
        result = prob_down # 当天价格下跌，利用极端值因子估计法估计出第二天继续下跌的概率
        print('当天价格下跌,该函数计算的事第二天价格继续下跌的概率')
    df.drop(columns = ['close_pct_ch', 'close_diff'], inplace = True)
    return result
    

        


































