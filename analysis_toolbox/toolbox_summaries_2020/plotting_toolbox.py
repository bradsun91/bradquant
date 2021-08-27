import psycopg2
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyecharts
import urllib3,time,csv,datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import matplotlib.dates as mpd
import plotly.plotly as py
import plotly.offline as py_offline
import plotly.graph_objs as go
from IPython.display import clear_output

%matplotlib inline


# ===================================================================================================================
# 4-9-2019 updated
# 计算程序运行时间
import datetime
starttime = datetime.datetime.now()
print ("Executing...")
endtime = datetime.datetime.now()
duration = (endtime - starttime).seconds
print ("Execution takes {} seconds".format(duration))


# ===================================================================================================================
# 转变不同周期、用Plotly实时plot，并且以蜡烛图的形式，加上交易信号
# 转变数据周期：T/5T/15T/30T/H/2H/4H/D/W/M
def resample(df, period):
    convrted_df = df.resample(period).last()
    convrted_df['open'] = df['open'].resample(period).first()
    convrted_df['high'] = df['high'].resample(period).max()
    convrted_df['low'] = df['low'].resample(period).min()
    convrted_df['close'] = df['close'].resample(period).last()
    convrted_df['volume'] = df['volume'].resample(period).sum()
    # Keep rows with at least 5 non-NaN values
    convrted_df.dropna(thresh=5, inplace=True)
    convrted_df.index = convrted_df['time']
    convrted_df['time'] = pd.DatetimeIndex(convrted_df['time'])
    return convrted_df

# 为能够plot蜡烛图做timestamps处理的准备：
def cnvrt_date(convrted_df):
    cnvrted_date_df = convrted_df.copy()
    cnvrted_date_df['date'] = mpd.date2num(cnvrted_date_df['time'].dt.to_pydatetime())
    return cnvrted_date_df

# 计算两个资产的相对差累计值：
def two_assets_tmsum(cnvrted_date_df1, cnvrted_date_df2, N4):
    cnvrted_date_df1['close_shifted'] = cnvrted_date_df1['close'].shift(1)
    cnvrted_date_df2['close_shifted'] = cnvrted_date_df2['close'].shift(1)
    T1 = cnvrted_date_df1['close'].diff()/cnvrted_date_df1['close_shifted']
    T2 = cnvrted_date_df2['close'].diff()/cnvrted_date_df2['close_shifted']
    TM = T1*(1-N4/100) - T2*(N4/100)
    tmsum_sr = TM.cumsum()
    return tmsum_sr
    
# 计算相对差的移动平均快慢线：
def MAs_of_tmsum(tmsum_sr, N1, N2):
    MA1 = tmsum_sr.ewm(span= N1).mean() #快线
    MA2 = tmsum_sr.ewm(span= N2).mean() #慢线   
    return MA1, MA2
    
# Plot TMSUM图线：
def plot_tmsum(tmsum_sr):
    py_offline.init_notebook_mode()
    tmsum_df = pd.DataFrame(tmsum_sr, columns=['tmsum'])
    tmsum_df = go.Scatter(x = tmsum_df.index,
                          y = tmsum_df['tmsum'])
    data = [tmsum_df]
    return py_offline.iplot(data, filename='TMSUM')
    
def plot_tmsum_MAs(MA1, MA2):
    # Here I didn't use offline's version of plotly, going forward  
    # will need to be consistent when moving to pycharm for plotting
    ma1_df = pd.DataFrame(MA1, columns=['MA1'], index = MA1.index)
    ma2_df = pd.DataFrame(MA2, columns=['MA2'], index = MA2.index)
    trace1 = go.Scatter(x = ma1_df.index, 
              y = ma1_df['MA1'])
    trace2 = go.Scatter(x = ma2_df.index, 
              y = ma2_df['MA2'])
    data = [trace1, trace2]
    fig = go.Figure(data=data, 
    #                     layout=layout
                   )
    return py.iplot(fig, filename='plot_tmsum_MAs')
    
    
# Plot 资产实时图
def plot_candlestick(cnvrted_date_df):
    py_offline.init_notebook_mode()
    candle_df = go.Candlestick(x = cnvrted_date_df.index,
                               open = cnvrted_date_df['open'],
                               high = cnvrted_date_df['high'],
                               low  = cnvrted_date_df['low'],
                               close = cnvrted_date_df['close'])
    data = [candle_df]
    return py_offline.iplot(data, filename='Candle Stick', image_width=2, image_height=4)

# Build a function that plots charts of two moving averages with their crossover trading signals 
# Here we want to make sure that both MA1_sr and MA2_sr are: series with timestanmps as their indexes

def MA_crossover_plot_signals(MA1_sr, MA2_sr):
    # 构建一个由 MA1和 MA2构成的dataframe：
    ma_signal_df = pd.DataFrame(MA1_sr, columns=['MA1'], index= MA1_sr.index)
    ma_signal_df['MA2'] = MA2_sr
    # 用两者的差表示 MA1在位置上高于还是低于MA2，负值说明低于，正值说明高于：
    ma_signal_df['MA1_mns_MA2'] = ma_signal_df['MA1'] - ma_signal_df['MA2']
    # 将正负值同义转换为二元的 1或者 -1便于观察和处理
    ma_signal_df['signs'] = ma_signal_df['MA1_mns_MA2']*abs(1/(ma_signal_df['MA1_mns_MA2']))
    # 用当前值和前一个周期的值决定当前状态是金叉信号还是死叉信号
    ma_signal_df['pre_signs'] = ma_signal_df['signs'].shift(1)
    ma_signal_df['signals'] = ma_signal_df['signs'] - ma_signal_df['pre_signs']
    ma_signal_df['signals_alert'] = ma_signal_df['signals'].apply(lambda x: "金叉" if x==2 else "死叉" if x==-2 else "无信号")
    # 将所有出现信号的rows挑出来建立一个字典：
    ma_df_with_signals = ma_signal_df[ma_signal_df['signals_alert'] != "无信号"]['signals_alert']
    signal_dict = dict(ma_df_with_signals)
    # Add each and all signal information(dictionary format) to annotations(list format) 
    # so that it can be put into the go.Layout() function.
    annotations = []
    each_dict = {}
    for i, k in enumerate(signal_dict):
        each_dict['x'] = k
        each_dict['y'] = ma_signal_df['MA1'].loc[k]
        each_dict['text']=ma_signal_df['signals_alert'].loc[k]
        each_dict['showarrow']=True
        each_dict['arrowhead']=7
        each_dict['ax']=0
        each_dict['ay']=-40

        annotations.append(each_dict.copy())
    # 用plotly进行绘图，包括了之前处理好的annotations，作为显示信号的功能
    ma1_df = pd.DataFrame(tmsum_ma1, columns=['MA1'], index = tmsum_ma1.index)
    ma2_df = pd.DataFrame(tmsum_ma2, columns=['MA2'], index = tmsum_ma2.index)
    trace1 = go.Scatter(x = ma1_df.index, 
              y = ma1_df['MA1'])
    trace2 = go.Scatter(x = ma2_df.index, 
              y = ma2_df['MA2'])
    layout = go.Layout(
        showlegend=False,
        annotations=annotations
    )
    data = [trace1, trace2]
    fig = go.Figure(data=data, 
                    layout=layout)
    return py.iplot(fig, filename='plot_MAs_with_signals')




# plot using iplot function to generate a more interactive plotting interface.
# df should have its
def interface_plt_1(df, col, main_title, y_label):
    plot_1 = go.Scatter(x=df.index, y=df[col], name = col)

    data = [plot_1]
    layout = go.Layout(
       title=main_title,
       yaxis=dict(
           title=y_label
       )
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='styling-names.html')
    
    
# plot using iplot function to generate a more interactive plotting interface.

def interface_plt_2(df, col_1, col_2, main_title, y_label):
    plot_1 = go.Scatter(x=df.index, y=df[col_1], name = col_1)
    plot_2 = go.Scatter(x=df.index, y=df[col_2], name = col_2)

    data = [plot_1, plot_2]
    layout = go.Layout(
       title = main_title,
       yaxis=dict(
           title=y_label
       )
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='styling-names.html')
    
    
    
# plot using iplot function to generate a more interactive plotting interface.

def interface_plt_3(df, col_1, col_2, col_3, main_title, y_label):
    plot_1 = go.Scatter(x=unempt_df.index, y=unempt_df[col_1], name = col_1)
    plot_2 = go.Scatter(x=unempt_df.index, y=unempt_df[col_2], name = col_2)
    plot_3 = go.Scatter(x=unempt_df.index, y=risk_level_score_diff, name = col_3)


    data = [plot_1, plot_2, plot_3]
    layout = go.Layout(
       title = main_title,
       yaxis=dict(
           title=y_label
       )
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='styling-names.html')









