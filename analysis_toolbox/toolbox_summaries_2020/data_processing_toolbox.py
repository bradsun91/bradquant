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
# 11-08-2020
# 简单高效地运用pivottable来转换数据格式，用以进行corr analysis或者根据date来plot returns

fund_nav_df
"""
fund_nav_df长什么样子：

	ts_code	ann_date	end_date	unit_nav	accum_nav	accum_div	net_asset	total_netasset	adj_nav	update_flag
0	511990.SH	20201107	20201106	1.0000	None	None	NaN	NaN	12735.5844	1
1	511990.SH	20201106	20201105	1.0000	None	None	NaN	NaN	12734.8939	1
2	511990.SH	20201105	20201104	1.0000	None	None	NaN	NaN	12734.1857	1
3	511990.SH	20201104	20201103	1.0000	None	None	NaN	NaN	12733.4646	1
4	511990.SH	20201103	20201102	1.0000	None	None	NaN	NaN	12732.7244	1
...	...	...	...	...	...	...	...	...	...	...
168	515030.SH	20200304	20200303	0.9819	0.9819	None	NaN	NaN	0.9819	0
169	515030.SH	20200229	20200228	0.9494	0.9494	None	NaN	NaN	0.9494	0
170	515030.SH	20200228	20200226	0.9944	0.9944	None	1.070217e+10	1.07022e+10	0.9944	0
171	515030.SH	20200222	20200221	1.0007	1.0007	None	NaN	NaN	1.0007	1
172	515030.SH	20200221	20200220	1.0000	1	None	1.076288e+10	1.07629e+10	1.0000	1

其特征就是，所有的tickers信息都被上下concat在一起，而不是根据date的columns被merge到一起，因此很不方便进行相关性分析或者plot其序列在同一张图里

解决方法：

直接用Pivot table来转换数据格式

"""

transformed_df = fund_nav_df_test_2.pivot_table(index='ann_date', columns=['ts_code'], values='adj_nav')

"""
transformed_df 长什么样子？
ts_code	511990.SH	515030.SH
ann_date		
20121229	10001.1902	NaN
20130101	10004.1876	NaN
20130105	10008.9664	NaN
20130112	10010.6887	NaN
20130119	10011.8450	NaN
...	...	...
20201103	12732.3685	1.2921
20201104	12733.4646	1.2889
20201105	12734.1857	1.3169
20201106	12734.8939	1.3890
20201107	12735.5844	1.3823

"""


# ===================================================================================================================
# 05-17-2019 
# 分割一栏into两栏split one column into two columns
# new data frame with split value columns 
new = data["Name"].str.split(" ", n = 1, expand = True) 
  
# making separate first name column from new data frame 
data["First Name"]= new[0] 
  
# making separate last name column from new data frame 
data["Last Name"]= new[1] 


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
# 4-2-2019 updated
# Python - 利用zip函数将两个列表(list)组成字典(dict)
keys = ['a', 'b', 'c']
values = [1, 2, 3]
dictionary = dict(zip(keys, values))
print (dictionary)
# 输出:
# {'a': 1, 'c': 3, 'b': 2}

# ===================================================================================================================
# 3-24-2019
# download data from yahoo finance and plot correlation heatmaps on downloaded stocks' returns
def corr_heatmaps_yf(symbol_list, price_col, start_str, end_str, corr_thresh):
    """
    Documentation: 
    1. start/end_str is of the format of, e.g. "2017-09-15"
    2. corr_thresh ranges from -1 to 1
    
    """
    df = yf.download(symbol_list, start = start_str, end = end_str)
    stacked = df.stack().reset_index()
    stacked.columns = ['date', 'tickers', 'close', 'close2', 'high', 'low', 'open','volume']
    stacked_ = stacked[['date', 'tickers', 'open', 'high', 'low', 'close', 'volume']]
    stacked_col = stacked_[['date', 'tickers', price_col]]
    stacked_col_pvt = pd.pivot_table(stacked_col, values = price_col, index = 'date', columns = 'tickers')
    stacked_col_pvt_pctchg = stacked_col_pvt.pct_change()
    fig, ax = plt.subplots(figsize = (12, 8))
    sns.heatmap(stacked_col_pvt_pctchg.corr()[(stacked_col_pvt_pctchg.corr()>corr_thresh)|(stacked_col_pvt_pctchg.corr()<-corr_thresh)], ax = ax, cmap = 'Blues', vmax = 1.0, vmin = -1.0, annot=True)
    plt.xlabel('stocks', fontsize = 15)
    plt.ylabel('stocks', fontsize = 15)
    plt.xticks(fontsize = 17)
    plt.yticks(fontsize = 17)
    return stacked_col_pvt_pctchg


# ===================================================================================================================
# 3-7-2019 updated
# 最终目标：plot correlation heatmaps
# 重要新知识点：利用reduce一次性merge多个csv dataframes
# 下载上证50个个股数据：
import tushare as ts
from functools import reduce #重要知识点：reduce

sz50 = ts.get_sz50s()
sz50_code_list = list(sz50['code'])
folder_all = "C:/Users/workspace/brad_public_workspace_on_win/non_code_files_brad_public_workspace_on_win/brad_public_workspace_on_win_non_code_files/SH_tongliang/data/SZ50_daily_data/1998_2019_all_51/"
n = 0
for code in sz50_code_list[48:]:
    cons = ts.get_apis()
    df = ts.bar(code, conn=cons, freq='D', start_date='1998-01-01', end_date='2019-03-06')
    df.reset_index(inplace=True)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'vol', 'amount']]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount']
    # 看有多少可以被下载下来的文件：
    len_ = len(df)
    n = n+1
    df.to_csv(folder_all+code+"_1998_2019.csv", index = False)
    print ("No.{}, {}的数据量：{}，起始时间: {}".format(n, code, len_, df['date'].values[-1]))


stock_list = []
len_ = 0
for fname in glob.glob(all_csvs)[:]:
#     print (fname)
    stock = pd.read_csv(fname)
    stock = stock.sort_values('date')
    stock = stock[['date','close']]
    stock['pct_chg'] = stock['close'].pct_change()
    ticker = fname[-20:-14]
    stock.columns = ['date', 'close', ticker]
    stock = stock[['date', ticker]].dropna()
    stock['date'] = pd.to_datetime(stock['date'])
#     stock.set_index('date', inplace=True)
    stock_list.append(stock)
    # print ("Length of {}: {}".format(ticker, len(stock)))
    # print (stock.head(20))
    # len_ = len_+len(stock)
    # print ("Total length:{}".format(len_))
    # print ("===========")

# 先位置后使用reduce铺路：创造一个merge的函数：
def merge_df(df1, df2):
    df1.sort_values('date', inplace = True)
    merged = df1.merge(df2, on = 'date', how = 'outer')
    merged.sort_values('date', inplace = True)
    return merged

# 重要知识点：reduce
merged_all = reduce(merge_df, stock_list)
merged_all.set_index('date', inplace=True)

# 最后一步：plot heatmap:
fig, ax = plt.subplots(figsize = (40, 30))
sns.heatmap(merged_all.corr()[abs(merged_all.corr())>-2], ax = ax, cmap = 'Blues', vmax = 1.0, vmin = -1.0, annot=True)
plt.xlabel('stocks', fontsize = 15)
plt.ylabel('stocks', fontsize = 15)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17);


# ===================================================================================================================
# 2-25-2019 udpated
print("{:.2f}".format(number)) # 两位小数


# ===================================================================================================================
# 2-21-2019 udpated
# 生成信号代码：小于0为-1；大于0为1；等于0为0：
signal_df['signal'] = signal_df['signal'].apply(lambda x: 1 if x>0 else -1 if x < 0 else 0)

# 根据信号dataframe，生成回测收益结果：

def calc_single_performance(signal_df, price_col):
    """
    1. date是经过函数pd.to_datetime()处理过后的index；
    2. signal的值为0或者-1或者1，分别代表不持仓、空头信号和多头信号；
    3. price可以是close, open等需要当作计算收益基础的价格数据；
    
    signal_df的格式示例如下：
    =============================
                  price   signal
       date
    2017-07-28     256.3    -1
    2017-07-29     259.5     0
    =============================
    """
    signal_df['price_diff'] = signal_df[price_col].diff()
    signal_df['forward_signal'] = signal_df['signal'].shift(1)
    signal_df['returns'] = signal_df['forward_signal']*signal_df['price_diff']
    signal_df['cum_returns'] = signal_df['returns'].cumsum()
    return signal_df



# ===================================================================================================================
# 12_13_2018: 使用tushare
import tushare as ts
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

ts.set_token("2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67")
pro = ts.pro_api()




# ===================================================================================================================
# 12_3_2018: 数据全部标准化到百位，除了BTC

import pandas as pd, numpy as np
from datetime import datetime
import psycopg2

def all_assts_from_sql(asst1, asst2, asst3, asst4, asst5, asst6, asst7, asst8, sql_limit_num, location, till_date):
    conn = psycopg2.connect(database="bitmexdata", user="postgres", password="tongKen123", host="128.199.97.202",
                            port="5432")
    asset1 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst1,
                                                                                                    sql_limit_num)
    asset2 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst2,
                                                                                                    sql_limit_num)
    asset3 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst3,
                                                                                                    sql_limit_num)
    asset4 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst4,
                                                                                                    sql_limit_num)
    asset5 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst5,
                                                                                                    sql_limit_num)
    asset6 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst6,
                                                                                                    sql_limit_num)
    asset7 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst7,
                                                                                                    sql_limit_num)
    asset8 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst8,
                                                                                                    sql_limit_num)
    df1 = pd.read_sql(asset1, con=conn)
    df2 = pd.read_sql(asset2, con=conn)
    df3 = pd.read_sql(asset3, con=conn)
    df4 = pd.read_sql(asset4, con=conn)
    df5 = pd.read_sql(asset5, con=conn)
    df6 = pd.read_sql(asset6, con=conn)
    df7 = pd.read_sql(asset7, con=conn)
    df8 = pd.read_sql(asset8, con=conn)
    conn.close()
    #     return df1, df2, df3, df4, df5, df6, df7, df8
    df1.to_csv(location + "{}".format(till_date) + asst1 + ".csv", index=False)
    df2.to_csv(location + "{}".format(till_date) + asst2 + ".csv", index=False)
    df3.to_csv(location + "{}".format(till_date) + asst3 + ".csv", index=False)
    df4.to_csv(location + "{}".format(till_date) + asst4 + ".csv", index=False)
    df5.to_csv(location + "{}".format(till_date) + asst5 + ".csv", index=False)
    df6.to_csv(location + "{}".format(till_date) + asst6 + ".csv", index=False)
    df7.to_csv(location + "{}".format(till_date) + asst7 + ".csv", index=False)
    df8.to_csv(location + "{}".format(till_date) + asst8 + ".csv", index=False)


# standardize and reset the price:
# file_suffix example: 'z18_1d.csv'
# added_note example: 'u18z18乘'
# till_date example: '11_10_'
def reset_price(location, till_date, file_suffix, added_note):
    symlist = ['ada', 'bch', 'eth', 'eos', 'trx', 'xrp', 'ltc']

    #     for s in symlist:
    #         if s == 'ada':
    #             c = 10000000
    #         elif s == 'bch':
    #             c = 1000
    #         elif s == 'eos':
    #             c = 100000
    #         elif s == 'eth' or s == 'ltc':
    #             c = 10000
    #         elif s == 'trx':
    #             c = 100000000
    #         elif s == 'xrp':
    #             c = 10000000
    for s in symlist:
        if s == 'ada':
            c = 100000000  #
        elif s == 'bch':
            c = 10000  #
        elif s == 'eos':
            c = 1000000  #
        elif s == 'eth':
            c = 10000
        elif s == 'ltc':
            c = 100000  #
        elif s == 'trx':
            c = 100000000
        elif s == 'xrp':
            c = 10000000

        # b = pd.read_csv(location + till_date + s + file_suffix,header=None)
        b = pd.read_csv(location + "{}".format(till_date) + s + file_suffix + ".csv")
        b.iloc[:, [1, 2, 3, 4]] = b.iloc[:, [1, 2, 3, 4]].astype(float)
        b.iloc[:, [1, 2, 3, 4]] = (b.iloc[:, [1, 2, 3, 4]]) * c
        b.to_csv(location + 'res_' + till_date + s + file_suffix + '.csv',
                 sep=',', header=False, index=False, float_format='%.4f')


location = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/extracting_and_transforming_data/"
till_date = "12_03_og_all_hundreds_"
frequency = "_1d"
file_suffix = 'u18z18' + frequency
added_note = 'u18z18'

all_assts_from_sql("ada" + file_suffix, "bch" + file_suffix, "eos" + file_suffix,
                   "eth" + file_suffix, "ltc" + file_suffix, "trx" + file_suffix,
                   "xbtusd" + frequency, "xrp" + file_suffix, 1000,
                   "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/extracting_and_transforming_data/",
                   till_date)

reset_price(location, till_date, file_suffix, added_note)

btc_df = pd.read_csv(location + till_date + "xbtusd" + frequency + ".csv", engine="python", header=None)
btc_df = btc_df.iloc[1:, :]
btc_df.to_csv(location + "res_" + till_date + "xbtusd" + frequency + ".csv",
              sep=',', header=False, index=False, float_format='%.4f')


# ===================================================================================================================
# 11_24_1min 数据变成四个小时：

location = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/11_24_bitmex分钟线_multiplied/bitmex分钟线1124/"

def chg_col(df):
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    return df

def time_to_timestamp(timestr):
    """
    时间字符串转unix时间戳
    :param str: 时间字符串
    :return: unix时间戳，str类型
    """
    dt = datetime.strptime(str(timestr), '%Y-%m-%d %H:%M:%S')
    timestamp = time.mktime(dt.timetuple())
    return str(int(timestamp))

def myresample(df, period, min):
    convrted_df = df.resample(period).last()
    convrted_df['open'] = df['open'].resample(period).first()
    convrted_df['high'] = df['high'].resample(period).max()
    convrted_df['low'] = df['low'].resample(period).min()
    convrted_df['close'] = df['close'].resample(period).last()
    convrted_df['volume'] = df['volume'].resample(period).sum()
    # Keep rows with at least 5 non-NaN values
    convrted_df.dropna(thresh=5, inplace=True)
    convrted_df.index = convrted_df['time']
    convrted_df['time'] = pd.DatetimeIndex(time_translation(t, min) for t in convrted_df['time'])
    convrted_df['timestamp'] = [time_to_timestamp(i) for i in convrted_df['time']]
    return convrted_df

def time_translation(ltime, min):
    res_time = (datetime.datetime.strptime(ltime, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=min)).strftime(
        '%Y-%m-%d %H:%M:%S')
    return res_time


# Create a fucntion that transforms 1 min of data into 4-hour data:
def transform_1min_into_4hr(csv_file, location):
    df = pd.read_csv(location+csv_file, engine="python", header=None)
    df.index = pd.to_datetime(df.iloc[:, 0])
    df = chg_col(df)
    resampled_df = myresample(df, "4h", -59)
    resampled_df.index = resampled_df['time']
    del resampled_df['time']
    resampled_df.reset_index(inplace = True)
    resampled_df = resampled_df.iloc[:-1, :]
    return resampled_df

resampled_ada = transform_1min_into_4hr("res_adau18z18乘10000000.csv", location)
resampled_trx = transform_1min_into_4hr("res_trxu18z18乘100000000.csv", location)
resampled_bch = transform_1min_into_4hr("res_bchu18z18乘1000.csv", location)
resampled_eos = transform_1min_into_4hr("res_eosu18z18乘100000.csv", location)
resampled_eth = transform_1min_into_4hr("res_ethu18z18乘10000.csv", location)
resampled_ltc = transform_1min_into_4hr("res_ltcu18z18乘10000.csv", location)
resampled_xrp = transform_1min_into_4hr("res_xrpu18z18乘10000000.csv", location)
resampled_btc = transform_1min_into_4hr("xbtusd_1m.csv", location)



# ===================================================================================================================
# 11_12_2018:
# Example: 
# rank_loc = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/"
# rank_file = "11_12_ranks_all.csv"

def clean_rank_data(rank_loc, rank_file):
    rank = pd.read_csv(rank_loc+rank_file, header=None)
    rank.columns = ['Ranks']
    rank['Dates'] = np.nan
    for i, date in enumerate(rank['Ranks']):
        if "-" in rank['Ranks'][i]:
            rank['Dates'][i] = rank['Ranks'][i]
    rank.ffill(inplace=True)
    rank_ = rank[rank['Ranks'] != rank['Dates']]
    rank_['Dates'] = rank_['Dates'].apply(lambda x: x.replace("排名", ""))
    rank_['Assets'], rank_['Rank'] = rank_['Ranks'].str.split(' ', 1).str
    rank_['Rank'] = rank_['Rank'].apply(lambda x: int(x))
    rank_cleaned = pd.pivot_table(rank_, values='Rank', columns='Assets', index = 'Dates')
    return rank_cleaned


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
    

def print_correction_rate(asst, rank_df, close_res_df):
    rank_asst = rank[['Date',asst]]
    rank_asst.index = rank_asst['Date']
    del rank_asst['Date']
    rank_asst.index = pd.to_datetime(rank_asst.index)
    rank_asst['date'] = rank_asst.index.astype(str)
    close_res_df['date'] = close_res_df.index.astype(str)
    merged = rank_asst.merge(close_res_df, on = 'date')
    merged['next_day_pct'] = merged['pct_chg']
    merged.dropna(inplace=True)
    print ("数字货币： ", merged.columns[0])
    print ("检验时间段：",merged['date'].values[0]," to ",merged['date'].values[-1])
    # correct prediction rate:
    try:
        correct_long_prdct = len(merged[((merged[asst] == 7) | (merged[asst] == 6) | (merged[asst] == 5)) &(merged['pct_chg']>0)])/len(merged[((merged[asst] == 7) | (merged[asst] == 6) | (merged[asst] == 5))])
    except ZeroDivisionError:
        print (print ('没有过做多信号'))
    else:
        print ("预测准确率：多头",correct_long_prdct)
        
    try:    
        correct_short_prdct = len(merged[((merged[asst] == 0) | (merged[asst] == 1) | (merged[asst] == 2)) &(merged['pct_chg']<0)])/len(merged[((merged[asst] == 0) | (merged[asst] == 1) | (merged[asst] == 2))])    
    except ZeroDivisionError:
        print (print ('没有过做空信号'))
    else:      
        print ("预测准确率：空头",correct_short_prdct)



# =============================================================================================================
# 6/18/2018 Updated:


"""
Convert io platform weekly dataset still to weekly dataset for Quantopian to backtest

"""

def next_week_start(df, time_col):
    """
    This function helps to find the next Monday, 
    which are the next_week_start column in the weekly data file 
    """
    df['next_week_start'] = pd.DatetimeIndex(df[time_col]) + pd.DateOffset(7)
    return df

tech_accern['week_dt'] = tech_accern['week'].apply(lambda x: x[:10])
# nw means new week.
tech_accern_nw = next_week_start(tech_accern, 'week_dt')
tech_accern_nw['current_date'] = tech_accern_nw['next_week_start']
"""
Print all trading dates from dt1 to dt2:

"""


"""
Print all previous weekday holidays & number of trading days in any years.

"""
from dateutil import rrule 
import datetime as dt

# Generate ruleset for holiday observances on the NYSE
# Change this forward looking days to backward looking days, I'll need to change to - datetime.timedelta(days)
def NYSE_holidays(a=datetime.date.today()-datetime.timedelta(days=1200), b=datetime.date.today()):
    rs = rrule.rruleset()

    # Include all potential holiday observances
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth=12, bymonthday=31, byweekday=rrule.FR)) # New Years Day  
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 1, bymonthday= 1))                     # New Years Day  
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 1, bymonthday= 2, byweekday=rrule.MO)) # New Years Day    
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 1, byweekday= rrule.MO(3)))            # Martin Luther King Day   
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 2, byweekday= rrule.MO(3)))            # Washington's Birthday
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, byeaster= -2))                                  # Good Friday
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 5, byweekday= rrule.MO(-1)))           # Memorial Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 7, bymonthday= 3, byweekday=rrule.FR)) # Independence Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 7, bymonthday= 4))                     # Independence Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 7, bymonthday= 5, byweekday=rrule.MO)) # Independence Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 9, byweekday= rrule.MO(1)))            # Labor Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth=11, byweekday= rrule.TH(4)))            # Thanksgiving Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth=12, bymonthday=24, byweekday=rrule.FR)) # Christmas  
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth=12, bymonthday=25))                     # Christmas  
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth=12, bymonthday=26, byweekday=rrule.MO)) # Christmas 
    
    # Exclude potential holidays that fall on weekends
    rs.exrule(rrule.rrule(rrule.WEEKLY, dtstart=a, until=b, byweekday=(rrule.SA,rrule.SU)))

    return rs
    
# Generate ruleset for NYSE trading days

def NYSE_tradingdays(a=datetime.date.today(), b=datetime.date.today()+datetime.timedelta(days=365)):
    rs = rrule.rruleset()
    rs.rrule(rrule.rrule(rrule.DAILY, dtstart=a, until=b))
    
    # Exclude weekends and holidays
    rs.exrule(rrule.rrule(rrule.WEEKLY, dtstart=a, byweekday=(rrule.SA,rrule.SU)))
    rs.exrule(NYSE_holidays(a,b))
    
    return rs

# Examples

# List all NYSE holiday observances for the coming year
print ("NYSE Holidays\n")
for dy in NYSE_holidays():
    print (dy.strftime('%b %d %Y'))

# Count NYSE trading days in next 5 years
print ("\n\nTrading Days\n")
for yr in range(2014,2018):
    tdays = len(list(NYSE_tradingdays(datetime.datetime(yr,1,1),datetime.datetime(yr,12,31))))
    print ("{0}  {1}".format(yr,tdays))
    
    
# List all NYSE holiday observances for the coming year
wd_holidays = []
print ("NYSE Holidays\n")
for dy in NYSE_holidays():
    print ('Adding to list: ', str(dy.strftime('%Y-%m-%d')))
    wd_holidays.append(str(dy.strftime('%Y-%m-%d')))
    
print ('\nWeekyday Holiday List: ')
wd_holidays


# =============================================================================================================

# 6/8/2018 Updated

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))



# 6/7/2018 Updated

# df in the input here should already be the processed signal dataframe
def hourly_df_trans_to_Q(df, score_col):
    # Transform into normal time format
    df['hour'] = df['hour'].apply(lambda x: x.replace('T', ' '))
    df['ts'] = df['hour'].apply(lambda x: x[:19])
    
    df['normed_score'] = df.groupby(['ts'])[score_col].apply(lambda x: crrnt_norm_val_rank(x))
    df_hour_cols = df[['entity_ticker', 'ts', 'normed_score']]
    df_hour_cols.columns = ['symbol', 'ts', 'normed_score']
    
    # Shift the hours to the next hours for quantopian to recognize to trade
    # The idea is to trade on next hour bar's open based on previous hour bar's normed score value
    new_hour_shifted = shift_1_hour(df_hour_cols, 'ts', 'next_hour')
    new_hour_shifted.columns = ['symbol', 'ts', 'normed_score', 'next_hour']
    new_hour_shifted_Q = new_hour_shifted[['symbol', 'normed_score', 'next_hour']]
    new_hour_shifted_Q['enter_hour'] = new_hour_shifted_Q['next_hour']
    
    # Trim the hours to be only trading hours in the dataset
    new_hour_shifted_Q['hour_to_filter'] = new_hour_shifted_Q['next_hour'].dt.hour
    final_df_ = new_hour_shifted_Q[(new_hour_shifted_Q['hour_to_filter']>9)&(new_hour_shifted_Q['hour_to_filter']<16)]
    
    # Only select those columns that are needed in Quantopian's backtesting IDE
    final_df = final_df_[['symbol', 'normed_score', 'enter_hour', 'next_hour']]
    return final_df



# 6/1/2018 Updated


"""
Convert calendar Monday (for weekly datasets) to first business day of the week for Quantopian to run.
"""

def replace_calendar_Monday_for_Q(df, time_col):
    df[time_col] = df[time_col].apply(lambda x: str(x))
    df[time_col] = df[time_col].apply(lambda x: x.replace(' 00:00:00', ''))
    
    df.replace('2014-09-01', '2014-09-02', inplace=True)
    df.replace('2014-12-29', '2015-01-02', inplace=True)
    df.replace('2015-01-19', '2015-01-20', inplace=True)
    df.replace('2015-02-16', '2015-02-17', inplace=True)
    df.replace('2015-05-25', '2015-05-26', inplace=True)
    df.replace('2015-09-07', '2015-09-08', inplace=True)
    df.replace('2016-01-18', '2016-01-19', inplace=True)
    df.replace('2016-02-15', '2016-02-16', inplace=True)
    df.replace('2016-05-30', '2016-05-31', inplace=True)
    df.replace('2016-07-04', '2016-07-05', inplace=True)
    df.replace('2016-09-05', '2016-09-06', inplace=True)
    df.replace('2016-12-26', '2016-12-27', inplace=True)
    df.replace('2017-01-02', '2017-01-03', inplace=True)
    df.replace('2017-01-16', '2017-01-17', inplace=True)
    df.replace('2017-02-20', '2017-02-21', inplace=True)
    df.replace('2017-05-29', '2017-05-30', inplace=True)
    df.replace('2017-09-04', '2017-09-05', inplace=True)
    df.replace('2017-12-25', '2017-12-26', inplace=True)
    df.replace('2018-01-01', '2018-01-02', inplace=True)
    df.replace('2018-01-15', '2018-01-16', inplace=True)
    df.replace('2018-02-19', '2018-02-20', inplace=True)
    
    df['year'] = pd.to_datetime(df['next_week_start']).dt.year
    df['month'] = pd.to_datetime(df['next_week_start']).dt.month
    df['day'] = pd.to_datetime(df['next_week_start']).dt.day
    
    return df

ws2 = replace_calendar_Monday_for_Q(weekly_df, 'next_week_start')


"""
Build AIO daily-weekly mean func
"""
def next_week_start(df, time_col):
    """
    This function helps to find the next Monday, 
    which are the next_week_start column in the weekly data file 
    """
    df['next_week_start'] = pd.DatetimeIndex(df[time_col]) + pd.DateOffset(7)
    return df


def find_week_dates(time, week_dates):
    """
    This function finds all 7 days that fall into each week.
    This funciton is called in the DS2_weekly_mean_trans function.
    """
    week_dates_i = week_dates.query("week_start <= '" + str(time) + "'and week_end >= '" + str(time) + "'").copy()
    return week_dates_i


def DS2_weekly_mean_trans(ds2, first_sunday_prior, last_day_in_file):
    """
    This function transforms daily DS2 dataset to weekly-mean dataset through the following steps:
    1. Find all each week's Monday(week_start) and Sunday(week_end) and put them in line with daily dates
    2. Aggregate and group by entity tickers and each week to calculate weekly mean values for each stock
    3. Create the next Monday's date column, as the next_week_start column in the final weekly file
    """
    
    ds2['date'] = pd.to_datetime(ds2['date']).dt.date
    ds2['date'] = pd.to_datetime(ds2['date'])
    del ds2['timestamp']
    ds2.columns = ['time', 'entity_ticker', 'score']
    ds2["month"] = ds2["time"].dt.month
    ds2["year"] = ds2["time"].dt.year
    ds2["day"] = ds2["time"].dt.day
    ds2["dayofweek"] = ds2["time"].dt.dayofweek
    
    # Pick a Sunday prior to the starting date('2014-07-27' in this case), as the starting week_end. 
    week_end = pd.to_datetime(first_sunday_prior)
    week_ends = []
    
    # Create all Sundays 
    while week_end <= pd.to_datetime(last_day_in_file):# last date in the file, '2018-3-28' in this case
        week_ends.append(week_end)
        week_end += datetime.timedelta(days=7)
    week_dates = pd.Series(week_ends, name='week_end').to_frame()
    
    # Create the week_start col
    week_dates['week_start'] = week_dates['week_end'].shift(1)
    week_dates['week_start'] = week_dates['week_start'].apply(lambda x: x + datetime.timedelta(days=1))
    week_dates = week_dates[['week_start', 'week_end']].dropna()
    
    # Find all daily dates in the daily DS2 file
    unique_times = ds2['time'].apply(lambda x: x.date()).unique().tolist()
    
    # Create a new dataframe, get daily dates and match with their week_start and week_end.
    ds2_week_dates = pd.DataFrame()
    for time in unique_times:
        week_dates_i = find_week_dates(time, week_dates)
        week_dates_i['time'] = time
        ds2_week_dates = ds2_week_dates.append(week_dates_i)
    ds2_week_dates = ds2_week_dates.reset_index(drop=True) 
    ds2_week_dates = ds2_week_dates.apply(lambda x: pd.to_datetime(x), axis = 1)
    merged = ds2_week_dates.merge(ds2)
    ws2 = merged[['week_start','week_end','entity_ticker', 'score', 'time']].sort_values(['entity_ticker', 'time'])
    
    # ****************** Calculate weekly mean ******************
    ws2_mean = ws2.groupby(['entity_ticker', 'week_start', 'week_end']).mean()
    ws2_mean_reset = ws2_mean.reset_index()[['entity_ticker', 'week_start', 'score']]
    ws2_mean_reset.columns = ['symbol', 'week_start', 'score']
    
    # Create next_week_start column 
    ws2_mean_reset_nextd = next_week_start(ws2_mean_reset, 'week_start')
    # Generate the final version of the file, with only 'symbol', 'score', 'next_week_start' columns
    ws2_mean_reset_nextd_ = ws2_mean_reset_nextd[['symbol', 'score', 'next_week_start']]
    
    return ws2_mean_reset_nextd_

folder_name = '/Users/brad_sun/Downloads/accern/accern_data/'
file_name = '4_30_gtm_3_days_filled.csv'
ds2 = pd.read_csv(folder_name+file_name)

ws2_mean = DS2_weekly_mean_trans(ds2, '2014-07-27', '2018-3-28')


"""
Build AIO daily-weekly sum func:
"""

def next_week_start(df, time_col):
    """
    This function helps to find the next Monday, 
    which are the next_week_start column in the weekly data file 
    """
    df['next_week_start'] = pd.DatetimeIndex(df[time_col]) + pd.DateOffset(7)
    return df

def find_week_dates(time, week_dates):
    """
    This function finds all 7 days that fall into each week.
    This funciton is called in the DS2_weekly_mean_trans function.
    """
    week_dates_i = week_dates.query("week_start <= '" + str(time) + "'and week_end >= '" + str(time) + "'").copy()
    return week_dates_i

def DS2_weekly_sum_trans(ds2, first_sunday_prior, last_day_in_file):
    """
    This function transforms daily DS2 dataset to weekly-sum dataset through the following steps:
    1. Find all each week's Monday(week_start) and Sunday(week_end) and put them in line with daily dates
    2. Aggregate and group by entity tickers and each week to calculate weekly sum values for each stock
    3. Create the next Monday's date column, as the next_week_start column in the final weekly file
    """
    ds2['date'] = pd.to_datetime(ds2['date']).dt.date
    ds2['date'] = pd.to_datetime(ds2['date'])
    del ds2['timestamp']
    ds2.columns = ['time', 'entity_ticker', 'score']
    ds2["month"] = ds2["time"].dt.month
    ds2["year"] = ds2["time"].dt.year
    ds2["day"] = ds2["time"].dt.day
    ds2["dayofweek"] = ds2["time"].dt.dayofweek
    
    # Pick a Sunday prior to the starting date('2014-07-27' in this case), as the first week_end. 
    week_end = pd.to_datetime(first_sunday_prior)
    week_ends = []
    
    # Create all Sundays 
    while week_end <= pd.to_datetime(last_day_in_file):# last date in the file, '2018-3-28' in this case
        week_ends.append(week_end)
        week_end += datetime.timedelta(days=7)
    week_dates = pd.Series(week_ends, name='week_end').to_frame()
    
    # Create the week_start col
    week_dates['week_start'] = week_dates['week_end'].shift(1)
    week_dates['week_start'] = week_dates['week_start'].apply(lambda x: x + datetime.timedelta(days=1))
    week_dates = week_dates[['week_start', 'week_end']].dropna()
    
    # Find all daily dates in the daily DS2 file
    unique_times = ds2['time'].apply(lambda x: x.date()).unique().tolist()
    
    # Create a new dataframe, get daily dates and match with their week_start and week_end.
    ds2_week_dates = pd.DataFrame()
    for time in unique_times:
        week_dates_i = find_week_dates(time, week_dates)
        week_dates_i['time'] = time
        ds2_week_dates = ds2_week_dates.append(week_dates_i)
    ds2_week_dates = ds2_week_dates.reset_index(drop=True) 
    ds2_week_dates = ds2_week_dates.apply(lambda x: pd.to_datetime(x), axis = 1)
    merged = ds2_week_dates.merge(ds2)
    ws2 = merged[['week_start','week_end','entity_ticker', 'score', 'time']].sort_values(['entity_ticker', 'time'])
    
    # ****************** Calculate weekly sum ******************
    ws2_sum = ws2.groupby(['entity_ticker', 'week_start', 'week_end']).sum()
    ws2_sum_reset = ws2_sum.reset_index()[['entity_ticker', 'week_start', 'score']]
    ws2_sum_reset.columns = ['symbol', 'week_start', 'score']
    
    # Create next_week_start(next Monday) column 
    ws2_sum_reset_nextd = next_week_start(ws2_sum_reset, 'week_start')
    # Generate the final version of the file, with only ['symbol', 'score', 'next_week_start'] as columns
    ws2_sum_reset_nextd_ = ws2_sum_reset_nextd[['symbol', 'score', 'next_week_start']]
    
    return ws2_sum_reset_nextd_


folder_name = '/Users/brad_sun/Downloads/accern/accern_data/'
file_name = '4_30_gtm_3_days_filled.csv'
ds2 = pd.read_csv(folder_name+file_name)
ws2_sum = DS2_weekly_sum_trans(ds2, '2014-07-27', '2018-3-28')




# 5/31/2018 Updated - Daily value converting to Monthly Values


def replace_month_start_open(df):
    df['next_month'] = df['next_month'].apply(lambda x: str(x))
    # Pay attention here to the replace format since this could change because of our updates in the production data
    df['next_month'] = df['next_month'].apply(lambda x: x.replace(' 00:00:00', ''))
    df.replace('2014-09-01', '2014-09-02', inplace=True)
    df.replace('2015-01-01', '2015-01-02', inplace=True)
    df.replace('2016-01-01', '2016-01-04', inplace=True)
    df.replace('2017-01-02', '2017-01-03', inplace=True)
    df.replace('2018-01-01', '2018-01-02', inplace=True)
    df['next_month'] = pd.to_datetime(df['next_month'])
    return df


raw_io_data['month'] = pd.to_datetime(raw_io_data['date']).dt.month
raw_io_data['year'] = pd.to_datetime(raw_io_data['date']).dt.year
raw_io_data['yr_month'] = raw_io_data['year'].astype(str) + '-' + raw_io_data['month'].astype(str) 


# This part need to be changed every time we change agg func
monthly_score = pd.DataFrame(raw_io_data.groupby(['yr_month', 'symbol'])['score'].sum())
monthly_score.reset_index(inplace=True)

monthly_score['date'] = pd.to_datetime(monthly_score['yr_month'])
monthly_score['next_month'] = monthly_score['date'].apply(lambda x: pd.bdate_range(start=x, periods=2, freq='BMS')[1])


monthly_score_bday = replace_month_start_open(monthly_score)
monthly_score_bday.sort_values('next_month', inplace = True)
monthly_score_bday_for_Q = monthly_score_bday[['symbol', 'score', 'next_month']]


monthly_score_bday_for_Q['day'] = monthly_score_bday_for_Q['next_month'].dt.day
monthly_score_bday_for_Q['month'] = monthly_score_bday_for_Q['next_month'].dt.month
monthly_score_bday_for_Q['year'] = monthly_score_bday_for_Q['next_month'].dt.year


# 5/27/2018 Updated

"""
This set of code aims at transforming Quantopian's ticker format into normal string ticker format.

"""
# Getting the return data of assets. 
# start = '2016-01-01'
# end = '2016-02-01'

# symbols = ['AAPL', 'MSFT', 'BRK-A', 'GE', 'FDX', 'SBUX']
# prices = get_pricing(symbols, start_date = start, end_date = end, fields = 'price')
# prices.columns = map(lambda x: x.symbol, prices.columns)
# returns = prices.pct_change()[1:]




# 5/17/2018 Updated ===================================================================================================================================================

# Shift the hours to the next hours
def shift_1_hour(df, time_col, next_hour):
    df[next_hour] = pd.to_datetime(df[time_col]) + datetime.timedelta(hours = 1, minutes = 0)
    return df

# Create a function for locating the current value in the historical normalized range based on ranks:
def crrnt_norm_val_rank(sr):
    norm_val_sr = sr.rank()/sr.rank().max()
    return norm_val_sr


# Build a function to transform into final Quantopian backtested file
# df in the input here should already be the processed signal dataframe
def hourly_df_trans_to_Q(df, score_col):
    # Transform into normal time format
    df['hour'] = df['hour'].apply(lambda x: x.replace('T', ' '))
    df['ts'] = df['hour'].apply(lambda x: x[:19])
    del df['hour']
    
    df['normed_score'] = df.groupby(['ts'])[score_col].apply(lambda x: crrnt_norm_val_rank(x))
    df_hour_cols = df[['entity_ticker', 'ts', 'normed_score']]
    df_hour_cols.columns = ['symbol', 'hour', 'normed_score']
    
    # Shift the hours to the next hours for quantopian to recognize
    # The idea is to trade on next hour bar's open based on previous hour bar's normed score value
    new_hour_shifted = shift_1_hour(ds1_hour_cols, 'ts', 'next_hour')
    new_hour_shifted.columns = ['symbol', 'hour', 'normed_score', 'next_hour']
    new_hour_shifted_Q = new_hour_shifted[['symbol', 'normed_score', 'next_hour']]
    new_hour_shifted_Q['enter_hour'] = new_hour_shifted_Q['next_hour']
    
    # Trim the hours to be only trading hours in the dataset
    new_hour_shifted_Q['hour_to_filter'] = new_hour_shifted_Q['next_hour'].dt.hour
    final_df_ = new_hour_shifted_Q[(new_hour_shifted_Q['hour_to_filter']>9)&(new_hour_shifted_Q['hour_to_filter']<16)]
    
    # Only select those columns that are needed in Quantopian's backtesting IDE
    final_df = final_df_[['symbol', 'normed_score', 'enter_hour', 'next_hour']]
    return final_df




# 5/15/2018 Updated ===================================================================================================================================================


# Create a function for locating the current value in the historical normalized range based on maxmin:
def crrnt_norm_val(sr):
    norm_val_sr = (sr - sr.min())/(sr.max()-sr.min())
    return norm_val_sr

# Create a function for locating the current value in the historical normalized range based on ranks:
def crrnt_norm_val_rank(sr):
    norm_val_sr = sr.rank()/sr.rank().max()
    return norm_val_sr



# 5/14/2018 Updated ===================================================================================================================================================

def build_vol_fltred_df_all(accern_df, ticker_col, time_col, rolling_wdw, avg_vol_fltr, score_col):
    
    accern_df = accern_df.copy()
    # Define start date and end date:
    df_sorted = accern_df.sort_values(time_col)
    df_sorted.reset_index()[time_col][len(df_sorted)-1]
    
    if isinstance(df_sorted.reset_index()[time_col][0], str) == True:
        start_date = df_sorted.reset_index()[time_col][0]
        end_date = df_sorted.reset_index()[time_col][len(df_sorted)-1]
    else:
        start_date = df_sorted.reset_index()[time_col][0].strftime('%Y-%m-%d')
        end_date = df_sorted.reset_index()[time_col][len(df_sorted)-1].strftime('%Y-%m-%d')
        
    # Get unique ticker volume data from yahoo finance
    ticker_list = list(accern_df[ticker_col].unique())
    price_all = yf.download(ticker_list, start_date, end_date)
    price_df_rst_indx = price_all.to_frame().reset_index()
    price_df_rst_indx.columns = ['Date', 'Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    vol_to_merge = price_df_rst_indx[['Date', 'Ticker', 'Volume']]
    
    # Calculate rolling mean on the volume column:
    vol_to_merge['{}d_avg_vol'.format(rolling_wdw)] = vol_to_merge.groupby(['Ticker'])['Volume'].apply(lambda x: x.rolling(window = rolling_wdw).mean())
    vol_only = vol_to_merge[['Date', 'Ticker', '{}d_avg_vol'.format(rolling_wdw)]]
    vol_only.columns = [time_col, ticker_col,'{}d_avg_vol'.format(rolling_wdw)]
    
    # Merge accern_df and price_df
    vol_only.reset_index(inplace=True)
    accern_df.reset_index(inplace=True)
    vol_only[time_col] = pd.to_datetime(vol_only[time_col])
    accern_df[time_col] = pd.to_datetime(accern_df[time_col])
    merged = vol_only.merge(accern_df, on = [time_col, ticker_col])
    
    # Filter based on volume
    final_fltred = merged[merged['{}d_avg_vol'.format(rolling_wdw)]>avg_vol_fltr]
    final_fltred_ = final_fltred[[time_col, ticker_col, '{}d_avg_vol'.format(rolling_wdw), score_col]]
    
    return final_fltred_


def build_vol_fltred_df_vol_lcl(accern_df, ticker_col, volume_file, time_col, rolling_wdw, avg_vol_fltr, score_col):
    
    accern_df = accern_df.copy()
    vol_to_merge = pd.read_csv("/Users/brad_sun/Downloads/accern/accern_data/"+volume_file)
    
    # Calculate rolling mean on the volume column:
    vol_to_merge['{}d_avg_vol'.format(rolling_wdw)] = vol_to_merge.groupby(['Ticker'])['Volume'].apply(lambda x: x.rolling(window = rolling_wdw).mean())
    vol_only = vol_to_merge[['Date', 'Ticker', '{}d_avg_vol'.format(rolling_wdw)]]
    vol_only.columns = [time_col, ticker_col,'{}d_avg_vol'.format(rolling_wdw)]
    
    # Merge accern_df and price_df
    vol_only.reset_index(inplace=True)
    accern_df.reset_index(inplace=True)
    vol_only[time_col] = pd.to_datetime(vol_only[time_col])
    accern_df[time_col] = pd.to_datetime(accern_df[time_col])
    merged = vol_only.merge(accern_df, on = [time_col, ticker_col])
    
    # Filter based on volume
    final_fltred = merged[merged['{}d_avg_vol'.format(rolling_wdw)]>avg_vol_fltr]
    final_fltred_ = final_fltred[[time_col, ticker_col, '{}d_avg_vol'.format(rolling_wdw), score_col]]
    
    return final_fltred_


def vol_df_trans_to_Q_format(final_fltred_df, time_col, symbol_col, score_col):
    final_fltred_df['enter_date'] = final_fltred_df[time_col]
    df_for_Q = final_fltred_df[[time_col, ticker_col, score_col, 'enter_date']]
    return df_for_Q



def replace_calendar_Monday_for_Q(df, time_col):
    df[time_col] = df[time_col].apply(lambda x: str(x))
    df[time_col] = df[time_col].apply(lambda x: x.replace(' 00:00:00', ''))
    
    df.replace('2014-09-01', '2014-09-02', inplace=True)
    df.replace('2014-12-29', '2015-01-02', inplace=True)
    df.replace('2015-01-19', '2015-01-20', inplace=True)
    df.replace('2015-02-16', '2015-02-17', inplace=True)
    df.replace('2015-05-25', '2015-05-26', inplace=True)
    df.replace('2015-09-07', '2015-09-08', inplace=True)
    df.replace('2016-01-18', '2016-01-19', inplace=True)
    df.replace('2016-02-15', '2016-02-16', inplace=True)
    df.replace('2016-05-30', '2016-05-31', inplace=True)
    df.replace('2016-07-04', '2016-07-05', inplace=True)
    df.replace('2016-09-05', '2016-09-06', inplace=True)
    df.replace('2016-12-26', '2016-12-27', inplace=True)
    
    df.replace('2017-01-02', '2017-01-03', inplace=True)
    
    df.replace('2017-01-16', '2017-01-17', inplace=True)
    df.replace('2017-02-20', '2017-02-21', inplace=True)
    df.replace('2017-05-29', '2017-05-30', inplace=True)
    df.replace('2017-09-04', '2017-09-05', inplace=True)
    df.replace('2017-12-25', '2017-12-26', inplace=True)
    df.replace('2018-01-01', '2018-01-02', inplace=True)
    df.replace('2018-01-15', '2018-01-16', inplace=True)
    df.replace('2018-02-19', '2018-02-20', inplace=True)
    
    df['year'] = pd.to_datetime(df['next_week_start']).dt.year
    df['month'] = pd.to_datetime(df['next_week_start']).dt.month
    df['day'] = pd.to_datetime(df['next_week_start']).dt.day
    
    return df


def replace_day_open_for_Q(df, time_col):
    df['next_day'] = pd.DatetimeIndex(df[time_col]) + pd.DateOffset(1)
    return df


# This part need to be re-editted
def keep_downloading_data():
    folder_path = '/Users/brad_sun/Downloads/accern/accern_data/'
    ticker_file = 'sp500_csv_list.csv'

    ticker_df = pd.read_csv(folder_path+ticker_file)
    ticker_list = list(ticker_df['Ticker symbol'])

    new_sp500_df = pd.DataFrame()
    while new_sp500_df.empty == True:
        i = 1
        print ('New SP500 dataframe is empty. Trial {}, downloading data...'.format(i))
        new_sp500_df = yf.download(ticker_list, '2013-08-01', '2018-04-01')
        i = i + 1
        
        if new_sp500_df.empty == False:
            print ('All data Collected')
        
    sp500_df = new_sp500_df.to_frame().unstack()
    return sp500_df


####################################### Completed/Updated on 3_23_2018 #######################################

# This part is for taking an initial look at entity_ticker count distribution across dataset's timeframe:
def entity_temporal_distribution(df, time_col, entity_col, start_date, end_date, freq):
    entity_count = df.groupby(time_col)[entity_col].count()
    entity_count_df = pd.DataFrame(entity_count)
    entity_count_df.columns = ['entity_count']
    entity_count_df.plot.bar(figsize = (14, 4))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    plt.title('Dataset Entity Count Coverage', fontsize = 14)
    plt.ylabel('Entity Ticker Count Per {}'.format(freq))
    plt.xlabel('Timeframe: {} to {}'.format(start_date, end_date))
    
# Compare different lengths of dataframe by dropping NaNs and duplicates
def dropna_dropduplicates(df):
    print ('Before dropping NaNs, the length of df: ', len(df))
    NaNs_dropped = df.dropna()
    print ('After dropping NaNs, the length of df: ', len(NaNs_dropped))
    duplicates_dropped = df.drop_duplicates()
    print ('After dropping duplicates, the length of df: ', len(duplicates_dropped))
    both_dropped = NaNs_dropped.drop_duplicates()
    print ('After dropping NaNs and duplicates, the length of df: ', len(both_dropped))
    return both_dropped

# Trim dateframe based on a certain selected timeframe
def trim_timeframe(df, time_col, start_date, end_date):
    df = df[(df[time_col]>=start_date)&(df[time_col]<=end_date)]
    return df

# Show part of the dataframe
def show_df_head_tail(df):
    print ('The first two rows: ')
    print ('The last two rows: ')
    return df.head(2), df.tail(2)
    
# Plot temporal distribution of the processed siganl data
def signal_temporal_distribution_df(df, time_col, score_col, start_date, end_date, freq):
    signal_count = df.groupby(time_col)[score_col].count()
    signal_count_df = pd.DataFrame(signal_count)
    signal_count_df.columns = ['signal_count']
    signal_count_df.plot.bar(figsize = (14, 4))
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
    plt.title('Processed Dataset Signal Count Coverage', fontsize = 14)
    plt.ylabel('Signal Count Per {}'.format(freq))
    plt.xlabel('Timeframe: {} to {}'.format(start_date, end_date))
                       
# Plot basic stats and plotting of the data. 
def data_hist_plot(df, col):
    df[col].hist(bins=50, figsize=(12, 4))
    plt.title('Distribution on "{}"'.format(col), fontsize = 14)
    plt.xlabel('{}'.format(col))
    plt.ylabel('{} Total Frequency'.format(col))            
            
# This ticker info is downloaded from Wikipedia:
def entity_sector_QA(df, entity_col, signal_col):
    ticker_info = pd.read_csv(data_path+'sp500_csv_list.csv')
    ticker_sector_mapping = {}
    for i, row in enumerate(ticker_info['Ticker symbol']):
        ticker_sector_mapping[ticker_info['Ticker symbol'][i]] = ticker_info['GICS Sector'][i]
    df['sectors'] = df[entity_col].map(ticker_sector_mapping)
    print ('Total Unique Entity Tickers: ', df[entity_col].nunique())
    print ('Total Unique Sectors: ', df['sectors'].nunique())
    print ('Total rows of tickers: ', len(df[entity_col]))
    print ('Total rows of sectors: ', len(df['sectors']))
    print ('*'*60)
    print ('10 stocks with the LEAST signals; number of signals: ') 
    print (df.groupby([entity_col])[signal_col].count().sort_values().head(10))
    print ('*'*60)
    print ('10 stocks with the MOST signals; number of signals: ') 
    print (df.groupby([entity_col])[signal_col].count().sort_values().tail(10))
    print ('*'*60)
    print ('Total Signal Distribution - Sector-Wise - Stats')
    print (df.groupby(['sectors'])[signal_col].count())
    print ('*'*60)
    print ('Total Signal Distribution - Sector-Wise - Plot')
    df.groupby(['sectors'])[signal_col].count().plot.bar(figsize = (12, 4));        

# This part is for (if needed) reducing signal files to reduce number of stocks to trade on a daily basis.
# Mainly because of Quantopian's lack of capacity of loading large csv data file.
def reduce_size_for_Q(df, signal_col, reduce_down_to_num, time_col):
    top = df.dropna().sort_values(signal_col, ascending=False).groupby(time_col).head(reduce_down_to_num)
    bottom = df.dropna().sort_values(signal_col, ascending=False).groupby(time_col).tail(reduce_down_to_num)
    reduced_df = top.append(bottom)
    reduced_df = reduced_df.sort_values(time_col)
    reduced_df = reduced_df.drop_duplicates()
    return reduced_df    
    
# This is originally built based on open-open strategies. It replaces dates with the next dates to avoid forward-looking bias.
def shift_1_day(df, time_col, next_day):
    df[next_day] = pd.DatetimeIndex(df[time_col]) + pd.DateOffset(1)
    return df

def shift_1_hour(df, time_col, next_hour):
    df[next_hour] = pd.to_datetime(df[time_col]) + datetime.timedelta(hours = 1, minutes = 0)
    return df
    
# Convert the month to the next month start (first business day recognized by Quantopian)
def replace_month_start_open_for_Q(df, time_col):
    df['next_month'] = df[time_col].apply(lambda x: pd.bdate_range(start=x, periods=2, freq='BMS')[1])
    df['next_month'] = df['next_month'].apply(lambda x: str(x))
    # Pay attention here to the replace format since this could change because of our updates in the production data
    df['next_month'] = df['next_month'].apply(lambda x: x.replace(' 00:00:00+00:00', ''))
    df.replace('2013-09-02', '2013-09-03', inplace=True)
    df.replace('2014-01-01', '2014-01-02', inplace=True)
    df.replace('2014-09-01', '2014-09-02', inplace=True)
    df.replace('2015-01-01', '2015-01-02', inplace=True)
    df.replace('2016-01-01', '2016-01-04', inplace=True)
    df.replace('2017-01-02', '2017-01-03', inplace=True)
    df.replace('2018-01-01', '2018-01-02', inplace=True)
    df['next_month'] = pd.to_datetime(df['next_month'])
    return df 
    
# Convert to Quantopian's backtesting format.
def convert_to_Q_format(df, time_col, ticker_col):
    df['time'] = pd.to_datetime(df[time_col])
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['day'] = df['time'].dt.day
    df['symbol'] = df[ticker_col]
    return df

####################################### Updated on 3_29_2018 #######################################
# See each column's length and their plots for lengths, as well as total row number, updated on 3_29_2018
def column_stats(df):
    print ('Total length of the dataframe :', len(df))
    all_columns = df.columns
    col_value = {}
    for col in all_columns:
        col_value[col] = len(df[col].dropna())
    new_df = pd.DataFrame(list(col_value.values()), index = list(col_value.keys()), columns=['Non_Null_Length'])
    new_df.plot.bar(figsize = (12, 4))
    plt.legend(loc="best")
    return new_df

# Let's take a further look at what specific rows have these nan values. 
def get_nan_rows(df, col):
    nan_rows = df.loc[df[col].isnull()]
    print ('Total nan rows for column {}: '.format(col), len(nan_rows))
    return nan_rows


# Initialize everything here.
io_file = '3_22_rerun_automation_entity_relevance_filter_lt_70_without_story_source_filter.csv'
data_path = '/Users/brad_sun/Downloads/accern/accern_data/'
raw_io_og = pd.read_csv(data_path+io_file)

time_col = 'date'
entity_col = 'entity_ticker'
start_date = '2013-08-01'
col_to_drop = 'alphatrend_dst_score' # To clarify, not meaning drop this col.
score_col = 'alphatrend_dst_score'
signal_col = 'alphatrend_dst_score'
end_date = '2018-02-28'
freq = 'Day'
reduce_down_to_num = 60






# Make sure in the dataframe the columne 'time' type is string
def execution_QA(df, score_col, check_days, time_col, num_stocks_to_trade, symbol_col):
    # Get the sorted datetime strings:
    timestamps = df[time_col].sort_values(ascending = True).unique()
    
    # get the full stock number lengths of the first three days:
    len_first_day = len(df[df[time_col] == timestamps[:check_days][0]].sort_values(score_col, ascending = False))
    len_second_day = len(df[df[time_col] == timestamps[:check_days][1]].sort_values(score_col, ascending = False))
    len_third_day = len(df[df[time_col] == timestamps[:check_days][2]].sort_values(score_col, ascending = False))
    
    df_len_dict_three_days = {timestamps[:check_days][0]:len_first_day, 
                          timestamps[:check_days][1]: len_second_day, 
                          timestamps[:check_days][2]: len_third_day}
    
    for key, value in sorted(df_len_dict_three_days.items()):
        print(key,'- Total Stock Num:', value)
        
    # Pay attention here, every time we run this, if we want to rerun, need to run the num_stocks_to_trade = the parameter we want to set
    actual_num_stocks_to_trade_dict = {}

    for key, value in sorted(df_len_dict_three_days.items()):
        num_to_trade = num_stocks_to_trade
        if value < 2*num_to_trade:
            num_to_trade = int(value/2)
            print ('{} stock num insufficient! Actual num to trade is: '.format(key), num_to_trade)
        else:
            print ('{} stock num sufficient. Actual num to trade is: '.format(key), num_to_trade)
        actual_num_stocks_to_trade_dict[key] = num_to_trade
        
    day_1_actual_num = actual_num_stocks_to_trade_dict[timestamps[:check_days][0]]
    day_2_actual_num = actual_num_stocks_to_trade_dict[timestamps[:check_days][1]]
    day_3_actual_num = actual_num_stocks_to_trade_dict[timestamps[:check_days][2]]
    
    ticker_to_long_day_1 = list(df[df[time_col] == timestamps[:check_days][0]].sort_values(score_col, ascending = False)[:day_1_actual_num][symbol_col])
    ticker_to_long_day_2 = list(df[df[time_col] == timestamps[:check_days][1]].sort_values(score_col, ascending = False)[:day_2_actual_num][symbol_col])
    ticker_to_long_day_3 = list(df[df[time_col] == timestamps[:check_days][2]].sort_values(score_col, ascending = False)[:day_3_actual_num][symbol_col])
    
    ticker_to_short_day_1 = list(df[df[time_col] == timestamps[:check_days][0]].sort_values(score_col, ascending = False)[-day_1_actual_num:][symbol_col])
    ticker_to_short_day_2 = list(df[df[time_col] == timestamps[:check_days][1]].sort_values(score_col, ascending = False)[-day_2_actual_num:][symbol_col])
    ticker_to_short_day_3 = list(df[df[time_col] == timestamps[:check_days][2]].sort_values(score_col, ascending = False)[-day_3_actual_num:][symbol_col])
    
    # For day 1, the executions include either long or short only
    positions_to_long_day_1 = ticker_to_long_day_1
    positions_to_short_day_1 = ticker_to_short_day_1


    # For day 2, the executions include 1) long, 2) short, 3) close long, 4) close short positions
    positions_to_long_day_2 = ticker_to_long_day_2
    positions_to_short_day_2 = ticker_to_short_day_2

    positions_to_close_long_day_2 = []
    for ticker in positions_to_long_day_1:
        if ticker not in positions_to_long_day_2:
            positions_to_close_long_day_2.append(ticker)

    positions_to_close_short_day_2 = []
    for ticker in positions_to_short_day_1:
        if ticker not in positions_to_short_day_2:
            positions_to_close_short_day_2.append(ticker)


    # For day 3, the executions include 1) long, 2) short, 3) close long, 4) close short positions
    positions_to_long_day_3 = ticker_to_long_day_3
    positions_to_short_day_3 = ticker_to_short_day_3

    positions_to_close_long_day_3 = []
    for ticker in positions_to_long_day_2:
        if ticker not in positions_to_long_day_3:
            positions_to_close_long_day_3.append(ticker)

    positions_to_close_short_day_3 = []
    for ticker in positions_to_short_day_2:
        if ticker not in positions_to_short_day_3:
            positions_to_close_short_day_3.append(ticker)
            
    # Printing the log:

    print ('='*60)
    print ('Positions to long on {}:\n{}'.format(timestamps[:check_days][0], sorted(positions_to_long_day_1)))
    print ('Number of stocks: {}'.format(len(positions_to_long_day_1)))
    print ('-'*60)
    print ('Positions to short on {}:\n{}'.format(timestamps[:check_days][0], sorted(positions_to_short_day_1)))
    print ('Number of stocks: {}'.format(len(positions_to_short_day_1)))

    print ('='*60)
    print ('Positions to long on {}:\n{}'.format(timestamps[:check_days][1], sorted(positions_to_long_day_2)))
    print ('Number of stocks: {}'.format(len(positions_to_long_day_2)))
    print ('-'*60)
    print ('Positions to short on {}:\n{}'.format(timestamps[:check_days][1], sorted(positions_to_short_day_2)))
    print ('Number of stocks: {}'.format(len(positions_to_short_day_2)))
    print ('-'*60)
    print ('Close long positions on {}:\n{}'.format(timestamps[:check_days][1], sorted(positions_to_close_long_day_2)))
    print ('Number of stocks: {}'.format(len(positions_to_close_long_day_2)))
    print ('-'*60)
    print ('Close short positions on {}:\n{}'.format(timestamps[:check_days][1], sorted(positions_to_close_short_day_2)))
    print ('Number of stocks: {}'.format(len(positions_to_close_short_day_2)))

    print ('='*60)
    print ('Positions to long on {}:\n{}'.format(timestamps[:check_days][2], sorted(positions_to_long_day_3)))
    print ('Number of stocks: {}'.format(len(positions_to_long_day_3)))
    print ('-'*60)
    print ('Positions to short on {}:\n{}'.format(timestamps[:check_days][2], sorted(positions_to_short_day_3)))
    print ('Number of stocks: {}'.format(len(positions_to_short_day_3)))
    print ('-'*60)
    print ('Close long positions on {}:\n{}'.format(timestamps[:check_days][2], sorted(positions_to_close_long_day_3)))
    print ('Number of stocks: {}'.format(len(positions_to_close_long_day_3)))
    print ('-'*60)
    print ('Close short positions on {}:\n{}'.format(timestamps[:check_days][2], sorted(positions_to_close_short_day_3)))
    print ('Number of stocks: {}'.format(len(positions_to_close_short_day_3)))
    print ('='*60)



def create_alphatrend_dst_score(df):
    df['ent_sent'] = df.groupby(['day'])['avg_entity_sentiment'].rank(ascending=True)
    df['eve_sent'] = df.groupby(['day'])['avg_event_sentiment'].rank(ascending=True)
    df['timeliness'] = df.groupby(['day'])['avg_entity_source_timeliness_score'].rank(ascending=True)
    df['traffic_sum'] = df.groupby(['day'])['avg_story_group_traffic_sum'].rank(ascending=True)
    df['alphatrend_dst_score'] = df['ent_sent']+df['eve_sent']+df['timeliness']+df['traffic_sum']
    return df


def generate_missing_backtests(result_file, config_file):
    results = pd.read_csv(result_file)
    sharpe_null_df = results[results['sharpe'].isnull()]
    rerun_result_df = sharpe_null_df[~sharpe_null_df.title.isnull()]
    rerun_algo_names = list(rerun_result_df.title)
    config = pd.read_csv(config_file)
    rerun_config_df = config[config['algo_name'].isin(rerun_algo_names)]
    return rerun_result_df

# =============================================================================================================



def get_price_df_from_liquid_russell_open_csv():
    price_df = pd.read_csv('Russell_top_mkt_cap_larger_than_1000mil_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df


def get_price_df_from_growth_stock_russell_open_csv():
    price_df = pd.read_csv('Russell_2000_renaissance_growth_stocks_daily_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df


def get_price_df_from_all_stock_russell_open_csv():
    price_df = pd.read_csv('2_8_Russell_2000_daily_all_data_all_tickers_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df


# ------------------- Make sure we don't have any forward-looking bias for daily strategy. meaning that we need to shift down all the dates using shift(1) -------------------
def get_daily_accern_for_Q(file_path):
    df = pd.read_csv(file_path)
    df['shifted_time'] = df.groupby('entity_ticker')['day'].shift(-1)
    df['time'] = pd.to_datetime(df['shifted_time'])
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['day'] = df['time'].dt.day
    df['symbol'] = df['entity_ticker']
    del df['entity_ticker']
    del df['shifted_time']
    return df.sort_values('time')
# ------------------- Make sure we don't have any forward-looking bias for daily strategy. meaning that we need to shift down all the dates using shift(1) -------------------


def get_price_df_from_liquid_russell_open_csv():
    price_df = pd.read_csv('Russell_top_mkt_cap_larger_than_1000mil_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df

def get_price_df_from_growth_russell_open_csv():
    price_df = pd.read_csv('Russell_growth_stocks_daily_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df
    
def get_price_df_from_sp500_open_csv():
    price_df = pd.read_csv('sp500_open_price_daily.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df



def last_trading_day_of_month_Quantopian():
  last_trading_day_of_month = \
  ['2013-08-30',
 '2013-09-30',
 '2013-10-31',
 '2013-11-29',
 '2013-12-31',
 '2014-01-31',
 '2014-02-28',
 '2014-03-31',
 '2014-04-30',
 '2014-05-30',
 '2014-06-30',
 '2014-07-31',
 '2014-08-29',
 '2014-09-30',
 '2014-10-31',
 '2014-11-28',
 '2014-12-31',
 '2015-01-30',
 '2015-02-27',
 '2015-03-31',
 '2015-04-30',
 '2015-05-29',
 '2015-06-30',
 '2015-07-31',
 '2015-08-31',
 '2015-09-30',
 '2015-10-30',
 '2015-11-30',
 '2015-12-31',
 '2016-01-29',
 '2016-02-29',
 '2016-03-31',
 '2016-04-29',
 '2016-05-31',
 '2016-06-30',
 '2016-07-29',
 '2016-08-31',
 '2016-09-30',
 '2016-10-31',
 '2016-11-30',
 '2016-12-30',
 '2017-01-31',
 '2017-02-28',
 '2017-03-31',
 '2017-04-28',
 '2017-05-31',
 '2017-06-30',
 '2017-07-31',
 '2017-08-31',
 '2017-09-29',
 '2017-10-31',
 '2017-11-30']
  return last_trading_day_of_month

  def first_trading_day_of_month_Quantopian():
    first_trading_day_of_month = \
    [
 '2013-09-03',
 '2013-10-01',
 '2013-11-01',
 '2013-12-02',
 '2014-01-02',
 '2014-02-03',
 '2014-03-03',
 '2014-04-01',
 '2014-05-01',
 '2014-06-02',
 '2014-07-01',
 '2014-08-01',
 '2014-09-02',
 '2014-10-01',
 '2014-11-03',
 '2014-12-01',
 '2015-01-02',
 '2015-02-02',
 '2015-03-02',
 '2015-04-01',
 '2015-05-01',
 '2015-06-01',
 '2015-07-01',
 '2015-08-03',
 '2015-09-01',
 '2015-10-01',
 '2015-11-02',
 '2015-12-01',
 '2016-01-04',
 '2016-02-01',
 '2016-03-01',
 '2016-04-01',
 '2016-05-02',
 '2016-06-01',
 '2016-07-01',
 '2016-08-01',
 '2016-09-01',
 '2016-10-03',
 '2016-11-01',
 '2016-12-01',
 '2017-01-03',
 '2017-02-01',
 '2017-03-01',
 '2017-04-03',
 '2017-05-01',
 '2017-06-01',
 '2017-07-03',
 '2017-08-01',
 '2017-09-01',
 '2017-10-02',
 '2017-11-01',
 '2017-12-01']
  return first_trading_day_of_month

def get_yahoo_multiple_tickers(ticker_list, start_date, end_date):
    df = yf.download(ticker_list, start = start_date, end = end_date)
    df_unstacked = df.to_frame().unstack()
    return df_unstacked

# import fix_yahoo_finance as yf
# data = yf.download("SPY", start="2017-01-01", end="2017-04-30")

def get_yahoo_single_ticker(ticker, start_date, end_date):
    df = yf.download(ticker, start = start_date, end = end_date)
    return df



# Note that metric_columns is a list containig the metric strings.
def trim_columns(read_path_name, metric_columns, export_path_name):
    df = pd.read_csv(read_path_name)
    default_columns = ['time', 'month', 'year', 'day', 'hour', 'symbol']
    metric_columns = metric_columns
    columns_to_keep =  set(default_columns) | set(metric_columns)
    columns_to_drop = list(set(df.columns) - set(columns_to_keep))
    df_trimmed = df.drop(columns_to_drop, axis = 1)
    df_trimmed.to_csv(export_path_name)
    return df_trimmed


def sector_names():

    sector_category = ['Real Estate', 'Industrials', 'Materials', 'Information Technology', 'Financials', 'Utilities', 'Telecommunication Services', 'Health Care', 'Energy', 'Consumer Staples', 'Consumer Discretionary']

    return sector_category


def sector_category_dict():
    c = pd.read_csv('/Users/workspace/Accern/Project1_backtesting_shared_on_github/Backtesting/sp500_csv_list.csv')
    sector_list = list(set(open_sector_df['GICS Sector']))
    iter_range = range(11)
    single_sector_dict = {}
    for i in iter_range:    
      all_info = open_sector_df[open_sector_df['GICS Sector'] == sector_list[i]]
      single_sector_dict[str(sector_list[i])] = all_info['Ticker symbol']
    return single_sector_dict


"""
This is the function that converts the raw Accern data's format to alphalens's required factor dataframe format, 
specifically for daily data. 'factor_csv_file' looks like: e.g. '1_16_10_ETFs_daily.csv'(raw data file downloaded from Accern's io platform).
"""

def process_factor_daily_data(factor_csv_file):
    accern_factor = pd.read_csv(factor_csv_file)
    accern_factor['datetime'] = pd.to_datetime(accern_factor['day'])
    accern_factor = accern_factor.set_index(['datetime', 'entity_ticker'])
    factor_index = accern_factor.index.get_level_values(0).tz_localize("UTC")
    accern_factor.set_index([factor_index, accern_factor.index.get_level_values(1)], inplace=True)
    del accern_factor['day']
    return accern_factor


# Look at the percentage change for the N forward periods in a dataframe for a column
def forward_looking_pct_periodwise(df, df_col, periods):
    df['{}_periods_away_pct_chg'.format(periods)] = df[df_col].pct_change(periods).shift(-periods)
    df['{}_periods_away_value'.format(periods)] = df[df_col].shift(-periods)
    return df



"""
This is the function that downloads yahoo finance adjusted close data based on Accern factor file's tickers and also converts 
to Alphalens's price dataframe format. 'factor_csv_file' looks like: e.g. '1_16_10_ETFs_daily.csv' (raw data file downloaded 
from Accern's io platform).

"""


def process_daily_adj_close_price_data(factor_csv_file):
    factor_df = pd.read_csv(factor_csv_file)
    factor_ticker_list = list(factor_df.entity_ticker.unique())
    price_data = get_yahoo_data(factor_ticker_list, '2013-08-01', '2017-11-30')
    close_price_reninv = price_data['Adj Close']
    price_index = close_price_reninv.index.tz_localize("UTC")
    close_price_reninv.set_index(price_index, inplace=True)
    
    return close_price_reninv


def process_monthly_adjclose_price_data(factor_csv_file):
    factor_df = pd.read_csv(factor_csv_file)
    factor_ticker_list = list(factor_df.entity_ticker.unique())
    price_data = get_yahoo_data(factor_ticker_list, '2013-08-01', '2017-11-30')
    close_price_reninv = price_data['Adj Close']
    
    close_price_reninv['datetime'] = pd.to_datetime(close_price_reninv.index)
    close_price_reninv['date'] = close_price_reninv['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))

    close_price_reninv['year'] = close_price_reninv.datetime.dt.year
    close_price_reninv['month'] = close_price_reninv.datetime.dt.month
    close_price_reninv['day'] = close_price_reninv.datetime.dt.day
    
    last_day_price_date_list = []
    for i, month in enumerate(close_price_reninv['month']):
        try: 
            if close_price_reninv['month'][i] != close_price_reninv['month'][i-1]:
                last_day_price_date_list.append(close_price_reninv['date'][i-1])
            else:
                continue
        except:
            continue
            
    close_price_reninv.index = close_price_reninv.datetime
    rows_str_to_drop = []
    for i, date in enumerate(close_price_reninv['date']):
        if close_price_reninv['date'][i] not in last_day_price_date_list:
            rows_str_to_drop.append(close_price_reninv['date'][i])
        else:
            continue

    close_price_reninv['to_be_dropped'] = close_price_reninv['date'].apply(lambda x: x in rows_str_to_drop)
    close_price_reninv.drop(close_price_reninv[close_price_reninv.to_be_dropped == True].index, inplace=True)
    
    pricing_index = close_price_reninv.index.tz_localize('UTC')
    close_price_reninv.index = pricing_index
    close_price_reninv = close_price_reninv.drop(['datetime', 'date', 'year', 'month', 'day'], axis=1)
    del close_price_reninv['to_be_dropped']
    return close_price_reninv



# We already have the monthly aggragted data from io platform, and convert to the last trading day of the month for the timestamp
def process_monthly_end_factor_data(factor_csv_file):
    open_accern_reninv = pd.read_csv(factor_csv_file)
    accern_factor_ = open_accern_reninv
    accern_factor_['time'] = pd.to_datetime(accern_factor_['month']) 
    
    offset = BMonthEnd()
    for i, date in enumerate(accern_factor_['time']):
        d = accern_factor_['time'][i].date()
        d = offset.rollforward(d)
        accern_factor_['time'][i] = d
    del accern_factor_['month']
    
    open_accern_reninv = accern_factor_  
    open_accern_reninv['datetime'] = pd.to_datetime(open_accern_reninv.time)
    accern_factor = open_accern_reninv
    accern_factor['datetime'] = pd.to_datetime(accern_factor['time'])
    accern_factor = accern_factor.set_index(['datetime', 'entity_ticker'])
    factor_index = accern_factor.index.get_level_values(0).tz_localize("UTC")
    accern_factor.set_index([factor_index, accern_factor.index.get_level_values(1)], inplace=True)
    del accern_factor['time']
    
    return accern_factor



"""
This function draws the period-wise mean return quantile factor charts of only one ticker, for a specific 
"""
def single_ticker_quantile_return_analysis(factor_csv_file, target_metric_str):
    single_df = pd.read_csv(factor_csv_file)
    single_df.index = pd.to_datetime(single_df['day'])
    single_df_index = single_df.index.tz_localize("UTC")
    single_df.set_index(single_df_index, inplace=True)
    del single_df['day']
    
    single_ticker = list(single_df.entity_ticker.unique())
    price_df = yf.download(single_ticker, start = '2013-08-01', end = '2017-11-30')
    price_df_adjclose =pd.DataFrame(price_df['Adj Close'])
    price_df_adjclose.columns = single_ticker
    price_df_index = price_df_adjclose.index.tz_localize("UTC")
    price_df_adjclose.set_index(price_df_index, inplace=True)
    
    single_df['merged_ts'] = single_df.index
    price_df_adjclose['merged_ts'] = price_df_adjclose.index
    merged_daily = price_df_adjclose.merge(single_df, on = 'merged_ts')
    target_metric_str = target_metric_str
    merged_daily['quantiles'] = pd.qcut(merged_daily[target_metric_str], 5, labels=np.arange(1, 6, 1))
    
    merged_daily['1_period_return_forward'] = merged_daily[single_ticker].pct_change(periods = 1).shift(-1)
    merged_daily['5_period_return_forward'] = merged_daily[single_ticker].pct_change(periods = 5).shift(-5)
    merged_daily['10_period_return_forward'] = merged_daily[single_ticker].pct_change(periods = 10).shift(-10)
    
    one_period_mean_return_forward = merged_daily.groupby(['quantiles'])['1_period_return_forward'].mean()
    five_period_mean_return_forward = merged_daily.groupby(['quantiles'])['5_period_return_forward'].mean()
    ten_period_mean_return_forward = merged_daily.groupby(['quantiles'])['10_period_return_forward'].mean()
    
    mean_return_quantile_analysis_df = pd.DataFrame()
    mean_return_quantile_analysis_df['1_period_mean_return_forward'] = one_period_mean_return_forward
    mean_return_quantile_analysis_df['5_period_mean_return_forward'] = five_period_mean_return_forward
    mean_return_quantile_analysis_df['10_period_mean_return_forward'] = ten_period_mean_return_forward

    mean_return_quantile_analysis_df.plot(kind='bar', figsize=(18, 6))




def sector_names_mapping():
  sector_names = {
    0 : "Information Technology",
    1 : "Financials",
    2 : "Health Care",
    3 : "Industrials",
    4 : "Utilities", 
    5 : "Real Estate", 
    6 : "Materials", 
    7 : "Telecommunication Services", 
    8 : "Consumer Staples", 
    9 : "Consumer Discretionary", 
    10 : "Energy" 
    }
  return sector_names

def sector_names_mapping_rus():
    sector_names = {
    0 : "Information Technology",
    1 : "Financials",
    2 : "Health Care",
    3 : "Industrials",
    4 : "Utilities", 
    5 : "Real Estate", 
    6 : "Materials", 
    7 : "Telecommunication Services", 
    8 : "Consumer Staples", 
    9 : "Consumer Discretionary", 
    10 : "Energy",
    11 : "Cash and/or Derivatives"
    }
    return sector_names
# def sp500_ticker_sector_mapping():
#     ticker_sector = {'CTL': 7, 'COTY': 8, 'DISCK': 9, 'AIV': 5, 'FRT': 5, 'CF': 6, 'CA': 0, 'MHK': 9, 'FLIR': 0, 'GS': 1, 'HD': 9, 'AMD': 0, 'MAA': 5, 'KR': 8, 'RTN': 3, 'MKC/V': 8, 'BLL': 6, 'UAA': 9, 'CCI': 5, 'IRM': 5, 'NVDA': 0, 'SYMC': 0, 'DUK': 4, 'IBM': 0, 'LNC': 1, 'LMT/WD': 3, 'EBAY': 0, 'EXC': 4, 'ARNC': 3, 'MON': 6, 'TAP/A': 8, 'PXD': 10, 'FL': 9, 'USB': 1, 'TIF': 9, 'GGP': 5, 'KSS': 9, 'CTXS': 0, 'RMDCD': 2, 'CAH': 2, 'MCO': 1, 'PH': 3, 'IP': 6, 'HRS': 0, 'SYY': 8, 'BIIB': 2, 'TWX': 9, 'MS': 1, 'UDR': 5, 'WMT': 8, 'OMC': 9, 'NFX': 10, 'NRG': 4, 'KHC': 8, 'SPGI': 1, 'AXP': 1, 'TRV': 1, 'ZION': 1, 'PKG': 6, 'HON-W': 3, 'WBA': 8, 'EVHC': 2, 'NCLH': 9, 'ITW': 3, 'FB': 0, 'D': 4, 'EA': 0, 'CAG-W': 8, 'CTSH': 0, 'COO': 2, 'NSC': 3, 'JNJ': 2, 'XRX-W': 0, 'DGX': 2, 'DHR-W': 2, 'BA/': 3, 'SPG': 5, 'IFF': 6, 'PM': 8, 'GE': 3, 'ABC': 2, 'ETR': 4, 'ABBV': 2, 'JEC': 3, 'ROST': 9, 'CINF': 1, 'NWS': 9, 'INFO': 3, 'MTB': 1, 'SBAC': 5, 'ADM': 8, 'STZ/B': 8, 'SWKS': 0, 'AAPL': 0, 'GT': 9, 'HRL': 8, 'NUE': 6, 'INCY': 2, 'KMX': 9, 'CB': 1, 'MCD': 9, 'GOOG': 0, 'PPG': 6, 'F': 9, 'TSN': 8, 'MRO': 10, 'V': 0, 'DFS': 1, 'UHS': 2, 'SCG': 4, 'O': 5, 'EOG': 10, 'BBT': 1, 'AVGO': 0, 'XEC': 10, 'AMAT': 0, 'LLL': 3, 'MDT': 2, 'RHI': 3, 'ES': 4, 'ROK': 3, 'TMK': 1, 'TGT': 9, 'EXR': 5, 'EL': 8, 'GLW': 0, 'REG': 5, 'GIS': 8, 'ROP': 3, 'BMY': 2, 'ADBE': 0, 'GPS': 9, 'CI': 2, 'HUM': 2, 'ECL': 6, 'APD-W': 6, 'BRK.B': 1, 'MCK': 2, 'RJF': 1, 'PX': 6, 'KO': 8, 'CBS': 9, 'VAR-W': 2, 'TROW': 1, 'WYNN': 9, 'KEY': 1, 'YUM-W': 9, 'KMB': 8, 'RHT': 0, 'CNC': 2, 'DISCA': 9, 'APC': 10, 'SLG': 5, 'HSIC': 2, 'ACN': 0, 'MU': 0, 'ANTM': 2, 'ATVI': 0, 'IDXX': 2, 'DOV': 3, 'T': 7, 'XEL': 4, 'CDNS': 0, 'K': 8, 'NOC': 3, 'FOXA': 9, 'PPL': 4, 'CMG': 9, 'PNW': 4, 'PSX': 10, 'ABT': 2, 'SIG': 9, 'QCOM': 0, 'CHD': 8, 'AMG': 1, 'ORLY': 9, 'WY': 5, 'PLD': 5, 'PRGO': 2, 'GILD': 2, 'NTRS': 1, 'HAS': 9, 'XOM': 10, 'CMI': 3, 'GPC': 9, 'ALL': 1, 'JWN': 9, 'FTI': 10, 'SNPS': 0, 'LH': 2, 'FISV': 0, 'MAT': 9, 'DVN': 10, 'AEE': 4, 'STT': 1, 'GOOGL': 0, 'AAP': 9, 'NLSN': 3, 'PHM': 9, 'ALB': 6, 'UNM': 1, 'BDX': 2, 'JCI-W': 3, 'CMA': 1, 'PRU': 1, 'PCLN': 9, 'SLB': 10, 'PVH': 9, 'ADSK': 0, 'DVA': 2, 'ZBH': 2, 'AAL': 3, 'MLM': 6, 'MAC': 5, 'XRAY': 2, 'DHI': 9, 'IT': 0, 'IR': 3, 'MYL': 2, 'VTR': 5, 'CHRW': 3, 'CXO': 10, 'WLTW': 1, 'MAR': 9, 'AMGN': 2, 'LUK-W': 1, 'XLNX': 0, 'NFLX': 0, 'UAL': 3, 'PKI': 2, 'ALXN': 2, 'HCP-W': 5, 'PAYX': 0, 'COL': 3, 'LEG': 9, 'MDLZ': 8, 'LYB': 6, 'INTC': 0, 'UNP': 3, 'STI': 1, 'PEP': 8, 'ILMN': 2, 'DE': 3, 'WM': 3, 'JNPR': 0, 'MOS': 6, 'TDG': 3, 'AZO': 9, 'DIS': 9, 'DTE': 4, 'VLO': 10, 'HAL': 10, 'TEL': 0, 'EIX': 4, 'CMS': 4, 'WU': 0, 'CTAS': 3, 'ISRG': 2, 'RSG': 3, 'PBCT': 1, 'LKQ': 9, 'ORCL': 0, 'TMO': 2, 'SWK': 9, 'DXC': 0, 'BK': 1, 'SBUX': 9, 'XYL': 3, 'WAT': 2, 'TSCO': 9, 'KMI': 10, 'CSCO': 0, 'BHF': 1, 'ALGN': 2, 'DRE': 5, 'ETN': 3, 'CFG': 1, 'XL': 1, 'SHW': 6, 'SYF': 1, 'WDC': 0, 'WFC': 1, 'SO': 4, 'HP': 10, 'NAVI': 1, 'TPR': 9, 'CHK': 10, 'AJG': 1, 'COG': 10, 'ARE': 5, 'AWK': 4, 'DG': 9, 'LRCX': 0, 'HCN': 5, 'APH': 0, 'DPS': 8, 'EMN': 6, 'FFIV': 0, 'INTU': 0, 'LUV': 3, 'HOG': 9, 'AME': 3, 'FLR': 3, 'CSRA': 0, 'HES': 10, 'SJM': 8, 'ALLE': 3, 'UA': 9, 'PFG': 1, 'NI': 4, 'OKE': 10, 'NWL': 9, 'WHR': 9, 'NWSA': 9, 'AMP': 1, 'ANSS': 0, 'CME': 1, 'QRVO': 0, 'COF': 1, 'ULTA': 9, 'EXPD': 3, 'ALK': 3, 'NTAP': 0, 'C': 1, 'NEM': 6, 'MMC': 1, 'APA': 10, 'COP': 10, 'NBL': 10, 'FITB': 1, 'AVB': 5, 'BBY': 9, 'FCX': 6, 'VNO-W': 5, 'CSX': 3, 'FDX': 3, 'PSA': 5, 'PGR': 1, 'PG/WD': 8, 'PNR': 3, 'HBI': 9, 'CVX': 10, 'MCHP': 0, 'CAT': 3, 'LNT': 4, 'RF': 1, 'L': 1, 'FOX': 9, 'PDCO': 2, 'AET': 2, 'BCR': 2, 'CELG': 2, 'MET': 1, 'DWDP': 6, 'BSX': 2, 'EMR': 3, 'LEN/B': 9, 'GPN': 0, 'DLTR': 9, 'RCL': 9, 'DRI': 9, 'TSS': 0, 'FLS': 3, 'GRMN': 9, 'BWA': 9, 'VMC': 6, 'TRIP': 9, 'ESRX': 2, 'MMM': 3, 'OXY': 10, 'BLK': 1, 'BXP': 5, 'SYK': 2, 'DAL': 3, 'BAC': 1, 'CNP': 4, 'FIS': 0, 'PFE': 2, 'ADP': 0, 'AOS': 3, 'DLR': 5, 'EQR': 5, 'LLY': 2, 'MRK': 2, 'AYI': 3, 'WYN': 9, 'CERN': 2, 'LB': 9, 'KIM': 5, 'PCAR': 3, 'KSU': 3, 'AON': 1, 'GM': 9, 'TJX': 9, 'GD': 3, 'MPC/C': 10, 'UTX': 3, 'CHTR': 9, 'APTV': 9, 'VZ': 7, 'HIG': 1, 'VRSK': 3, 'AVY': 6, 'LOW': 9, 'SNI': 9, 'HPE-W': 0, 'CVS': 8, 'EFX': 3, 'MGM': 9, 'HOLX': 2, 'CL': 8, 'SCHW': 1, 'KORS': 9, 'TXN': 0, 'BF.B': 8, 'HRB': 1, 'ADI': 0, 'HST': 5, 'PEG': 4, 'FBHS': 3, 'ED': 4, 'VRSN': 0, 'MAS': 3, 'EW': 2, 'M': 9, 'ADS': 0, 'JBHT': 3, 'CBOE': 1, 'NEE': 4, 'UNH': 2, 'RRC': 10, 'IVZ': 1, 'BEN': 1, 'HCA': 2, 'VIAB': 9, 'MA': 0, 'IPG': 9, 'VFC': 9, 'AKAM': 0, 'HSY': 8, 'CMCSA': 9, 'CBG': 5, 'ICE': 1, 'PWR': 3, 'DISH': 9, 'AFL': 1, 'COST': 8, 'WEC': 4, 'NKE': 9, 'AMZN': 9, 'PNC': 1, 'NDAQ': 1, 'REGN': 2, 'URI': 3, 'SNA': 9, 'CCL': 9, 'MTD': 2, 'SRCL': 3, 'AMT': 5, 'HLT': 9, 'FE': 4, 'A': 2, 'SEE': 6, 'CPB': 8, 'MSFT': 0, 'KLAC': 0, 'ANDV': 10, 'PCG': 4, 'AES': 4, 'EXPE': 9, 'WMB': 10, 'FTV': 3, 'PYPL': 0, 'ETFC': 1, 'WRK-W': 6, 'ZTS': 2, 'HPQ': 0, 'MO': 8, 'CRM': 0, 'FMC': 6, 'ESS': 5, 'BHGE': 10, 'STX': 0, 'RL': 9, 'NOV': 10, 'AIZ': 1, 'FAST': 3, 'AIG': 1, 'TXT': 3, 'VRTX': 2, 'AGN': 2, 'BAX': 2, 'EQT': 10, 'CLX': 8, 'IQV': 2, 'EQIX': 5, 'RE': 1, 'JPM': 1, 'SRE': 4, 'GWW': 3, 'HBAN': 1, 'AEP': 4, 'UPS': 3, 'MSI': 0, 'MNST': 8}
#     return ticker_sector

def rus_growth_ticker_sector_mapping():
    ticker_sector = {'ITRI': 0, 'TTEC': 0, 'KTWO': 2, 'CHFN': 1, 'XONE': 3, 'GLDD': 3, 'NGHC': 1, 'BLKB': 0, 'HRTG': 1, 'EVRI': 0, 'SRPT': 2, 'MGEE': 4, 'GTN': 9, 'HOMB': 1, 'XNCR': 2, 'NAV': 3, 'ILG': 9, 'LNTH': 2, 'FBK': 1, 'CRDB': 1, 'CORE': 9, 'ATRI': 2, 'UVE': 1, 'BSTC': 2, 'BMI': 0, 'NKTR': 2, 'ONCE': 2, 'MLHR': 3, 'CAKE': 9, 'KPTI': 2, 'MYE': 6, 'NLNK': 2, 'BIG': 9, 'CROX': 9, 'SUPN': 2, 'PLNT': 9, 'OPTN': 2, 'VGR': 8, 'HCHC': 3, 'BJRI': 9, 'FRGI': 9, 'XOXO': 0, 'CHUBK': 0, 'EXTR': 0, 'TPRE': 1, 'EMKR': 0, 'CWT': 4, 'SRT': 0, 'COLB': 1, 'LLNW': 0, 'WEB': 0, 'QTM': 0, 'COLL': 2, 'FLDM': 2, 'NX': 3, 'GST': 10, 'NUVA': 2, 'CTLT': 2, 'UNB': 1, 'PSMT': 8, 'UPL': 10, 'MCB': 1, 'JBSS': 8, 'COTV': 2, 'PLSE': 2, 'HIL': 3, 'WVE': 2, 'RGR': 9, 'OBLN': 2, 'YEXT': 0, 'PXLW': 0, 'NSTG': 2, 'SOI': 10, 'BOLD': 2, 'LTC': 5, 'TR': 8, 'TWOU': 0, 'HLNE': 1, 'GWRS': 4, 'GRUB': 0, 'BCO': 3, 'DIN': 9, 'MHO': 9, 'NSIT': 0, 'NTRI': 9, 'DDD': 0, 'SXT': 6, 'IMMR': 0, 'MNR': 5, 'YORW': 4, 'HSKA': 2, 'SND': 10, 'HIIQ': 1, 'PVBC': 1, 'NFBK': 1, 'ARCB': 3, 'OCLR': 0, 'SAMG': 1, 'COWN': 1, 'CASS': 0, 'HIVE': 0, 'MOGA': 3, 'CWH': 9, 'HEES': 3, 'BCC': 6, 'NVEE': 3, 'HALO': 2, 'ADES': 6, 'SXI': 3, 'RRD': 3, 'WLB': 10, 'EBS': 2, 'WLDN': 3, 'BLD': 9, 'TREE': 1, 'VIAV': 0, 'FLOW': 3, 'SUM': 6, 'AIMT': 2, 'RDFN': 5, 'TSE': 6, 'AMBA': 0, 'WTW': 9, 'DLTH': 9, 'REXR': 5, 'INOV': 2, 'ABAX': 2, 'CSII': 2, 'LMAT': 2, 'STRL': 3, 'AT': 4, 'KALA': 2, 'AGX': 3, 'AGEN': 2, 'GPX': 3, 'GBL': 1, 'AXGN': 2, 'LITE': 0, 'TPC': 3, 'HBMD': 1, 'XBIT': 2, 'SELB': 2, 'CW': 3, 'GHDX': 2, 'LHCG': 2, 'PEGI': 4, 'PCH': 5, 'SLP': 2, 'EGHT': 0, 'MPX': 9, 'CVRS': 2, 'PTGX': 2, 'SNHY': 3, 'KIN': 2, 'NRCIA': 2, 'APAM': 1, 'USNA': 8, 'LTS': 1, 'INO': 2, 'LSCC': 0, 'HSTM': 2, 'VRNT': 0, 'GEO': 5, 'SSTK': 0, 'BRSS': 3, 'CYBE': 0, 'CBM': 2, 'OLBK': 1, 'OLLI': 9, 'TRUE': 0, 'LNDC': 8, 'DIOD': 0, 'MGI': 0, 'IRTC': 2, 'LIND': 9, 'OFED': 1, 'XPER': 0, 'PAYC': 0, 'OKTA': 0, 'PATK': 3, 'PAHC': 2, 'CTMX': 2, 'DORM': 9, 'CVA': 3, 'CLPR': 5, 'EVI': 3, 'TWNK': 8, 'SSD': 3, 'TRHC': 2, 'NBHC': 1, 'RNET': 10, 'KERX': 2, 'HUBS': 0, 'EYE': 9, 'KOP': 6, 'ASMB': 2, 'GMS': 3, 'AQ': 0, 'FOE': 6, 'OCX': 2, 'SEND': 0, 'ROIC': 5, 'CTO': 5, 'SHLO': 9, 'LL': 9, 'IPCC': 1, 'AAOI': 0, 'CHEF': 8, 'WK': 0, 'LTXB': 1, 'MMSI': 2, 'VHC': 0, 'MATX': 3, 'RLI': 1, 'VREX': 2, 'GEN': 2, 'CENTA': 8, 'WWW': 9, 'UBNT': 0, 'RIGL': 2, 'APPF': 0, 'EROS': 9, 'BPMC': 2, 'HOFT': 9, 'REV': 8, 'STMP': 0, 'AXDX': 2, 'SSYS': 0, 'ELF': 8, 'VERI': 0, 'TILE': 3, 'EBIX': 0, 'BIOS': 2, 'BCPC': 6, 'JILL': 9, 'SPAR': 3, 'KBAL': 3, 'DBD': 0, 'FNGN': 1, 'CHCT': 5, 'PRSC': 2, 'ROSE': 10, 'CRBP': 2, 'REI': 10, 'QADA': 0, 'SMCI': 0, 'UE': 5, 'CCS': 9, 'TRUP': 1, 'BPI': 9, 'MLAB': 0, 'ADUS': 2, 'MLNT': 2, 'RMTI': 2, 'CWST': 3, 'HTLD': 3, 'GTS': 2, 'OFIX': 2, 'AEIS': 0, 'FWRD': 3, 'MTDR': 10, 'NPO': 3, 'KIDS': 2, 'SGH': 0, 'STC': 1, 'HRI': 3, 'IMMU': 2, 'NXST': 9, 'MCFT': 9, 'HLI': 1, 'CPF': 1, 'MODN': 0, 'KRA': 6, 'CTRL': 0, 'DERM': 2, 'MTSI': 0, 'SPSC': 0, 'BLMN': 9, 'BREW': 8, 'COHU': 0, 'UTMD': 2, 'ZGNX': 2, 'INAP': 0, 'FLWS': 9, 'PVAC': 10, 'GBNK': 1, 'CHGG': 9, 'CRMT': 9, 'IART': 2, 'EXPO': 3, 'PRAH': 2, 'PFGC': 8, 'RARX': 2, 'TNC': 3, 'WWE': 9, 'HOME': 9, 'RP': 0, 'MC': 1, 'ATKR': 3, 'HIFS': 1, 'HONE': 1, 'MSEX': 4, 'WETF': 1, 'DHIL': 1, 'ASGN': 3, 'NYNY': 9, 'ADSW': 3, 'SAFE': 5, 'PLOW': 3, 'MRSN': 2, 'NSA': 5, 'CPLA': 9, 'PNK': 9, 'NATH': 9, 'HELE': 9, 'FSS': 3, 'MWA': 3, 'WING': 9, 'PTLA': 2, 'TMHC': 9, 'SFLY': 9, 'AXAS': 10, 'FRPT': 8, 'WRE': 5, 'MACK': 2, 'FTK': 6, 'PLUG': 3, 'INSM': 2, 'TAX': 9, 'SLAB': 0, 'PQG': 6, 'DXPE': 3, 'MB': 0, 'AIN': 3, 'AQUA': 4, 'DSKE': 3, 'EPM': 10, 'EVTC': 0, 'AAXN': 3, 'WAGE': 3, 'ADXS': 2, 'PTCT': 2, 'WMS': 3, 'CYTK': 2, 'HDSN': 3, 'AFH': 1, 'GDOT': 1, 'CRCM': 0, 'KBH': 9, 'HAIR': 2, 'SAIA': 3, 'GTHX': 2, 'PCYG': 0, 'RMBS': 0, 'TELL': 10, 'BBSI': 3, 'RAVN': 3, 'FBM': 3, 'SAIC': 0, 'BL': 0, 'PHX': 10, 'SNX': 0, 'ITI': 0, 'ESPR': 2, 'CORI': 2, 'ADRO': 2, 'LOXO': 2, 'ICHR': 0, 'KEM': 0, 'LJPC': 2, 'MG': 3, 'ISRL': 10, 'TTEK': 3, 'FUL': 6, 'JELD': 3, 'FMI': 2, 'MLP': 5, 'FIVN': 0, 'LMNR': 8, 'SSB': 1, 'CSGS': 0, 'IMGN': 2, 'IRWD': 2, 'ENSG': 2, 'SWM': 6, 'COUP': 0, 'RTYH8': 11, 'EHTH': 1, 'ZYNE': 2, 'CUTR': 2, 'RVLT': 3, 'PRTK': 2, 'MGLN': 2, 'GNBC': 1, 'MRLN': 1, 'NSSC': 0, 'IIIN': 3, 'SCS': 3, 'GFF': 3, 'PGTI': 3, 'KTOS': 3, 'RVNC': 2, 'BY': 1, 'ALGT': 3, 'LNN': 3, 'BRKS': 0, 'LORL': 9, 'ANIK': 2, 'ABCD': 9, 'MKSI': 0, 'SN': 10, 'LDL': 3, 'SLCA': 10, 'MTNB': 2, 'SGMS': 9, 'PFBC': 1, 'IDTI': 0, 'ALX': 5, 'JCOM': 0, 'OPB': 1, 'CASH': 1, 'RHP': 5, 'CAR': 3, 'ACXM': 0, 'ARRY': 2, 'QTWO': 0, 'AMKR': 0, 'EEX': 9, 'ASPS': 5, 'HAWK': 0, 'COLM': 9, 'WD': 1, 'NVCR': 2, 'VRNS': 0, 'ZIXI': 0, 'DFIN': 1, 'VEC': 3, 'FFIN': 1, 'CNCE': 2, 'POWI': 0, 'CVNA': 9, 'MLI': 3, 'ENV': 0, 'TLRD': 9, 'QTNT': 2, 'CRUS': 0, 'NVAX': 2, 'MBIN': 1, 'MRCY': 3, 'COBZ': 1, 'NYT': 9, 'FORM': 0, 'MGEN': 2, 'MTH': 9, 'NHTC': 8, 'EDGE': 2, 'EFII': 0, 'ZIOP': 2, 'MULE': 0, 'MHLD': 1, 'ULH': 3, 'OMCL': 2, 'EBSB': 1, 'CULP': 9, 'AWR': 4, 'LXRX': 2, 'CSTR': 1, 'CATM': 0, 'NPK': 3, 'VHI': 6, 'WLH': 9, 'GCBC': 1, 'PCRX': 2, 'SGC': 9, 'KOPN': 0, 'CHUY': 9, 'WNC': 3, 'REPH': 2, 'AMOT': 3, 'TGTX': 2, 'DS': 9, 'RGCO': 4, 'PFPT': 0, 'MSTR': 0, 'CSU': 2, 'DMRC': 0, 'RSYS': 0, 'CPK': 4, 'VVI': 3, 'ENTG': 0, 'HSC': 3, 'PSB': 5, 'NEO': 2, 'GEF': 6, 'BMTC': 1, 'MJCO': 0, 'IDCC': 0, 'IIVI': 0, 'MATW': 3, 'AVXS': 2, 'GVA': 3, 'OMAM': 1, 'NP': 6, 'VRTS': 1, 'CDE': 6, 'SPXC': 3, 'MCRB': 2, 'MEI': 0, 'BECN': 3, 'FLIC': 1, 'UIHC': 1, 'NVTA': 2, 'EPAM': 0, 'APOG': 3, 'SHOO': 9, 'SGRY': 2, 'VAC': 9, 'NSP': 3, 'SRDX': 2, 'BNFT': 0, 'APTI': 0, 'GOGO': 0, 'CSWI': 3, 'SEAS': 9, 'PSDO': 0, 'HRG': 8, 'CHE': 2, 'ONVO': 2, 'OMNT': 0, 'CFMS': 2, 'ORA': 4, 'WSBF': 1, 'RGEN': 2, 'VDSI': 0, 'IMDZ': 2, 'AFAM': 2, 'POL': 6, 'AJRD': 3, 'FOLD': 2, 'CLSD': 2, 'JONE': 10, 'SRCI': 10, 'BATRA': 9, 'PENN': 9, 'AZPN': 0, 'NEOS': 2, 'DEPO': 2, 'NCOM': 1, 'FFNW': 1, 'AMED': 2, 'FRAC': 10, 'KNX': 3, 'JBT': 3, 'EXXI': 10, 'FARO': 0, 'CIO': 5, 'BCOV': 0, 'VSAT': 0, 'RDUS': 2, 'SMTC': 0, 'GNMK': 2, 'PLUS': 0, 'MSA': 3, 'TNTR': 0, 'CNS': 1, 'BLUE': 2, 'TISI': 3, 'IBP': 9, 'USLM': 6, 'RUSHB': 3, 'TTD': 0, 'AMBR': 0, 'QTS': 5, 'HABT': 9, 'UEIC': 9, 'MNOV': 2, 'TRNC': 9, 'DECK': 9, 'BGC': 3, 'PLT': 0, 'VCYT': 2, 'PRAA': 1, 'BLKFDS': 11, 'PSTG': 0, 'UFPI': 3, 'GPRO': 9, 'RST': 0, 'HCI': 1, 'FATE': 2, 'SYNT': 0, 'DRRX': 2, 'KODK': 0, 'ATRC': 2, 'IPAR': 8, 'MDLY': 1, 'LFUS': 0, 'PCTY': 0, 'ELLI': 0, 'LQ': 9, 'III': 0, 'KWR': 6, 'ICUI': 2, 'LWAY': 8, 'VIVO': 2, 'NTNX': 0, 'PBH': 2, 'ACRS': 2, 'SBOW': 10, 'XCRA': 0, 'FIX': 3, 'LGIH': 9, 'AKTS': 0, 'CSV': 9, 'PETQ': 2, 'MYRG': 3, 'BLDR': 3, 'WTBA': 1, 'PMTS': 0, 'FSCT': 0, 'WTI': 10, 'CLVS': 2, 'WMGI': 2, 'USCR': 6, 'RLGT': 3, 'LLEX': 10, 'PAY': 0, 'KAMN': 3, 'EPAY': 0, 'ACLS': 0, 'ISTR': 1, 'SAGE': 2, 'BRC': 3, 'WLFC': 3, 'PZN': 1, 'MDCO': 2, 'ROG': 0, 'BHVN': 2, 'OCUL': 2, 'ZAGG': 9, 'ABM': 3, 'TPHS': 5, 'SYX': 0, 'FR': 5, 'TRNO': 5, 'GEFB': 6, 'AMN': 2, 'RRR': 9, 'INSY': 2, 'MTZ': 3, 'ARA': 2, 'TLGT': 2, 'KURA': 2, 'TPB': 8, 'JACK': 9, 'MPWR': 0, 'SBBP': 2, 'KBR': 3, 'YRCW': 3, 'MCS': 9, 'ENTL': 2, 'CSOD': 0, 'CSLT': 2, 'ABCB': 1, 'WNEB': 1, 'EXLS': 0, 'YELP': 0, 'HRTX': 2, 'ITIC': 1, 'RTIX': 2, 'ADMS': 2, 'WWD': 3, 'CLFD': 0, 'RM': 1, 'ALNA': 2, 'CRVL': 2, 'LKFN': 1, 'TRVN': 2, 'TSC': 1, 'ANAB': 2, 'PEN': 2, 'AERI': 2, 'REN': 10, 'PRIM': 3, 'SNDX': 2, 'NTB': 1, 'GKOS': 2, 'RMR': 5, 'SGYP': 2, 'UMH': 5, 'BMCH': 3, 'ATEN': 0, 'CIR': 3, 'SYNH': 2, 'HCKT': 0, 'TWLO': 0, 'THRM': 9, 'GNTY': 1, 'EHC': 2, 'TDOC': 2, 'FC': 3, 'SNDR': 3, 'CZR': 9, 'BHBK': 1, 'PETS': 9, 'HF': 5, 'SWX': 4, 'CSTE': 3, 'MSL': 1, 'CHUBA': 0, 'USPH': 2, 'VCRA': 2, 'NCSM': 10, 'KAI': 3, 'PLAY': 9, 'LZB': 9, 'EVR': 1, 'CELC': 2, 'BLMT': 1, 'UPLD': 0, 'CUDA': 0, 'PJC': 1, 'CALD': 0, 'BGSF': 3, 'RGNX': 2, 'RARE': 2, 'AYX': 0, 'DVAX': 2, 'FBNK': 1, 'CARO': 1, 'ARNA': 2, 'WINA': 9, 'VBIV': 2, 'TCMD': 2, 'QUOT': 0, 'HMSY': 2, 'EVC': 9, 'ECOM': 0, 'AXON': 2, 'APLS': 2, 'MMI': 5, 'NJR': 4, 'FONR': 2, 'CNAT': 2, 'AVID': 0, 'UCTT': 0, 'AQMS': 3, 'CPRX': 2, 'NDLS': 9, 'AMSWA': 0, 'RUTH': 9, 'FIZZ': 8, 'QLYS': 0, 'AVXL': 2, 'BCRX': 2, 'ABG': 9, 'PRTY': 9, 'SYRS': 2, 'PRI': 1, 'AAT': 5, 'WDFC': 8, 'CMP': 6, 'DCPH': 2, 'XLRN': 2, 'HOV': 9, 'IDRA': 2, 'ACBI': 1, 'FOXF': 9, 'FBIO': 2, 'SMP': 9, 'IRBT': 9, 'PEGA': 0, 'ITG': 1, 'ENVA': 1, 'LGND': 2, 'AIT': 3, 'RUSHA': 3, 'UIS': 0, 'EGBN': 1, 'SPA': 3, 'BEAT': 2, 'TVPT': 0, 'WATT': 3, 'MBUU': 9, 'NHI': 5, 'USAT': 0, 'BLBD': 3, 'CEVA': 0, 'RDI': 9, 'CRZO': 10, 'JNCE': 2, 'BABY': 2, 'TTGT': 0, 'AOBC': 9, 'NMIH': 1, 'MITK': 0, 'DAKT': 0, 'MDGL': 2, 'TBPH': 2, 'MNRO': 9, 'MDXG': 2, 'VIVE': 2, 'BELFB': 0, 'USD': 11, 'DF': 8, 'PGEM': 3, 'VALU': 1, 'BID': 9, 'NGVT': 6, 'DY': 3, 'FIVE': 9, 'SBGI': 9, 'QDEL': 2, 'MXL': 0, 'RYAM': 6, 'EVBG': 0, 'MOH': 2, 'BCOR': 0, 'FRBK': 1, 'PIRS': 2, 'TREX': 3, 'INGN': 2, 'NXTM': 2, 'TEN': 9, 'AST': 2, 'FRED': 9, 'SHLD': 9, 'AOSL': 0, 'FFWM': 1, 'GPT': 5, 'INST': 0, 'NTRA': 2, 'AMPH': 2, 'TNET': 3, 'CCXI': 2, 'SYNA': 0, 'SPKE': 4, 'ANCX': 1, 'RBB': 1, 'RH': 9, 'MYOK': 2, 'CVGW': 8, 'CMPR': 0, 'PODD': 2, 'AKBA': 2, 'LAWS': 3, 'BYD': 9, 'AKAO': 2, 'MEET': 0, 'RNG': 0, 'SFBS': 1, 'EAT': 9, 'FNKO': 9, 'PDFS': 0, 'ETSY': 0, 'PUB': 1, 'ZOES': 9, 'LPSN': 0, 'EDIT': 2, 'VIRT': 1, 'GERN': 2, 'OFLX': 3, 'QTNA': 0, 'BOJA': 9, 'PRMW': 8, 'CHDN': 9, 'PI': 0, 'PACB': 2, 'WOR': 6, 'PICO': 9, 'LMNX': 2, 'VSAR': 2, 'SCMP': 2, 'XENT': 2, 'CRY': 2, 'WOW': 9, 'TCBI': 1, 'ATNX': 2, 'EQBK': 1, 'REVG': 3, 'ELVT': 1, 'FLXN': 2, 'RPD': 0, 'WHG': 1, 'ESNT': 1, 'JJSF': 8, 'CDXS': 6, 'MLR': 3, 'GRPN': 9, 'DENN': 9, 'ASIX': 6, 'AIMC': 3, 'CPS': 9, 'FCPT': 5, 'TVTY': 2, 'PETX': 2, 'TYPE': 0, 'INWK': 3, 'DEL': 6, 'FNSR': 0, 'CMD': 2, 'LPX': 6, 'ALRM': 0, 'PCMI': 0, 'GTT': 0, 'RTEC': 0, 'COKE': 8, 'REIS': 0, 'PRO': 0, 'EGP': 5, 'VRAY': 2, 'NOVT': 0, 'ALRN': 2, 'NVRO': 2, 'TCX': 0, 'OMN': 6, 'VTVT': 2, 'CLCT': 9, 'CVGI': 3, 'ATRS': 2, 'PARR': 10, 'OSIS': 0, 'PRGS': 0, 'RYTM': 2, 'MVIS': 0, 'VRTU': 0, 'ETM': 9, 'INVA': 2, 'BSF': 1, 'NANO': 0, 'EGRX': 2, 'ALDR': 2, 'LOPE': 9, 'MED': 8, 'HAE': 2, 'CIVI': 2, 'EVH': 2, 'INSE': 9, 'GNRC': 3, 'CNOB': 1, 'HTBK': 1, 'HZN': 9, 'STRA': 9, 'TECD': 0, 'CCF': 6, 'FCFS': 1, 'NXRT': 5, 'MMS': 0, 'RCM': 2, 'CENX': 6, 'CALA': 2, 'CPSI': 2, 'GBCI': 1, 'KRO': 6, 'TTMI': 0, 'LC': 1, 'ENT': 9, 'BOX': 0, 'DLX': 3, 'SHAK': 9, 'ERI': 9, 'RDNT': 2, 'CAMP': 0, 'KFRC': 3, 'SPRO': 2, 'BGS': 8, 'B': 3, 'HWKN': 6, 'HCSG': 3, 'BFS': 5, 'THC': 2, 'GMRE': 5, 'RICK': 9, 'UEC': 10, 'HBP': 3, 'SONC': 9, 'PUMP': 10, 'GBT': 2, 'BWFG': 1, 'HQY': 2, 'STAA': 2, 'MBFI': 1, 'MTX': 6, 'QUAD': 3, 'ROX': 8, 'OCN': 1, 'OXFD': 2, 'HMTV': 9, 'SAM': 8, 'NLS': 9, 'DOC': 5, 'TXMD': 2, 'PRFT': 0, 'CCMP': 0, 'MDCA': 9, 'ASC': 10, 'ATSG': 3, 'CORT': 2, 'VRS': 6, 'CCC': 6, 'TPIC': 3, 'LAD': 9, 'WGO': 9, 'STML': 2, 'KW': 5, 'RETA': 2, 'HDP': 0, 'MRT': 5, 'EGOV': 0, 'BLCM': 2, 'KMG': 6, 'KNSL': 1, 'CVLT': 0, 'PRLB': 3, 'CBPX': 3, 'CARB': 0, 'LABL': 3, 'ABTX': 1, 'FRAN': 9, 'JAG': 10, 'PZZA': 9, 'MDC': 9, 'AKCA': 2, 'AZZ': 3, 'SREV': 0, 'CAI': 3, 'WTS': 3, 'CBTX': 1, 'NNBR': 3, 'VBTX': 1, 'DOOR': 3, 'LANC': 8, 'MCRN': 3, 'IMAX': 9, 'PRTA': 2, 'SHLM': 6, 'PLCE': 9, 'SITE': 3, 'CLXT': 2, 'HI': 3, 'SGMO': 2, 'FELE': 3, 'CTRE': 5, 'ACIW': 0, 'IVAC': 0, 'VNDA': 2, 'OVID': 2, 'ZEN': 0, 'EIGI': 0, 'ECOL': 3, 'HMHC': 9, 'SMBC': 1, 'MSBI': 1, 'WSFS': 1, 'KNL': 3, 'BOFI': 1, 'CIEN': 0, 'IOVA': 2, 'TXRH': 9, 'TBI': 3, 'NEWR': 0, 'KS': 6, 'NYMX': 2, 'OMER': 2, 'HZO': 9, 'OSTK': 9, 'ANIP': 2, 'ATU': 3, 'DAN': 9, 'AHH': 5, 'SPWH': 9, 'PGNX': 2, 'ASTE': 3, 'PPBI': 1, 'NVEC': 0, 'CCRN': 2, 'UHT': 5, 'CARA': 2, 'OSUR': 2, 'BATRK': 9, 'FRTA': 6, 'GMED': 2, 'FN': 0, 'AAON': 3, 'HNI': 3, 'FORR': 3, 'HA': 3, 'FGEN': 2, 'GDEN': 9, 'ERII': 3, 'PCYO': 4, 'CHRS': 2, 'VICR': 3, 'ACOR': 2, 'ENZ': 2, 'CLDR': 0, 'QSII': 2, 'MASI': 2, 'LCII': 9, 'IPHI': 0, 'NERV': 2, 'SEM': 2, 'PBYI': 2, 'OXM': 9, 'TNAV': 0, 'CBRL': 9, 'SNBR': 9, 'KLDX': 6, 'AMWD': 3, 'RRGB': 9, 'ARAY': 2, 'BCEI': 10, 'ACIA': 0, 'TOCA': 2, 'SNNA': 2, 'SP': 3, 'DYAX': 2, 'ENS': 3, 'GNCA': 2, 'CENT': 8, 'NCS': 3, 'MNTA': 2, 'FSB': 1, 'TPH': 9, 'GLUU': 0, 'KMT': 3, 'ELGX': 2, 'DOVA': 2, 'SBRA': 5, 'ATHX': 2, 'ALG': 3, 'CVCO': 9, 'FICO': 0, 'FMSA': 10, 'ROLL': 3, 'PBPB': 9, 'MOBL': 0, 'TTS': 9, 'RNGR': 10, 'SYKE': 0, 'NEOG': 2, 'ORN': 3, 'MGPI': 8, 'CERS': 2, 'EPZM': 2, 'MGNX': 2, 'HCCI': 3, 'MSFUT': 11, 'ATRO': 3, 'LOB': 1, 'EXAS': 2, 'HY': 3, 'MDSO': 2, 'SCL': 6, 'TMP': 1, 'CRIS': 2, 'EME': 3, 'CVI': 10, 'WTTR': 10, 'MGRC': 3, 'IMPV': 0}
    return ticker_sector


def sp500_ticker_sector_mapping():
    ticker_sector = {'CTL': 7, 'COTY': 8, 'DISCK': 9, 'AIV': 5, 'FRT': 5, 'CF': 6, 'CA': 0, 'MHK': 9, 'FLIR': 0, 'GS': 1, 'HD': 9, 'AMD': 0, 'MAA': 5, 'KR': 8, 'RTN': 3, 'MKC': 8, 'BLL': 6, 'UAA': 9, 'CCI': 5, 'IRM': 5, 'NVDA': 0, 'SYMC': 0, 'DUK': 4, 'IBM': 0, 'LNC': 1, 'LMT': 3, 'EBAY': 0, 'EXC': 4, 'ARNC': 3, 'MON': 6, 'TAP': 8, 'PXD': 10, 'FL': 9, 'USB': 1, 'TIF': 9, 'GGP': 5, 'KSS': 9, 'CTXS': 0, 'RMD': 2, 'CAH': 2, 'MCO': 1, 'PH': 3, 'IP': 6, 'HRS': 0, 'SYY': 8, 'BIIB': 2, 'TWX': 9, 'MS': 1, 'UDR': 5, 'WMT': 8, 'OMC': 9, 'NFX': 10, 'NRG': 4, 'KHC': 8, 'SPGI': 1, 'AXP': 1, 'TRV': 1, 'ZION': 1, 'PKG': 6, 'HON': 3, 'WBA': 8, 'EVHC': 2, 'NCLH': 9, 'ITW': 3, 'FB': 0, 'D': 4, 'EA': 0, 'CAG': 8, 'CTSH': 0, 'COO': 2, 'NSC': 3, 'JNJ': 2, 'XRX': 0, 'DGX': 2, 'DHR': 2, 'BA/': 3, 'SPG': 5, 'IFF': 6, 'PM': 8, 'GE': 3, 'ABC': 2, 'ETR': 4, 'ABBV': 2, 'JEC': 3, 'ROST': 9, 'CINF': 1, 'NWS': 9, 'INFO': 3, 'MTB': 1, 'SBAC': 5, 'ADM': 8, 'STZ': 8, 'SWKS': 0, 'AAPL': 0, 'GT': 9, 'HRL': 8, 'NUE': 6, 'INCY': 2, 'KMX': 9, 'CB': 1, 'MCD': 9, 'GOOG': 0, 'PPG': 6, 'F': 9, 'TSN': 8, 'MRO': 10, 'V': 0, 'DFS': 1, 'UHS': 2, 'SCG': 4, 'O': 5, 'EOG': 10, 'BBT': 1, 'AVGO': 0, 'XEC': 10, 'AMAT': 0, 'LLL': 3, 'MDT': 2, 'RHI': 3, 'ES': 4, 'ROK': 3, 'TMK': 1, 'TGT': 9, 'EXR': 5, 'EL': 8, 'GLW': 0, 'REG': 5, 'GIS': 8, 'ROP': 3, 'BMY': 2, 'ADBE': 0, 'GPS': 9, 'CI': 2, 'HUM': 2, 'ECL': 6, 'APD': 6, 'BRK.B': 1, 'MCK': 2, 'RJF': 1, 'PX': 6, 'KO': 8, 'CBS': 9, 'VAR': 2, 'TROW': 1, 'WYNN': 9, 'KEY': 1, 'YUM': 9, 'KMB': 8, 'RHT': 0, 'CNC': 2, 'DISCA': 9, 'APC': 10, 'SLG': 5, 'HSIC': 2, 'ACN': 0, 'MU': 0, 'ANTM': 2, 'ATVI': 0, 'IDXX': 2, 'DOV': 3, 'T': 7, 'XEL': 4, 'CDNS': 0, 'K': 8, 'NOC': 3, 'FOXA': 9, 'PPL': 4, 'CMG': 9, 'PNW': 4, 'PSX': 10, 'ABT': 2, 'SIG': 9, 'QCOM': 0, 'CHD': 8, 'AMG': 1, 'ORLY': 9, 'WY': 5, 'PLD': 5, 'PRGO': 2, 'GILD': 2, 'NTRS': 1, 'HAS': 9, 'XOM': 10, 'CMI': 3, 'GPC': 9, 'ALL': 1, 'JWN': 9, 'FTI': 10, 'SNPS': 0, 'LH': 2, 'FISV': 0, 'MAT': 9, 'DVN': 10, 'AEE': 4, 'STT': 1, 'GOOGL': 0, 'AAP': 9, 'NLSN': 3, 'PHM': 9, 'ALB': 6, 'UNM': 1, 'BDX': 2, 'JCI': 3, 'CMA': 1, 'PRU': 1, 'PCLN': 9, 'SLB': 10, 'PVH': 9, 'ADSK': 0, 'DVA': 2, 'ZBH': 2, 'AAL': 3, 'MLM': 6, 'MAC': 5, 'XRAY': 2, 'DHI': 9, 'IT': 0, 'IR': 3, 'MYL': 2, 'VTR': 5, 'CHRW': 3, 'CXO': 10, 'WLTW': 1, 'MAR': 9, 'AMGN': 2, 'LUK': 1, 'XLNX': 0, 'NFLX': 0, 'UAL': 3, 'PKI': 2, 'ALXN': 2, 'HCP': 5, 'PAYX': 0, 'COL': 3, 'LEG': 9, 'MDLZ': 8, 'LYB': 6, 'INTC': 0, 'UNP': 3, 'STI': 1, 'PEP': 8, 'ILMN': 2, 'DE': 3, 'WM': 3, 'JNPR': 0, 'MOS': 6, 'TDG': 3, 'AZO': 9, 'DIS': 9, 'DTE': 4, 'VLO': 10, 'HAL': 10, 'TEL': 0, 'EIX': 4, 'CMS': 4, 'WU': 0, 'CTAS': 3, 'ISRG': 2, 'RSG': 3, 'PBCT': 1, 'LKQ': 9, 'ORCL': 0, 'TMO': 2, 'SWK': 9, 'DXC': 0, 'BK': 1, 'SBUX': 9, 'XYL': 3, 'WAT': 2, 'TSCO': 9, 'KMI': 10, 'CSCO': 0, 'BHF': 1, 'ALGN': 2, 'DRE': 5, 'ETN': 3, 'CFG': 1, 'XL': 1, 'SHW': 6, 'SYF': 1, 'WDC': 0, 'WFC': 1, 'SO': 4, 'HP': 10, 'NAVI': 1, 'TPR': 9, 'CHK': 10, 'AJG': 1, 'COG': 10, 'ARE': 5, 'AWK': 4, 'DG': 9, 'LRCX': 0, 'HCN': 5, 'APH': 0, 'DPS': 8, 'EMN': 6, 'FFIV': 0, 'INTU': 0, 'LUV': 3, 'HOG': 9, 'AME': 3, 'FLR': 3, 'CSRA': 0, 'HES': 10, 'SJM': 8, 'ALLE': 3, 'UA': 9, 'PFG': 1, 'NI': 4, 'OKE': 10, 'NWL': 9, 'WHR': 9, 'NWSA': 9, 'AMP': 1, 'ANSS': 0, 'CME': 1, 'QRVO': 0, 'COF': 1, 'ULTA': 9, 'EXPD': 3, 'ALK': 3, 'NTAP': 0, 'C': 1, 'NEM': 6, 'MMC': 1, 'APA': 10, 'COP': 10, 'NBL': 10, 'FITB': 1, 'AVB': 5, 'BBY': 9, 'FCX': 6, 'VNO': 5, 'CSX': 3, 'FDX': 3, 'PSA': 5, 'PGR': 1, 'PG': 8, 'PNR': 3, 'HBI': 9, 'CVX': 10, 'MCHP': 0, 'CAT': 3, 'LNT': 4, 'RF': 1, 'L': 1, 'FOX': 9, 'PDCO': 2, 'AET': 2, 'BCR': 2, 'CELG': 2, 'MET': 1, 'DWDP': 6, 'BSX': 2, 'EMR': 3, 'LEN': 9, 'GPN': 0, 'DLTR': 9, 'RCL': 9, 'DRI': 9, 'TSS': 0, 'FLS': 3, 'GRMN': 9, 'BWA': 9, 'VMC': 6, 'TRIP': 9, 'ESRX': 2, 'MMM': 3, 'OXY': 10, 'BLK': 1, 'BXP': 5, 'SYK': 2, 'DAL': 3, 'BAC': 1, 'CNP': 4, 'FIS': 0, 'PFE': 2, 'ADP': 0, 'AOS': 3, 'DLR': 5, 'EQR': 5, 'LLY': 2, 'MRK': 2, 'AYI': 3, 'WYN': 9, 'CERN': 2, 'LB': 9, 'KIM': 5, 'PCAR': 3, 'KSU': 3, 'AON': 1, 'GM': 9, 'TJX': 9, 'GD': 3, 'MPC': 10, 'UTX': 3, 'CHTR': 9, 'APTV': 9, 'VZ': 7, 'HIG': 1, 'VRSK': 3, 'AVY': 6, 'LOW': 9, 'SNI': 9, 'HPE': 0, 'CVS': 8, 'EFX': 3, 'MGM': 9, 'HOLX': 2, 'CL': 8, 'SCHW': 1, 'KORS': 9, 'TXN': 0, 'BF.B': 8, 'HRB': 1, 'ADI': 0, 'HST': 5, 'PEG': 4, 'FBHS': 3, 'ED': 4, 'VRSN': 0, 'MAS': 3, 'EW': 2, 'M': 9, 'ADS': 0, 'JBHT': 3, 'CBOE': 1, 'NEE': 4, 'UNH': 2, 'RRC': 10, 'IVZ': 1, 'BEN': 1, 'HCA': 2, 'VIAB': 9, 'MA': 0, 'IPG': 9, 'VFC': 9, 'AKAM': 0, 'HSY': 8, 'CMCSA': 9, 'CBG': 5, 'ICE': 1, 'PWR': 3, 'DISH': 9, 'AFL': 1, 'COST': 8, 'WEC': 4, 'NKE': 9, 'AMZN': 9, 'PNC': 1, 'NDAQ': 1, 'REGN': 2, 'URI': 3, 'SNA': 9, 'CCL': 9, 'MTD': 2, 'SRCL': 3, 'AMT': 5, 'HLT': 9, 'FE': 4, 'A': 2, 'SEE': 6, 'CPB': 8, 'MSFT': 0, 'KLAC': 0, 'ANDV': 10, 'PCG': 4, 'AES': 4, 'EXPE': 9, 'WMB': 10, 'FTV': 3, 'PYPL': 0, 'ETFC': 1, 'WRK': 6, 'ZTS': 2, 'HPQ': 0, 'MO': 8, 'CRM': 0, 'FMC': 6, 'ESS': 5, 'BHGE': 10, 'STX': 0, 'RL': 9, 'NOV': 10, 'AIZ': 1, 'FAST': 3, 'AIG': 1, 'TXT': 3, 'VRTX': 2, 'AGN': 2, 'BA': 2, 'EQT': 10, 'CLX': 8, 'IQV': 2, 'EQIX': 5, 'RE': 1, 'JPM': 1, 'SRE': 4, 'GWW': 3, 'HBAN': 1, 'AEP': 4, 'UPS': 3, 'MSI': 0, 'MNST': 8}
    return ticker_sector


 # "Assets ['LUK-W', 'YUM-W', 'TAP/A', 'JCI-W', 'XRX-W', 'VAR-W', 'CAG-W', 'VNO-W', 'RMDCD', 'LEN/B', 'BA/', 'PG/WD', 'HCP-W', 'STZ/B', 'MKC/V', 'APD-W', 'LMT/WD', 'DHR-W', 'HPE-W', 'WRK-W', 'HON-W', 'MPC/C'] not in group mapping"

def select_industry_from_sp500_wiki_for_io(sector_str):
    sp500_df = pd.read_csv('sp500_csv_list.csv')
    sector_rows = sp500_df[sp500_df['GICS Sector'] == sector_str]
    sector_list = list(sector_rows['Ticker symbol'])
    sector_list_df = pd.DataFrame(sector_list, columns = ['Ticker'])
    return sector_list_df



# ===================================================================================================================
# 2020-11-16
# Process data - last step: replace all "" with nan
fin_data = fin_data.replace(r'', np.NaN)


# ===================================================================================================================
# 2020-11-17
# 日线数据转化为周线数据
# -*- coding: utf-8 -*-
import pandas
period_type = "W"

stock_data = pd.read_csv(data_location+"000001.SZ.csv")
period_stock_data = stock_data.resample(period_type, how="last")





