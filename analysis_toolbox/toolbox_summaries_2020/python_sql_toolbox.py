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
# 5-02-2019 updated
# 链接在本地电脑上的某个sql数据库需要的python代码和信息：
conn = psycopg2.connect(database="brad_particle", user=postgres, password=311***, host="127.0.0.1",
                                port="5432")

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


























