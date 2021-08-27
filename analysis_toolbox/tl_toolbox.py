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
# 10-23-2019
# 从表里一行行读取数据，再写入到另一张表里
test_list = []

module_num = 0
with open('20191023_test.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        if int(row[0])>module_num:
            module_num = int(row[0])
        else:
            joined = "".join(row[1:])
            print(joined)
            test_list.append("".join(row[1:]))
            
test_list_ = [(i,) for i in test_list]
with open('test_to_csv.csv','w',newline='') as f_:
    f_csv_ = csv.writer(f_)
    f_csv_.writerows(test_list_)


# ===================================================================================================================
# 10-17-2019
# 用difflib对比两个文本相似度
import difflib #python 自带库，不需额外安装
 
In [49]: test1
Out[49]: ['你好', '我是谁']
 
In [50]: test2
Out[50]: ['你好啊', '我谁']
 
In [51]: test3
Out[51]: [12, 'nihao']
 
In [52]: test4
Out[52]: ['你好', 'woshi']
 
In [53]: difflib.SequenceMatcher(a=test1, b=test2).quick_ratio()
Out[53]: 0.0
 
In [54]: difflib.SequenceMatcher(a=test1, b=test4).ratio()
Out[54]: 0.5


# ===================================================================================================================
# 05-23-2019
# 使用字典嵌套：
import collections
tree = lambda:collections.defaultdict(tree)
articles_dict=tree()
for keyword in keywords_list:
    for i, item in enumerate(splitted_T[keyword]):
        sub_dict_value = ...
        # 循环创建子字典：
        sub_dict[splitted_T[keyword][i]] = sub_dict_value
        # 循环将子字典套入到母字典中：
        articles_dict[keyword][splitted_T[keyword][i]] = sub_dict[splitted_T[keyword][i]]
articles_dict_ = dict(articles_dict)


# ===================================================================================================================
# 05-21-2019
# 如何避免写for循环：
# https://blog.csdn.net/wonengguwozai/article/details/78295484

# assert断言：
# python assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。可以理解assert断言语句为raise-if-not，
# 用来测试表示式，其返回值为假，就会触发异常。
# https://www.cnblogs.com/liuchunxiao83/p/5298016.html

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
# 5-08-2019 updated
"""
模式	可做操作	若文件不存在	是否覆盖
r	    只能读	      报错	       -
r+	  可读可写	   报错	        是
w	    只能写	      创建	       是
w+　	 可读可写	    创建	       是
a　　 只能写	      创建	   否，追加写
a+	  可读可写	   创建	    否，追加写
"""
# ===================================================================================================================
# 5-02-2019 updated
# 链接在本地电脑上的某个sql数据库需要的python代码和信息：
conn = psycopg2.connect(database="brad_particle", user=postgres, password=311***, host="127.0.0.1",
                                port="5432")



# ===================================================================================================================
# 4-23-2019 updated
file_path = file_path.replace("\\", "/")

# ===================================================================================================================
# 4-11-2019 updated
# 两个文件夹之间的文件做对比，若满足xx条件，则将一个文件夹的某个文件复制粘贴到目标文件夹之中

all_path = "C:\\Users\\workspace\\brad_public_workspace_on_win\\non_code_files_brad_public_workspace_on_win\\brad_public_workspace_on_win_non_code_files\\SH_tongliang\\reports\\百度流量项目\\2-20-2019文章自动化Brad_to_Sha\\4-9-2019-compose\\4-10-2019-4416篇/*.docx"
write_files = "C:/Users/Brad Sun/Desktop/write_files/"
read_files = "C:/Users/Brad Sun/Desktop/4-10-2019-4416篇/"
all_list_desktop = glob(all_path)

def copy_paste(read_file, write_file):
    content = open(read_file, 'rb').read()
    open(write_file, 'wb').write(content)
    
count_ = 0
for index in selected_index_list:
    for title in all_list_desktop:
        title_name = title.split("\\")[-1]
        title_name_number = int(title_name.split("_")[0])
        if index == title_name_number:
            count_+=1
            write_path = write_files + title_name 
            read_path = read_files + title_name
            try:
                copy_paste(read_path, write_path)
            except FileNotFoundError:
                print ("Error title: ", title_name)
#             print ("title_name", title_name)
#             print ("write_path", write_path)
#             print ("read_path", read_path)
print (count_)


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
# 统计中的排列组合：
def fac(n):
    factorial = 1
    if n == 0:
         factorial = 0
    else:
        for i in range(1, n + 1):
            factorial *= i
#             print(factorial)
    return factorial

def C_m_n(m, n):
    combo_num = fac(m)/(fac(m-n)*fac(n))
    return combo_num
    # 从m里挑选n个
    
def A_m_n(m, n):
    combo_num = fac(m)/fac(n)
    return combo_num


# ===================================================================================================================
# 4-2-2019 updated
# 在一个序列中进行factorial式的两两运算（在这里为比较两个字符串的相似度），并记录(append到list里)生成记录，便于后期生成dataframe：
test_df = pd.DataFrame(["test121121", "test121223", "test121123", "test123124", "test121215", "test123126"])
test_df.columns = ['articles']
start = 0
ratio_list_j = []
first_index = []
first_items = []
second_items = []

# 对于比较目标一进行循环：
for i, item in enumerate(test_df['articles']):
#     ratio_list_j_ = []
    # 确定特定的比较目标一之后，分别将比较目标一和比较目标二做循环运算处理：
    for j in range(i+1, len(test_df)):
        item_i = test_df['articles'][i]
        item_j = test_df['articles'][j]
        ratio = apply_difflib(item_i, item_j)
        ratio_list_j.append(ratio)
        print ("Index", start, item_i, item_j)
        print ("Repitition Rate", ratio)
        start = start+1
        print ("====")
#         max_ratio = np.max(ratio_list_j_)
        first_index.append(i)
        first_items.append(item_i)
        second_items.append(item_j)
    print (first_index)
    print ("ratio_list_j", ratio_list_j)
    print ("MAX", max_ratio)
    print ("++++")



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
# 3-27-2019 updated
# 使用os.getcwd()调用当前path
import os, sys
import codecs
current_path = os.getcwd()
text_file = current_path+"/readme.txt"
f = codecs.open(text_file, "r+", "utf-8")
s = f.read()
print (current_path)
print (s)


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
# 3-9-2019 updated
# 直接进入到存储libs python文件的folder之中，调用各种modules:
lib_path = r'C:/Users/workspace/brad_public_workspace_on_win/brad_public_workspace_on_win/backtester/brad_vbacktester/'
file_path = 'C:/Users/workspace/brad_public_workspace_on_win/brad_public_workspace_on_win/backtester/brad_vbacktester/'
j9000_file = "j9000_hr.csv"
jm000_file = "jm000_hr.csv"

from sys import path # 调用path module
path.append(lib_path) # 将存放modules的路径添加进来

from metrics import expected_returns, win_loss_rate
from performance import sngl_performance
from data import process_sngl_data
from plot import plot_cum_returns
from strategies import Strat_SMA
from indicators import MACD


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
# 3-7-2019 udpated
# loop through all csv files in a folder.
import glob
path = "path/to/dir/*.csv"
for fname in glob.glob(path):
    print(fname)


# ===================================================================================================================
# 2-25-2019 udpated
print("{:.2f}".format(number)) # 两位小数


# ===================================================================================================================
# 2-25-2019 udpated
# 直接锁定需要调用函数的文件夹：
from sys import path
path.append(r'C:\Users\workspace\brad_public_workspace_on_win\lib\test_brad_bt') #将存放module的路径添加进来
import ml_all_in_one # as an example


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
# 2-1-2019 udpated
# 随机森林单品种回测商品期货，AIO函数——研究参数random_state，参数优化一条龙：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.style
plt.style.use("ggplot")
%matplotlib inline

import sys
sys.version


location =  "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/commodities_data/12_28_commodities_daily/"
file = "j9000_d.csv"
exported_signal_file = "j9000_d_testing_signals.csv"
n = 10
test_size = 1/6


def preprocess_df(location, file):
    df = pd.read_csv(location+file, engine="python", header=None)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interests']
    return df

def get_indicators(data, n, indicator):
    
    ###### Step 1: Calculate necessary time series ######
    up, dw = data['close'].diff(), -data['close'].diff()
    up[up<0], dw[dw<0] = 0, 0
    # default set to be 12-period ema as the fast line, 26 as the slow line:
    macd = data['close'].ewm(12).mean() - data['close'].ewm(26).mean()
    # default set to be 9-period ema of the macd
    macd_signal = macd.ewm(9).mean()
    
    ###### Step 2: Create dataframe and fill with technical indicators: ######
    indicators = pd.DataFrame(data=0, index=data.index,
                              columns=['sma', 'ema', 'momentum', 'rsi', 'macd'])
#     indicators['date'] = data['date']
    indicators['sma'] = data['close'].rolling(n).mean()
    indicators['ema'] = data['close'].ewm(n).mean()
    indicators['momentum'] = data['close'] - data['close'].shift(n)
    indicators['rsi'] = 100 - 100 / (1 + up.rolling(n).mean() / dw.rolling(n).mean())
    indicators['macd'] = macd - macd_signal
    indicators.index = data['date']
    return indicators[[indicator]]

def get_data(df, n):
    # technical indicators
    sma = get_indicators(df, n, 'sma')
    ema = get_indicators(df, n, 'ema')
    momentum = get_indicators(df, n, 'momentum')
    rsi = get_indicators(df, n, 'rsi')
    macd = get_indicators(df, n, 'macd')
    tech_ind = pd.concat([sma, ema, momentum, rsi, macd], axis = 1)
    df.index = df['date']
    close = df['close']
    direction = (close > close.shift()).astype(int)
    target = direction.shift(-1).fillna(0).astype(int)
    target.name = 'target'
    master_df = pd.concat([tech_ind, close, target], axis=1)
    return master_df

def rebalance(unbalanced_data, rblnc_rs):
    # Sampling should always be done on train dataset: https://datascience.stackexchange.com/questions/32818/train-test-split-of-unbalanced-dataset-classification
    # Separate majority and minority classes
    if unbalanced_data.target.value_counts()[0]>unbalanced_data.target.value_counts()[1]:
        print ("majority:0, length: {}; minority:1, length: {}".format(unbalanced_data.target.value_counts()[0],unbalanced_data.target.value_counts()[1]))
        data_minority = unbalanced_data[unbalanced_data.target==1] 
        data_majority = unbalanced_data[unbalanced_data.target==0] 
    else:
        print ("majority:1, length: {}; minority:0, length: {}".format(unbalanced_data.target.value_counts()[1],unbalanced_data.target.value_counts()[0]))
        data_minority = unbalanced_data[unbalanced_data.target==0] 
        data_majority = unbalanced_data[unbalanced_data.target==1] 
    # Upsample minority class
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=rblnc_rs)
    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    data_upsampled.sort_index(inplace=True)
    # Display new class counts
    data_upsampled.target.value_counts()
    return data_upsampled

def normalize(x):
    scaler = StandardScaler()
    # 公式为：(X-mean)/std  计算时对每个属性/每列分别进行。
    # 将数据按期属性（按列进行）减去其均值，并除以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)
    return x_norm

def scores(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    
    
def train_test_validate(master_df, train_start, train_end, test_size, tts_rs, rblnc_rs, plot=True): 
    # train_start example: '2011-01-01'
    # test_size defaults as 1/6, 
    # test_size: parameter
    
    data = master_df.copy()
    data.index = pd.to_datetime(data.index)
    if plot == True:
        print ("Plotting data's close price series")
        ax = data[['close']].plot(figsize=(20, 5))
        ax.set_ylabel("Price (￥)")
        ax.set_xlabel("Time")
        plt.show()
    else:
        pass
    data_train = data[train_start : train_end]
    # Sampling should always be done on train dataset: https://datascience.stackexchange.com/questions/32818/train-test-split-of-unbalanced-dataset-classification
    data_train = rebalance(data_train, rblnc_rs).dropna()
    # y as the label target 
    y = data_train.target
    # X as the dataframe with their values to be normalized
    X = data_train.drop('target', axis=1)
    X = normalize(X)
    
    data_val = data[train_end:]
    data_val.dropna(inplace=True)
    # y_val as the label target in the validation period
    y_val = data_val.target
    # X_val as the dataframe with their values to be normalized in the validation period
    X_val = data_val.drop('target', axis=1)
    # normalize X_val dataframe
    X_val = normalize(X_val)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = tts_rs)
    print ("-----------------------------------------------")
    print ("X length: ", len(X))
    print ("X_val length: ", len(X_val))
    print ("X_train length: ", len(X_train))
    print ("X_test length: ", len(X_test))
    print ("-----------------------------------------------")
    print ("y length: ", len(y))
    print ("y_val length: ", len(y_val))
    print ("y_train length:", len(y_train))
    print ("y_test length:", len(y_test))
    print ("-----------------------------------------------")
    # Outputs of this function are 8 variables from above.
    return data, X, X_val, X_train, X_test, y, y_val, y_train, y_test
    
    
def optimize_model_paras(X_train, y_train, X_test, y_test):
    # first take a look at the default model's results:
    model = RandomForestClassifier(random_state=5)
    print ("Training default model...")
    model.fit(X_train, y_train)
    print ("Default model's scores:")
    scores(model, X_test, y_test)
    # set up parameters to be optimized
    grid_data =   {'n_estimators': [10, 50, 100],
                   'criterion': ['gini', 'entropy'],
                   'max_depth': [None, 10, 50, 100],
                   'min_samples_split': [2, 5, 10],
                   'random_state': [1]}
    grid = GridSearchCV(model, grid_data, scoring='f1').fit(X_train, y_train)
    print ("-----------------------------------------------")
    print ("Model's best parameters: ")
    print(grid.best_params_)
    model = grid.best_estimator_
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print("Performance of the train_test datasets: ")
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    print ("----------------------------------------------------")
    print ("Optimized Model from the train_test dataset: ", model)
    
    # Validate optimized model:
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    print ("----------------------------------------------------")
    
    optimized_model = model
    return optimized_model

def train_test_backtest(optimized_model, X, y, X_train, y_train):
    rf_model = optimized_model
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print("train_test datasets performance: ")
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    print ("----------------------------------------------------")
    mask = y_pred.copy()
    np.place(mask, y_pred==0, -1)
    mask = np.roll(mask, 1)
    data_returns = data['close'].diff()
    data_returns = data_returns[X.index]
    model_returns = mask * data_returns
    model_cum = model_returns.cumsum()
    equity = model_returns.sum()
    start_close = data["close"][X.index[0]]
    performance = equity / start_close * 100
#     ax = model_returns.plot(figsize=(15, 8))
#     ax.set_ylabel("Returns (￥)")
#     ax.set_xlabel("Time")
#     plt.show()
    ax = model_cum.plot(figsize=(15, 8))
    ax.set_ylabel("Cummulative returns (￥)")
    ax.set_xlabel("Time")
    plt.show()
    return model_cum, equity, performance, mask, y_pred, data_returns


# Trading system: testing real performance:
def validate_backtest(optimized_model, X_val, y_val, X_train, y_train):
    rf_model = optimized_model
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    print("validation datasets performance: ")
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    print ("----------------------------------------------------")
    mask = y_pred.copy()
    np.place(mask, y_pred==0, -1)
    mask = np.roll(mask, 1)
    data_returns = data['close'].diff()
    data_returns = data_returns[X_val.index]
    model_returns = mask * data_returns
    model_cum = model_returns.cumsum()
    equity = model_returns.sum()
    start_close = data["close"][X_val.index[0]]
    performance = equity / start_close * 100
#     ax = model_returns.plot(figsize=(15, 8))
#     ax.set_ylabel("Returns ($)")
#     ax.set_xlabel("Time")
#     plt.show()
    ax = model_cum.plot(figsize=(15, 8))
    ax.set_ylabel("Cummulative returns ($)")
    ax.set_xlabel("Time")
    plt.show()
#     print (pd.DataFrame(model_cum)) # 对了
    return model_cum, equity, performance, mask, y_pred, data_returns

# Create signal file that is to be imported to TB:
def create_TB_signal_df(df, X_val, y_pred, y_val, mask, data_returns, exported_file):
    print ("Processing signal dataframe...")
    master_pred_df = X_val.copy()
    master_pred_df['y_pred'] = y_pred
    master_pred_df['y_val'] = y_val
    master_pred_df['mask'] = mask
    master_pred_df['data_returns'] = data_returns
    master_pred_df['model_returns'] = mask * data_returns
    master_pred_df_dt = master_pred_df.copy()
    master_pred_df_dt.reset_index(inplace = True)
    
    print ("Processing original OHLCV dataframe...")
    df_dt = df.copy()
    del df_dt['date']
    df_dt.reset_index(inplace= True)
    df_dt['date'] = pd.to_datetime(df_dt['date'])

    print ("Merging signal dataframe and OHLCV dataframe...")
    master_pred_df_dt = master_pred_df_dt[['date', 'mask']]
    merged = df_dt[['date', 'open', 'high', 'low', 'close']].merge(master_pred_df_dt, on = 'date')
    merged.columns = ['date', 'open', 'high', 'low', 'close', 'signals']
    
    print ("Exporting final signal file...")
#     merged.to_csv(location + exported_file, index = False, header = False)
    print ("All done!")
    
    return merged, master_pred_df



rblnc_rs = [1,5]
tts_rs = [1,5]
RFC_rs = [1,5]

backtest_records = {
                    'rblnc_rs':[],
                    'tts_rs':[],
                    'RFC_rs':[],
                    'cum_returns':[]}

def RF_rs_loop_AIO(rblnc_rs, tts_rs, RFC_rs):

    # Part 1:
    df = preprocess_df(location, file)

    # Part 2:
    master_df = get_data(df, n)

    # Part 3:
    data, X, X_val, X_train, X_test, y, y_val, y_train, y_test = train_test_validate(master_df, '2011-01-01','2017-01-01', test_size, tts_rs, rblnc_rs, False)

    # Part 4: if we already have all optimized parameters we just run this step: 
    optmzd_model = RandomForestClassifier(
                bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=RFC_rs, verbose=0, warm_start=False
                )

    # Part 5: See in-sample backtest
    model_cum, equity, performance, mask, y_pred, data_returns = train_test_backtest(optmzd_model, X, y, X_train, y_train)

    # Part 6: See out-of-sample backtest
    model_cum_, equity_, performance_, mask_, y_pred_, data_returns_ = validate_backtest(optmzd_model, X_val, y_val, X_train, y_train)
#     print (pd.DataFrame(model_cum_))  # 已解决
    return model_cum_
    


backtest_curves = pd.DataFrame([])
for rs_1 in rblnc_rs:
    for rs_2 in tts_rs:
        for rs_3 in RFC_rs:
            model_cum_ = RF_rs_loop_AIO(rs_1, rs_2, rs_3)
            print ("rblnc_rs: ", rs_1)
            print ("tts_rs: ", rs_2)
            print ("RFC_rs: ", rs_3)
#             backtest_records['rblnc_rs'].append(rs_1)
#             backtest_records['tts_rs'].append(rs_2)
#             backtest_records['RFC_rs'].append(rs_3)
            print ("=============================================All Finished.==================================================")
            print ("model_cum_: ", pd.DataFrame(model_cum_).head(3))
            backtest_curves = pd.concat([backtest_curves, pd.DataFrame(model_cum_)], axis=1)

backtest_curves.plot(figsize=(16, 8))








# ===================================================================================================================
# 12_28_2018_将TB下下来的期货数据统一合并到一起并分别加上资产名称
# Create a function to streamline this process for future use, but: 
# 1) Specifically for the format of file name: e.g. a9000_d.csv
# 2) Specially for the downloaded data files from TB

def read_and_add_tickers(file):
    df = pd.read_csv(file, header=None, sep = ',')
    ticker = file.split("\\",1)[1].split(".",1)[0].split("_",1)[0]
    df['ticker'] = ticker
    return df

# all_csv_files format example:"C:/Users/12_28_commodities_daily/*.csv"
def concat_files_inside_folders(all_csv_files):
    files = glob.glob(all_csv_files)
    dfs = [read_and_add_tickers(file) for file in files]
    futuresdata = pd.concat(dfs,ignore_index=True)
    futuresdata.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'holdings', 'ticker']
    return futuresdata



# ===================================================================================================================
# 12_17_2018_计算最大未创新高时长
def AIO_get_max_down_dur(root, file):
    # delete #Bar column in the file before putting root and file into the function.
    df = pd.read_csv(root+file, engine="python")
    
    df.columns = ['time', 'long_margin', 'short_margin', 'capital_available', 'floating_equity', 'trading_costs', 'static_equity', 'accum_returns']
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)

    equity = df.copy()
    equity.index = equity['time']
    equity = equity[['floating_equity']]
    
    """权益最长未创新高的持续时间"""
    logger.debug('---equity in analysis---: {}'.format(equity))
    max_equity = 0
    duration = pd.Timedelta(0)  # 时间间隔为 0
    date_list = []
    date_dur = pd.DataFrame(columns=['duration', 'start', 'end'])
    for i in range(equity.shape[0]):
        if max_equity <= equity.values[i][0]: #
            max_equity = equity.values[i][0]
            date_list.append(equity.index[i])
    logger.debug('---date_list---: {}'.format(date_list))
    for j in range(len(date_list) - 1):  # len()-1
        duration_ = date_list[j + 1] - date_list[j]

        date_dur = date_dur.append(pd.Series(
            [duration_, date_list[j], date_list[j + 1]], index=['duration', 'start', 'end']), ignore_index=True)

    date_dur = date_dur.sort_values('duration')
    start_date = date_dur.iloc[-1]['start']
    if equity.iloc[-1].values <= max_equity:
        deltta = equity.index[-1] - date_list[-1]
        start_date = date_list[-1]
        end_date = equity.index[-1]
    else:
        end_date = date_dur.iloc[-1]['end']
    date = start_date.strftime('%Y-%m-%d') + ' - ' + \
        end_date.strftime('%Y-%m-%d')
    logger.debug('---date in analysis---: {}'.format(date))
#     return date
    return date_dur, equity




# ===================================================================================================================
# 12_13_2018: 使用tushare
import tushare as ts
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

ts.set_token("2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67")
pro = ts.pro_api()


# ===================================================================================================================
# 12_12_2018: 计算未创新高的天数以及对应的日期，并导出排名


def duration_of_equity_not_reaching_high(equity):
    """权益最长未创新高的持续时间"""
    # equity = context.fill.equity.df
    logger.debug('---equity in analysis---: {}'.format(equity))
    max_equity = 0
    duration = pd.Timedelta(0)  # 时间间隔为 0
    date_list = []
    date_dur = pd.DataFrame(columns=['duration', 'start', 'end'])
    for i in range(equity.shape[0]):
        if max_equity <= equity.values[i][0]: #
            max_equity = equity.values[i][0]
            date_list.append(equity.index[i])
    logger.debug('---date_list---: {}'.format(date_list))
    for j in range(len(date_list) - 1):  # len()-1
        duration_ = date_list[j + 1] - date_list[j]

        date_dur = date_dur.append(pd.Series(
            [duration_, date_list[j], date_list[j + 1]], index=['duration', 'start', 'end']), ignore_index=True)
        #
        # if duration < duration_:
        #     duration = duration_
        #     date_dict[duration] = [date_list[i], date_list[i + 1]]

    # date = date_dict[max(date_dict)][0] + '-' + date_dict[max(date_dict)][1]
    date_dur = date_dur.sort_values('duration')
    start_date = date_dur.iloc[-1]['start']
    if equity.iloc[-1].values <= max_equity:
        deltta = equity.index[-1] - date_list[-1]
        start_date = date_list[-1]
        end_date = equity.index[-1]
    else:
        end_date = date_dur.iloc[-1]['end']
    date = start_date.strftime('%Y-%m-%d') + ' - ' + \
        end_date.strftime('%Y-%m-%d')
    logger.debug('---date in analysis---: {}'.format(date))
#     return date
    return date_dur





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



# ada_df = pd.read_csv(location + "res_12_03_og_all_hundreds_adau18z18_1d.csv", engine="python", header=None)
# bch_df = pd.read_csv(location + "res_12_03_og_all_hundreds_bchu18z18_1d.csv", engine="python", header=None)
# eos_df = pd.read_csv(location + "res_12_03_og_all_hundreds_eosu18z18_1d.csv", engine="python", header=None)
# eth_df = pd.read_csv(location + "res_12_03_og_all_hundreds_ethu18z18_1d.csv", engine="python", header=None)
# ltc_df = pd.read_csv(location + "res_12_03_og_all_hundreds_ltcu18z18_1d.csv", engine="python", header=None)
# trx_df = pd.read_csv(location + "res_12_03_og_all_hundreds_trxu18z18_1d.csv", engine="python", header=None)
# btc_df = pd.read_csv(location + "res_12_03_og_all_hundreds_xbtusd_1d.csv", engine="python", header=None)
# xrp_df = pd.read_csv(location + "res_12_03_og_all_hundreds_xrpu18z18_1d.csv", engine="python", header=None)

# Get close price:

# def return_close(df):
#     df.columns = ['time', 'open', 'high', 'low', 'close', 'vol']
#     df.index = df['time']
#     close = df['close']
#     return close

# ada_close = pd.DataFrame(return_close(ada_df))
# bch_close = pd.DataFrame(return_close(bch_df))
# eos_close = pd.DataFrame(return_close(eos_df))
# eth_close = pd.DataFrame(return_close(eth_df))
# ltc_close = pd.DataFrame(return_close(ltc_df))
# trx_close = pd.DataFrame(return_close(trx_df))
# btc_close = pd.DataFrame(return_close(btc_df))
# xrp_close = pd.DataFrame(return_close(xrp_df))

# ada_close.index = pd.to_datetime(ada_close.index)
# bch_close.index = pd.to_datetime(bch_close.index)
# eos_close.index = pd.to_datetime(eos_close.index)
# eth_close.index = pd.to_datetime(eth_close.index)
# ltc_close.index = pd.to_datetime(ltc_close.index)
# trx_close.index = pd.to_datetime(trx_close.index)
# btc_close.index = pd.to_datetime(btc_close.index)
# xrp_close.index = pd.to_datetime(xrp_close.index)

# ada_close.reset_index(inplace = True)
# bch_close.reset_index(inplace = True)
# eos_close.reset_index(inplace = True)
# eth_close.reset_index(inplace = True)
# ltc_close.reset_index(inplace = True)
# trx_close.reset_index(inplace = True)
# btc_close.reset_index(inplace = True)
# xrp_close.reset_index(inplace = True)

# ===================================================================================================================
# 11_26_资产走势与策略曲线收益关系分析

# 分析一：思路：多空对冲策略回测部分时段走平原因：是否与价格波动率小有关
location = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/11_24_bitmex分钟线_multiplied/bitmex分钟线1124/"

ada_df = pd.read_csv(location+"11_24_4hr_ada.csv", engine="python")
trx_df = pd.read_csv(location+"11_24_4hr_trx.csv", engine="python")
bch_df = pd.read_csv(location+"11_24_4hr_bch.csv", engine="python")
eos_df = pd.read_csv(location+"11_24_4hr_eos.csv", engine="python")
eth_df = pd.read_csv(location+"11_24_4hr_eth.csv", engine="python")
ltc_df = pd.read_csv(location+"11_24_4hr_ltc.csv", engine="python")
xrp_df = pd.read_csv(location+"11_24_4hr_xrp.csv", engine="python")
btc_df = pd.read_csv(location+"11_24_4hr_btc.csv", engine="python")

def delete_unnamed(df):
    del df['Unnamed: 0']
    return df

ada_df = delete_unnamed(ada_df)
trx_df = delete_unnamed(trx_df)
bch_df = delete_unnamed(bch_df)
eos_df = delete_unnamed(eos_df)
eth_df = delete_unnamed(eth_df)
ltc_df = delete_unnamed(ltc_df)
xrp_df = delete_unnamed(xrp_df)
btc_df = delete_unnamed(btc_df)

df = ada_df
time_col = "time"
start_ts = "2018-09-28 00:00:00"
end_ts = "2018-11-02 00:00:00"


def all_period_stddev(df, time_col, backtest_start):
    close = df['close'][df[time_col]>=backtest_start].pct_change()
    stddev = close.std()
    return stddev


def specific_period_stddev(df, time_col, start_ts, end_ts):
    close = df['close'][(df[time_col]>start_ts) & (df[time_col]<end_ts) ].pct_change()
    stddev = close.std()
    return stddev
    
    
def relative_volatility_all_in_one(asst, df, time_col, start_ts, end_ts, backtest_start):
    """
    This "relative_volatility" measures how volatile a specific period of price series is compared to that of all period. 
    """
    specifc_vol = specific_period_stddev(df, time_col, start_ts, end_ts)
    all_vol = all_period_stddev(df, time_col, backtest_start)
    relative_vol = specifc_vol/all_vol
    print ("==============================")
    print (asst)
    print (str((relative_vol)*100)[:5]+"%")
    
    
def relative_volatility_val(asst, df, time_col, start_ts, end_ts, backtest_start):
    """
    This "relative_volatility" measures how volatile a specific period of price series is compared to that of all period. 
    """
    specifc_vol = specific_period_stddev(df, time_col, start_ts, end_ts)
    all_vol = all_period_stddev(df, time_col, backtest_start)
    relative_vol = specifc_vol/all_vol
    return relative_vol


# df = ada_df
time_col = "time"
start_ts = "2018-09-28 00:00:00"
end_ts = "2018-11-02 00:00:00"
backtest_start = "2018-06-29 00:00:00"

relative_volatility_all_in_one("ada", ada_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("bch", bch_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("trx", trx_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("ltc", ltc_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("eos", eos_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("eth", eth_df, time_col, start_ts, end_ts, backtest_start)

ada_rel_vol = relative_volatility_val("ada", ada_df, time_col, start_ts, end_ts, backtest_start)
bch_rel_vol = relative_volatility_val("bch", bch_df, time_col, start_ts, end_ts, backtest_start)
trx_rel_vol = relative_volatility_val("trx", trx_df, time_col, start_ts, end_ts, backtest_start)
ltc_rel_vol = relative_volatility_val("ltc", ltc_df, time_col, start_ts, end_ts, backtest_start)
eos_rel_vol = relative_volatility_val("eos", eos_df, time_col, start_ts, end_ts, backtest_start)
eth_rel_vol = relative_volatility_val("eth", eth_df, time_col, start_ts, end_ts, backtest_start)
avg_rel_vol= (ada_rel_vol+bch_rel_vol+trx_rel_vol+ltc_rel_vol+eos_rel_vol+eth_rel_vol)/6

print ("==============================")
print ("==============================")
print ("avg_relative_volatility_ratio", str(avg_rel_vol*100)[:5]+"%")

# output:
==============================
ada
74.90%
==============================
bch
46.37%
==============================
trx
85.09%
==============================
ltc
80.47%
==============================
eos
53.62%
==============================
eth
71.93%
==============================
==============================
avg_relative_volatility_ratio 68.73%


# 分析二：回测时期多头不赚钱空头赚钱的原因

def return_analysis(asst, df, time_col, backtest_start):
    close = df['close'][df[time_col]>=backtest_start]
    close_start = close.values[0]
    close_end = close.values[-1]
    avg_daily_return = np.mean(close.pct_change())
    total_return = close_end/close_start - 1
    print (asst)
    print ("avg_daily_return: ", avg_daily_return)
    print ("total_return: ", total_return)
    print ("=============================================")

return_analysis('ada', ada_df, time_col, backtest_start)
return_analysis('eth', eth_df, time_col, backtest_start)
return_analysis('bch', bch_df, time_col, backtest_start)
return_analysis('ltc', ltc_df, time_col, backtest_start)
return_analysis('eos', eos_df, time_col, backtest_start)
return_analysis('trx', trx_df, time_col, backtest_start)

# output:
ada
avg_daily_return:  -0.0007073363288827819
total_return:  -0.5249643366619116
=============================================
eth
avg_daily_return:  -0.000985795217863391
total_return:  -0.6098036485169196
=============================================
bch
avg_daily_return:  -0.0007433045821839179
total_return:  -0.5884917175239756
=============================================
ltc
avg_daily_return:  -0.0005882461890809978
total_return:  -0.4394171779141105
=============================================
eos
avg_daily_return:  -0.0003822458312288239
total_return:  -0.356045162302023
=============================================
trx
avg_daily_return:  -0.0006294343200917069
total_return:  -0.4951923076923077
=============================================


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



# ===================================================================================================================
# 直接用以下这些功能从数据库中拿数据，并处理成可以导入TB进行回测的数据价格格式
# 日线回测分析用这个里面的数据，包括用这个导入到TB：
# C:\Users\workspace\SH_tongliang\database\bitmex日线至1106_11_7_2018\bitmex日线至1106\standardized_version\11_7_日线导出数据\导出数据
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import psycopg2

%matplotlib inline



# OG 乘数version: (Finalized Version)

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

    for s in symlist:
        if s == 'ada':
            c = 10000000
        elif s == 'bch':
            c = 1000
        elif s == 'eos':
            c = 100000
        elif s == 'eth' or s == 'ltc':
            c = 10000
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
till_date = "11_30_og_"
frequency = "_1d"
file_suffix = 'u18z18' + frequency
added_note = 'u18z18'

all_assts_from_sql("ada" + file_suffix, "bch" + file_suffix, "eos" + file_suffix,
                   "eth" + file_suffix, "ltc" + file_suffix, "trx" + file_suffix,
                   "xbtusd" + frequency, "xrp" + file_suffix, 1000,
                   "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/extracting_and_transforming_data/",
                   till_date)

reset_price(location, till_date, file_suffix, added_note)

# ada_df = pd.read_csv(location + "res_" + till_date + "ada" + added_note + "10000000.csv", engine="python", header=None)
# bch_df = pd.read_csv(location + "res_" + till_date + "bch" + added_note + "1000.csv", engine="python", header=None)
# eos_df = pd.read_csv(location + "res_" + till_date + "eos" + added_note + "100000.csv", engine="python", header=None)
# eth_df = pd.read_csv(location + "res_" + till_date + "eth" + added_note + "10000.csv", engine="python", header=None)
# ltc_df = pd.read_csv(location + "res_" + till_date + "ltc" + added_note + "10000.csv", engine="python", header=None)
# trx_df = pd.read_csv(location + "res_" + till_date + "trx" + added_note + "100000000.csv", engine="python", header=None)
# xrp_df = pd.read_csv(location + "res_" + till_date + "xrp" + added_note + "10000000.csv", engine="python", header=None)
btc_df = pd.read_csv(location + till_date + "xbtusd" + frequency + ".csv", engine="python", header=None)
btc_df = btc_df.iloc[1:, :]
btc_df.to_csv(location + "res_" + till_date + "xbtusd" + frequency + ".csv",
              sep=',', header=False, index=False, float_format='%.2f')


# 11-27 version that still needs to be fixed:
# Extract all daily digital currency data from SQL database:
def all_assts_from_sql(asst1, asst2, asst3, asst4, asst5, asst6, asst7, asst8, sql_limit_num, location, till_date):
    conn = psycopg2.connect(database="bitmexdata", user="postgres", password="tongKen123", host="128.199.97.202", port="5432")
    asset1 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst1, sql_limit_num)
    asset2 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst2, sql_limit_num)
    asset3 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst3, sql_limit_num)
    asset4 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst4, sql_limit_num)
    asset5 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst5, sql_limit_num)
    asset6 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst6, sql_limit_num)
    asset7 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst7, sql_limit_num)
    asset8 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst8, sql_limit_num)
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
    df1.to_csv(location+"{}".format(till_date)+asst1+".csv", index = False)
    df2.to_csv(location+"{}".format(till_date)+asst2+".csv", index = False)
    df3.to_csv(location+"{}".format(till_date)+asst3+".csv", index = False)
    df4.to_csv(location+"{}".format(till_date)+asst4+".csv", index = False)
    df5.to_csv(location+"{}".format(till_date)+asst5+".csv", index = False)
    df6.to_csv(location+"{}".format(till_date)+asst6+".csv", index = False)
    df7.to_csv(location+"{}".format(till_date)+asst7+".csv", index = False)
    df8.to_csv(location+"{}".format(till_date)+asst8+".csv", index = False)

# standardize and reset the price:
# file_suffix example: 'z18_1d.csv'
# added_note example: 'u18z18乘'
# till_date example: '11_10_' 
def reset_price(location, till_date, file_suffix, added_note):

    symlist = ['ada', 'bch', 'eth', 'eos', 'trx', 'xrp', 'ltc']

    for s in symlist:
        if s == 'ada':
            c = 10000000
        elif s == 'bch':
            c = 10000
        elif s == 'eos':
            c = 1000000
        elif s == 'eth':
            c = 10000
        elif s == 'ltc':
            c = 100000
        elif s == 'trx':
            c = 100000000
        elif s == 'xrp':
            c = 10000000

#         b = pd.read_csv(location + till_date + s + file_suffix,header=None)
        b = pd.read_csv(location + "{}".format(till_date) + s + file_suffix + ".csv")
        b.iloc[:, [1, 2, 3, 4]] = b.iloc[:, [1, 2, 3, 4]].astype(float)
        b.iloc[:, [1, 2, 3, 4]] = (b.iloc[:, [1, 2, 3, 4]]) * c
        b.to_csv(location + 'res_' + till_date + s + added_note + str(c) + '.csv', 
                 sep=',', header=False, index=False, float_format='%.2f')

location = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/extracing_and_transforming_data/"
till_date = "11_27_"
frequency = "_1d"
file_suffix = 'u18z18'+frequency
added_note = 'u18z18乘'

all_assts_from_sql("ada"+file_suffix, "bch"+file_suffix, "eos"+file_suffix, 
                   "eth"+file_suffix, "ltc"+file_suffix, "trx"+file_suffix, 
                   "xbtusd"+frequency, "xrp"+file_suffix, 1000,
                   "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/extracing_and_transforming_data/",
                   till_date)

reset_price(location, till_date, file_suffix, added_note)

ada_df = pd.read_csv(location + "res_"+till_date+"ada"+added_note+"10000000.csv", engine="python", header=None)
bch_df = pd.read_csv(location + "res_"+till_date+"bch"+added_note+"10000.csv", engine="python", header=None)
eos_df = pd.read_csv(location + "res_"+till_date+"eos"+added_note+"1000000.csv", engine="python", header=None)
eth_df = pd.read_csv(location + "res_"+till_date+"eth"+added_note+"10000.csv", engine="python", header=None)
ltc_df = pd.read_csv(location + "res_"+till_date+"ltc"+added_note+"100000.csv", engine="python", header=None)
trx_df = pd.read_csv(location + "res_"+till_date+"trx"+added_note+"100000000.csv", engine="python", header=None)
xrp_df = pd.read_csv(location + "res_"+till_date + "xrp"+added_note+"10000000.csv", engine="python", header=None)
btc_df = pd.read_csv(location + till_date + "xbtusd" +frequency +".csv", engine="python", header = None)
btc_df = btc_df.iloc[1:, :]
btc_df.to_csv(location + "res_" + till_date + "xbtusd"+frequency +".csv", 
                 sep=',', header=False, index=False, float_format='%.2f')
btc_df = pd.read_csv(location + "res_" + till_date + "xbtusd"+frequency +".csv", engine="python", header=None)

# ===================================================================================================================
# Run code to get data on Pycharm to get ranking stats from database
# Refer to 11_7_export_ranking_info.py
# Refer to 11_6_ranking_analysis_part_2_with_functions_to_export_formatted_ranking_csv
def export_formmated_rank(location, rank_file):
    location = location
    rank_file = rank_file
    rank = pd.read_csv(location + rank_file, encoding="latin-1", delimiter=',')
    df = pd.DataFrame(rank.stack()).reset_index()
    del df['level_0']
    df.columns = ['date', 'to_split']
    df['asset'], df['rank'] = df['to_split'].str.split(' ', 1).str
    del df['to_split']
    df['rank'] = df['rank'].apply(lambda x: int(x))
    formatted_df = pd.pivot_table(df, values='rank', columns='asset', index = 'date')
    return formatted_df

# ===================================================================================================================

# A function that analyzes the performance table from TB regarding which assets gain or lose most:
# Need to put the code together and write into functions
# Notebook to refer to: 11_8_Assets_Performance_Analysis_with_ready-to-use-function

# Import raw performance csv file and build into a data table in a ready-to-process format:
def build_perf_df(location, perf_file):
    perf1 = pd.read_csv(location+perf_file)
    del perf1['Unnamed: 0']
    perf1.columns = ['position_direction', 'asset', 'entry_date', 'entry_price', 
                    'exit_date', 'exit_price', 'qty', 'trade_costs', 'net_gains', 
                    'cum_gains', 'returns', 'cum_returns']
    return perf1  

# Process performance table so that it can be used to generate further asset perf analysis:
def process_perf_df(perf1):
    date_fmt = "%Y-%m-%d"
    perf1['exit_dt'] = perf1['exit_date'].apply(lambda x: datetime.strptime(x, date_fmt))
    perf1['entry_dt'] = perf1['entry_date'].apply(lambda x: datetime.strptime(x, date_fmt))

    perf1['holding_days'] = ' '
    for i, item in enumerate(perf1['holding_days']):
        perf1['holding_days'][i] = perf1['exit_dt'][i] - perf1['entry_dt'][i] 
        perf1['holding_days'][i] =  perf1['holding_days'][i].days

    perf1['returns'] = perf1['returns'].apply(lambda x: float(x.replace("%", "")))
    perf1['returns'] = perf1['returns']/100

    perf1['cum_returns'] = perf1['cum_returns'].apply(lambda x: float(x.replace("%", "")))
    perf1['cum_returns'] = perf1['cum_returns']/100
    
    perf2 = perf1.copy()
    return perf2


def perf_df_analysis(perf2, show_minmax_returns = True, show_return_rank = True):
    perf2['worst_return'] = perf2.groupby(['asset'])['returns'].min()
    perf2['best_return'] = perf2.groupby(['asset'])['returns'].max()
    perf2_rank = perf2.copy()[['asset', 'entry_date', 'exit_date', 'returns']].groupby(['asset']).apply(lambda x: x.sort_values('returns'))
    if show_minmax_returns == True and show_return_rank == True:
                                                                                                        
        print ("worst_return: ", perf2['worst_return'])
        print ("best return: ", perf2['best_return'])
        print ("return ranks", perf2_rank)                                                                                     
    if show_minmax_returns == True and show_return_rank == False:
                                                                                                        
        print ("worst_return: ", perf2['worst_return'])
        print ("best return: ", perf2['best_return'])
    if show_minmax_returns == False and show_return_rank == True:
                                                                                                    
        print ("return ranks: ", perf2_rank)
    else:
        return 

# ===================================================================================================================

# Write a function that records the prediction rate:
# the function needs to be wirtten here, unfinished:
# Notebook to refer to: 11_6_ranking_analysis_part_2



# ===================================================================================================================
"""
自动数据库读取数据，计算排序，发送邮件
# py file to refer to: zdsort.py
"""
import numpy as np
import pandas as pd
import datetime
import time
import re
import talib
import operator
import psycopg2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header


# 计算两天close涨幅 T1,T2
def zf(data):
    resdata = []
    for r in data.itertuples():
        time = getattr(r, 'time')
        open = getattr(r, 'open')
        high = getattr(r, 'high')
        low = getattr(r, 'low')
        close = getattr(r, 'close')
        resdata.append([time, open, high, low, close])
    res = [0]
    for i in range(1, len(resdata)):
        T = (float(resdata[i][4]) - float(resdata[i - 1][4])) / float(resdata[i - 1][4]) * 100
        res.append(T)
    return res


# 计算TP和值
def myTP(T1, T2, Q):
    res = [0]

    for i in range(len(T1)):
        t = T1[i] * (1 - Q / 100) + T2[i] * (Q / 100)
        res.append(t)
    return res


# 计算TP差值
def myTM(T1, T2, Q):
    res = [0]
    for i in range(len(T1)):
        t = T1[i] * (1 - Q / 100) - T2[i] * (Q / 100)
        res.append(t)
    return res


# 计算和值累加 或 差值累加
def mySUM(TP):
    res = [0]
    for i in TP:
        res.append(res[-1] + i)
    return res


# 求指定周期均值，参数1数据列表，参数2周期，返回单个数值
def myAverage(lists, tp):
    if len(lists) > tp:
        return np.mean(lists[-tp:])
    else:
        return np.mean(lists)


# 判断趋势
def pdqs(lists):
    count = -1
    while lists[count] == lists[count - 1]:
        count -= 1
    if lists[count] > lists[count - 1]:
        return "趋势向上"
    elif lists[count] < lists[count - 1]:
        return "趋势向下"


# 判断交叉
# def pdjx(ddlist, eelist):
#     for i in range(len(ddlist)):
#         if ddlist[-i]:
#             return "金叉"
#         elif eelist[-i]:
#             return "死叉"
#         else:
#             return '没有交叉'
# 判断交叉
def pdjx(ddlist, eelist):
    for i in range(len(ddlist)):
        if ddlist[-i]:
            return "金叉"
        elif eelist[-i]:
            return "死叉"
        else:
            return '没有交叉'


# 计算结果 参数：数据1，数据2，短周期，长周期，权重1，权重2,截止日期
def comres(data0, data1, S, L, Q1, Q2, starttime, endtime):
    """计算两者排名"""
    pd.to_datetime(data0.ix[:, 0])  # 转换为时间格式
    pd.to_datetime(data1.ix[:, 0])  # 转换为时间格式

    # timestr = '2018-10-19 00:00:00'

    data0 = data0[(data0.ix[:, 0] <= endtime) & (data0.ix[:, 0] >= starttime)]
    data1 = data1[(data1.ix[:, 0] <= endtime) & (data1.ix[:, 0] >= starttime)]
    print('筛选后')
    print(len(data0), len(data1))
    T1 = zf(data0)  # 第一腿涨幅
    T2 = zf(data1)  # 第二腿涨幅

    TP = myTP(T1, T2, Q1)  # 相对和
    TM = myTM(T1, T2, Q2)  # 相对差

    TPSUM = mySUM(TP)  # 相对和累加
    TMSUM = mySUM(TM)  # 相对差累加

    TMS = np.array(TMSUM)
    TPS = np.array(TPSUM)

    MA1 = talib.MA(TMS, S)  # 相对差短周期均线
    MA2 = talib.MA(TMS, L)  # 相对差长周期均线

    # MAS = talib.MA(TPS, S)
    # MAL = talib.MA(TPS, L)

    if MA1[-1] > MA2[-1]:
        jx = '金叉'
    else:
        jx = '死叉'

    MA1qs = pdqs(MA1)
    MA2qs = pdqs(MA2)
    # print(MA2)
    # print("交叉：" + jx)
    # print("短周期相对差累加均线MA1趋势:" + MA1qs)
    # print("长周期相对差累加均线MA2趋势:" + MA2qs)
    if jx == '金叉':
        return 1
    if jx == '死叉':
        return -1


def get_data(conn, sym, period):
    """获取数据"""
    if sym == 'BTC':
        sym = 'xbtusd'
    else:
        sym = sym.lower() + 'u18z18'
    sqllasttime = " SELECT time, open, high, low, close, volume from " + sym + "_" + period + " order by time"
    df = pd.read_sql(sqllasttime, con=conn)
    return df


def send_email(username, password, receiver, text):
    """发送邮件"""
    smtpserver = 'smtp.qq.com'
    sender = username

    subject = '强弱排名'
    subject = Header(subject, 'utf-8').encode()

    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = username
    # msg['To'] = 'XXX@126.com'
    # 收件人为多个收件人,通过join将列表转换为以;为间隔的字符串
    msg['To'] = ";".join(receiver)
    # msg['Date']='2012-3-16'

    text_plain = MIMEText(text, 'plain', 'utf-8')
    msg.attach(text_plain)

    # 发送邮件
    smtp = smtplib.SMTP()
    smtp.connect('smtp.qq.com')
    # 我们用set_debuglevel(1)就可以打印出和SMTP服务器交互的所有信息。
    # smtp.set_debuglevel(1)
    smtp.login(username, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()


def sleeptime(timestr):
    now_time = datetime.datetime.now()
    nowhms = time.strftime("%H:%M:%S")
    # 目标时间与今天还未到达
    if nowhms < timestr:
        now_year = now_time.year
        now_month = now_time.month
        now_day = now_time.day
        aimtime = datetime.datetime.strptime(str(now_year) + "-" + str(now_month) + "-" + str(now_day) + " " + timestr,
                                             "%Y-%m-%d %H:%M:%S")
        sleeptime = (aimtime - now_time).total_seconds()
        print('当前时间', now_time)
        print('目标时间', aimtime)
        print('睡眠时间', sleeptime)
        time.sleep(sleeptime)
    else:
        # 目标时间今天已过
        aimtime = now_time + datetime.timedelta(days=+1)
        aim_year = aimtime.year
        aim_month = aimtime.month
        aim_day = aimtime.day
        aimtime = datetime.datetime.strptime(str(aim_year) + "-" + str(aim_month) + "-" + str(aim_day) + " " + timestr,
                                             "%Y-%m-%d %H:%M:%S")
        sleeptime = (aimtime - now_time).total_seconds()
        print('当前时间', now_time)
        print('目标时间', aimtime)
        print('睡眠时间', sleeptime)
        time.sleep(sleeptime)


if __name__ == '__main__':

    acttime = '08:05:00'
    sleeptime(acttime)

    username = '499428970@qq.com'
    password = 'oklvhkylrbtqbjja'
    sender = '499428970@qq.com'
    # receiver='XXX@126.com'
    # 收件人为多个收件人
    receiver = ['1340815253@qq.com']

    conn = psycopg2.connect(database="bitmex", user="postgres", password="hello", host="192.168.0.108", port="5432")
    # 合约列表
    symlist = ['ADA', 'TRX', 'BCH', 'BTC', 'LTC', 'XRP', 'ETH', 'EOS']
    binSize = '1d'
    starttime = '2018-06-29 00:00:00'
    endtime = '2018-11-06 00:00:00'

    Q1 = 50  # 权重1
    Q2 = 50  # 权重2 相对差
    S = 3  # 短周期
    L = 5  # 长周期w

    with open('比较结果.txt', 'w') as f:
        f.write('')
    for i in range(len(symlist)):
        for j in range(i + 1, len(symlist)):
            sym0 = symlist[i]
            sym1 = symlist[j]
            # print(sym0 + '与' + sym1 + '比较结果如下:')
            # data0 = path + sym0 + '_' + binSize + '.csv'
            # data1 = path + sym1 + '_' + binSize + '.csv'
            data0 = get_data(conn, sym0, binSize)
            data1 = get_data(conn, sym1, binSize)
            print(sym0, sym1)
            print(len(data0), len(data1))
            res = comres(data0, data1, S, L, Q1, Q2, starttime, endtime)
            # print(res)

            with open('比较结果.txt', 'a') as f:
                # 金叉
                if res == 1:
                    f.write(sym0 + ',' + sym1)
                else:
                    f.write(sym1 + ',' + sym0)
                f.write('\n')

    b = np.loadtxt('比较结果.txt', dtype=np.str, delimiter=',')
    sortres = {}
    for s in symlist:
        npsum = np.sum(b[:, 0] == s)
        sortres[s] = npsum

    # print('结果字典;')
    # print(sortres)
    sorted_res = sorted(sortres.items(), key=operator.itemgetter(1), reverse=True)
    now_time = str(datetime.datetime.now().strftime('%Y-%m-%d'))

    restext = now_time + '排名\n'
    for r in sorted_res:
        restext = restext + str(r[0]) + ' ' + str(r[1]) + '\n'
    print(restext)

    # send_email(username, password, receiver, restext)
    conn.close()


# ===================================================================================================================


"""
数字货币单位标准化（不然真实价值太小，不宜与计算）：把txt文件中一列数据乘以一个数值
# py file to refer to: cheng.py
"""

# b = np.loadtxt(sym+'u18_1m.csv', dtype=str, delimiter=',')  # 读取txt文件，逗号为分隔符
import datetime
symlist = ['ada', 'bch', 'eth', 'eos', 'trx', 'xrp', 'ltc']

for s in symlist:
    if s == 'ada':
        c = 10000000
    elif s == 'bch':
        c = 1000
    elif s == 'eos':
        c = 100000
    elif s == 'eth' or s == 'ltc':
        c = 10000
    elif s == 'trx':
        c = 100000000
    elif s == 'xrp':
        c = 10000000

    b = pd.read_csv(s + 'z18_1d.csv',header=None)
    print(b)
    b.ix[:, [1, 2, 3, 4]] = b.ix[:, [1, 2, 3, 4]].astype(float)

    b.ix[:, [1, 2, 3, 4]] = (b.ix[:, [1, 2, 3, 4]]) * c
    print(b)

    b.to_csv('res_' + s + 'u18z18乘' + str(c) + '.csv', sep=',', header=False, index=False, float_format='%.2f')
    print("end")

# ===================================================================================================================
"""
双均线策略品种插入顺序：
ADA
TRX
BCH
BTC
LTC
XRP
ETH
EOS
"""

# ===================================================================================================================

# Load csv file and preprocess: 如果直接从SQL读数据，这一个函数可以忽略不用
def preprocess(data_location, file):
    df = pd.read_csv(data_location + file)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df.index = pd.DatetimeIndex(df['time'])
    return df

# 直接从SQL数据库生成资产价格dataframe, 研究两辆对冲时可以调用这个双资产函数
def two_assts_from_sql(asset1, asset2, sql_limit_num):
    conn = psycopg2.connect(database="bitmexdata", user="postgres", password="tongKen123", host="128.199.97.202", port="5432")
    asset1 = """ SELECT time, open, high, low, close, volume from {} order by id desc limit {}""".format(asset1, sql_limit_num)
    asset2 = """ SELECT time, open, high, low, close, volume from {} order by id desc limit {}""".format(asset2, sql_limit_num)
    df1 = pd.read_sql(asset1, con=conn)
    df2 = pd.read_sql(asset2, con=conn)
    conn.close()
    return df1, df2


# Pull daily data from SQL data base for all 8 assets:
# asset string format to use: xbtusd_1d
def all_assts_from_sql(asset1, asset2, asset3, asset4, asset5, asset6, asset7, asset8, sql_limit_num):
    conn = psycopg2.connect(database="bitmexdata", user="postgres", password="tongKen123", host="128.199.97.202", port="5432")
    asset1 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asset1, sql_limit_num)
    asset2 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asset2, sql_limit_num)
    asset3 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asset3, sql_limit_num)
    asset4 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asset4, sql_limit_num)
    asset5 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asset5, sql_limit_num)
    asset6 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asset6, sql_limit_num)
    asset7 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asset7, sql_limit_num)
    asset8 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asset8, sql_limit_num)
    df1 = pd.read_sql(asset1, con=conn)
    df2 = pd.read_sql(asset2, con=conn)
    df3 = pd.read_sql(asset3, con=conn)
    df4 = pd.read_sql(asset4, con=conn)
    df5 = pd.read_sql(asset5, con=conn)
    df6 = pd.read_sql(asset6, con=conn)
    df7 = pd.read_sql(asset7, con=conn)
    df8 = pd.read_sql(asset8, con=conn)
    conn.close()
    return df1, df2, df3, df4, df5, df6, df7, df8

ada, bch, eos, eth, ltc, trx, btc, xrp = all_assts_from_sql("adau18z18_1d", "bchu18z18_1d", "eosu18z18_1d", 
                                                           "ethu18z18_1d", "ltcu18z18_1d", "trxu18z18_1d", 
                                                           "xbtusd_1d", "xrpu18z18_1d", 200)



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



# # 第二步：确定参数：
# N1 = 5
# N2 = 15
# N3 = 15
# N4 = 15
# period = '5T'

# # 第三步：链接数据库，提取两个资产
# bch_df, eth_df = two_assts_from_sql("bchz18_1m", "ethz18_1m", 600)

# # 第四步：转变时间周期
# bch_df.index = pd.DatetimeIndex(bch_df['time'])
# eth_df.index = pd.DatetimeIndex(eth_df['time'])
# res_bch = resample(bch_df, period)
# res_eth = resample(eth_df, period)

# # 第五步：处理时间column以此方便后面的作图
# cnvrted_date_df_bch = cnvrt_date(res_bch)
# cnvrted_date_df_eth = cnvrt_date(res_eth)

# # 第六步：计算相对差累积和、相对差累积和的快均线、相对差累计和的慢均线
# tmsum_sr = two_assets_tmsum(cnvrted_date_df_bch, cnvrted_date_df_eth, N4)
# tmsum_ma1, tmsum_ma2 = MAs_of_tmsum(tmsum_sr, N1, N2) 

# # 第七步：作图、显示生成信号
# plot_candlestick(cnvrted_date_df_eth)
# plot_candlestick(cnvrted_date_df_bch)
# # plot_tmsum(tmsum_sr)
# MA_crossover_plot_signals(tmsum_ma1, tmsum_ma2)



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


# ===================================================================================================================
