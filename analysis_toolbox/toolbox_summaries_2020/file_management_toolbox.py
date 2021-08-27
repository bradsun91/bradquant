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
# 05-21-2019
# 如何避免写for循环：
# https://blog.csdn.net/wonengguwozai/article/details/78295484

# assert断言：
# python assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。可以理解assert断言语句为raise-if-not，
# 用来测试表示式，其返回值为假，就会触发异常。
# https://www.cnblogs.com/liuchunxiao83/p/5298016.html



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
# 3-7-2019 udpated
# loop through all csv files in a folder.
import glob
path = "path/to/dir/*.csv"
for fname in glob.glob(path):
    print(fname)


# ===================================================================================================================
# 2-25-2019 udpated
# 直接锁定需要调用函数的文件夹：
from sys import path
path.append(r'C:\Users\workspace\brad_public_workspace_on_win\lib\test_brad_bt') #将存放module的路径添加进来
import ml_all_in_one # as an example


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



def combine_csv_files(path_drct, output_file_name):
    path = path_drct
    interesting_files = glob.glob(path) 
    header_saved = False
    with open(output_file_name,'wb') as fout:
        for filename in interesting_files:
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header.encode('utf-8'))
                    header_saved = True
                for line in fin:
                    fout.write(line.encode('utf-8'))


# ===================================================================================================================
# 读中文读出一大堆乱码，那么前面要加上以下代码：
#-*- coding : utf-8-*-
# coding:unicode_escape

fin_data = pd.read_csv("fin_data_20201113.csv", encoding='gbk')

























