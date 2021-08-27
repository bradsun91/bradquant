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
















