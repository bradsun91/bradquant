import pandas as pd
import numpy as np  


"""
此文件目的主要是根据不同的策略逻辑生成信号数据，以此导入performance.py文件
"""

# Strategy 1: simple movng average & price cross
def Strat_SMA(df, price_col, window):
	"""
	1. price_col是建立策略信号的对应价格列，比如较常用的"close"

	df格式如下：
	date已经是经过pd.to_datetime转换过后的index
						    open	 high	  low	  close	   volume
			date						
	2017-07-28 18:30:00	   152.0	152.4    150.5	  151.5     82344
	2017-07-28 18:45:00	   151.0	151.2 	 149.5	  150.0    120370
	"""
	df = df[[price_col]]
	# print (window_)
	SMA_col = '{}_SMA'.format(window)
	df[SMA_col] = df[price_col].rolling(window = window).mean()
	df['diff'] = df[SMA_col] - df[price_col]
	df['signal'] = df['diff'].apply(lambda x: 1 if x>0 else -1 if x<0 else 0)
	df = df[[price_col, 'signal']]
	return df



