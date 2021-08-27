import pandas as pd 
import numpy as np 


def QR(df1, df2, price_col):
	"""
	df1/df2分别代表两个资产的价格——columns: date open high low close volume 的 dataframe
	并且column中的date是string
	"""
	df1 = df1[['date', price_col]]
	df2 = df2[['date', price_col]]
	df_merged = df1.merge(df2, on = 'date', how = 'inner')
	df_merged.index = pd.to_datetime(df_merged['date'])
	p1_pct = df_merged["{}_x".format(price_col)].pct_change()
	p2_pct = df_merged["{}_y".format(price_col)].pct_change()
	df_merged['p1_pct'] = p1_pct
	df_merged['p2_pct'] = p2_pct
	df_merged['qr'] = df_merged['p1_pct'] - df_merged['p2_pct']
	return df_merged

def QRSUM(df1, df2, price_col):
	"""
	p1/p2分别代表两个资产的价格，可以是open high low close
	"""
	df1 = df1[['date', price_col]]
	df2 = df2[['date', price_col]]
	df_merged = df1.merge(df2, on = 'date', how = 'inner')
	df_merged.index = pd.to_datetime(df_merged['date'])
	p1_pct = df_merged["{}_x".format(price_col)].pct_change()
	p2_pct = df_merged["{}_y".format(price_col)].pct_change()
	df_merged['p1_pct'] = p1_pct
	df_merged['p2_pct'] = p2_pct
	df_merged['qr'] = df_merged['p1_pct'] - df_merged['p2_pct']
	df_merged['qrsum'] = df_merged['qr'].cumsum()
	return df_merged


def MACD(df, price_col, n_fast, n_slow, n_emadiff): # n_fast = 12, n_slow = 26, n_emadiff = 9
    EMAfast = df[price_col].ewm(span = n_fast).mean()
    EMAslow = df[price_col].ewm(span = n_slow).mean()
    diff = pd.Series(EMAfast - EMAslow, name = 'diff' + str(n_fast) + '_' + str(n_slow))
    EMAdiff = diff.ewm(span = n_emadiff).mean().rename('EMAdiff' + '_' + str(n_emadiff))
    MACD = pd.Series(diff - EMAdiff, name = 'MACD' + str(n_fast) + '_' + str(n_slow) + '_' + str(n_emadiff))
    df['EMAfast'] = EMAfast
    df['EMAslow'] = EMAslow
    df['diff'] = diff
    df['EMAdiff'] = EMAdiff
    df['MACD'] = MACD
    return df
