import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def plot_cum_returns(df, figsize):
	"""
	df的格式如下：date为index
	close	signal	price_diff	forward_signal	returns	cum_returns
date						
2017-12-08 00:00:00	62.9782	0	NaN	NaN	NaN	NaN
2017-12-08 00:15:00	63.2600	0	0.2818	0.0	0.0000	0.0000

	"""
	cum_returns = df['cum_returns']
	cum_returns.plot(figsize = figsize)
	plt.title("Cumulative Returns")
