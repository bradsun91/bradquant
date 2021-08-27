import pandas as pd 
import numpy as np 

def win_loss_rate(df):
	"""
	df格式如下：
	
	close	signal	price_diff	forward_signal	returns	cum_returns
date						
2017-12-08 00:00:00	62.9782	0	NaN	NaN	NaN	NaN
2017-12-08 00:15:00	63.2600	0	0.2818	0.0	0.0000	0.0000
2017-12-08 00:30:00	64.1725	0	0.9125	0.0	0.0000	0.0000

	"""
	print ("======================================================")
	pred_correct = len(df[df['returns']>0])
	df = df['returns'].dropna()
	total_tries = len(df)
	correct_rate = pred_correct/total_tries
	print ("Correct rate: ", correct_rate)

def expected_returns(df, price_col):
	print ("======================================================")
	exptd_returns = (df['returns']/df[price_col]*100).mean()
	print ("Expected returns: {:.4f}%".format(exptd_returns))
