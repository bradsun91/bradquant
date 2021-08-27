import numpy as np
import pandas as pd
import quandl   # Necessary for obtaining financial data easily

from backtest import Strategy, Portfolio

class AccernSentimentStrategy(Strategy):

	def __init__(self, symbol, bars):
		self.symbol = symbol
		self.bars = bars

	def generate_signals(self):
		signals = pd.DataFrame()



