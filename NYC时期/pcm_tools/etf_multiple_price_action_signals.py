import numpy as np, pandas as pd
from pcm_tools.techindicators.ema import series_ema
from pcm_tools.toolbox import two_graphs_twinx, three_graphs_twinx, two_separate_graphs
from datetime import datetime
import matplotlib.pyplot as plt
# Signal One: Acceleration
# Signal Two: MACD
# Signal Three: Slope of the EMA
# VIX
# CNN Fear Index


################### Signal 1:

# PriceMomentum
class PriceMomentum(object):
    
    def __init__(self, ticker_close, close_ema_window, velocity_ema_window, acc_ema_window, end):
        # the issue is to decide how long the rolling window should be, which needs be based on 
        # historical stast, e.g. what is the average drawdown duration of the SPY
        self.ticker_close = ticker_close
        self.close_ema_window = close_ema_window
        self.velocity_ema_window = velocity_ema_window
        self.acc_ema_window = acc_ema_window
        self.end = end
        
    def price_ema(self):
        asset_ema = series_ema(self.ticker_close, self.close_ema_window)
        return asset_ema
    
    def price_vlcty(self):
    # Calculate price change: momentum velocity
        asset_pct = self.ticker_close.pct_change()
        return asset_pct
    
    # Calculate price_change's pct_change: momentum acceleration
    # Let's smooth out by taking the ema of the velocity first:
    def price_vlcty_ema(self):
        asset_pct = self.ticker_close.pct_change()
        asset_pct_ema = series_ema(asset_pct, self.velocity_ema_window)
        return asset_pct_ema
    
    def price_acc(self):
    # Second to last step, let's calculate velocity ema's momentum velocity, which means price moving average's acceleration:
        asset_pct = self.ticker_close.pct_change()
        asset_pct_ema = series_ema(asset_pct, self.velocity_ema_window)
        asset_pct_ema_pct = asset_pct_ema.pct_change()
        return asset_pct_ema_pct
    
    def price_acc_ema(self):
    # Finally let's smooth out the acceleration by taking its ema:
        asset_pct = self.ticker_close.pct_change()
        asset_pct_ema = series_ema(asset_pct, self.velocity_ema_window)
        asset_pct_ema_pct = asset_pct_ema.pct_change()
        asset_pct_ema_pct_ema = series_ema(asset_pct_ema_pct, self.acc_ema_window)
        return asset_pct_ema_pct_ema

    def plot_(self):
        asset_pct = self.ticker_close.pct_change()
        asset_pct_ema = series_ema(asset_pct, self.velocity_ema_window)
        asset_pct_ema_pct = asset_pct_ema.pct_change()
        asset_pct_ema_pct_ema = series_ema(asset_pct_ema_pct, self.acc_ema_window)
        two_separate_graphs(asset_pct_ema_pct_ema, self.ticker_close)
        
    def stats(self):
        asset_pct = self.ticker_close.pct_change()
        asset_pct_ema = series_ema(asset_pct, self.velocity_ema_window)
        asset_pct_ema_pct = asset_pct_ema.pct_change()
        asset_pct_ema_pct_ema = series_ema(asset_pct_ema_pct, self.acc_ema_window).dropna(0)
        plt.hist(asset_pct_ema_pct_ema)
        plt.title("Price EMA Acc Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        print (asset_pct_ema_pct_ema.describe())
        
    def current_level_acc(self):
        asset_pct = self.ticker_close.pct_change()
        asset_pct_ema = series_ema(asset_pct, self.velocity_ema_window)
        asset_pct_ema_pct = asset_pct_ema.pct_change()
        asset_pct_ema_pct_ema = series_ema(asset_pct_ema_pct, self.acc_ema_window).dropna(0)
        p_stats_20 = asset_pct_ema_pct_ema.rolling(window = self.acc_ema_window).quantile(0.2).values[-1]
        p_stats_40 = asset_pct_ema_pct_ema.rolling(window = self.acc_ema_window).quantile(0.4).values[-1]
        p_stats_60 = asset_pct_ema_pct_ema.rolling(window = self.acc_ema_window).quantile(0.6).values[-1]
        p_stats_80 = asset_pct_ema_pct_ema.rolling(window = self.acc_ema_window).quantile(0.8).values[-1]
        p_stats_100 = asset_pct_ema_pct_ema.rolling(window = self.acc_ema_window).quantile(1.0).values[-1]

        if p_stats_80 < asset_pct_ema_pct_ema.values[-1] <= p_stats_100:
            current_level = '80-100. Decrease or Exit Position.'
        elif p_stats_60 < asset_pct_ema_pct_ema.values[-1] <= p_stats_80 and asset_pct_ema_pct_ema.values[-1] > asset_pct_ema_pct_ema.values[-2]:
            current_level = '60-80. Prep to Unload...'
        elif p_stats_60 < asset_pct_ema_pct_ema.values[-1] <= p_stats_80 and asset_pct_ema_pct_ema.values[-1] < asset_pct_ema_pct_ema.values[-2]:
            current_level = '60-80. Momentum Acc Reducing, Decrease or Exit Position.'
        elif p_stats_20 < asset_pct_ema_pct_ema.values[-1] <= p_stats_40 and asset_pct_ema_pct_ema.values[-1] > asset_pct_ema_pct_ema.values[-2]:
            current_level = '20-40. Up-ward Momentum Still strong'
        elif asset_pct_ema_pct_ema.values[-2] <= p_stats_20 and asset_pct_ema_pct_ema.values[-1] > asset_pct_ema_pct_ema.values[-2]:
            current_level = '0-20. Momentum Bouncing Back after Big Mkt Drop.'
        elif asset_pct_ema_pct_ema.values[-2] <= p_stats_20 and asset_pct_ema_pct_ema.values[-1] < asset_pct_ema_pct_ema.values[-2]:
            current_level = '0-20. Market Dropping, Panicking.'
        else:
            current_level = 'Normal Level'
        # what's the current level?
        # print ("p_stats_20:{}".format(p_stats_20))
        # print ("-"*60)
        # print ("p_stats_40:{}".format(p_stats_40))
        # print ("-"*60)
        # print ("p_stats_60:{}".format(p_stats_60))
        # print ("-"*60)
        # print ("p_stats_80:{}".format(p_stats_80))
        # print ("-"*60)
        # print ("p_stats_100:{}".format(p_stats_100))
        # print ("-"*60)
        print ("Current Acc signal level by {}: Acc_Signal_Level: {:0.02f}; {}".format(self.end, asset_pct_ema_pct_ema.values[-1], current_level))


