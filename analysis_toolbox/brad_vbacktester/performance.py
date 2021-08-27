import pandas as pd, numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style


def sngl_performance(signal_df, price_col):
    """
    1. date has been processed through pd.to_datetime() as an index
    2. signalis either 0 or -1 or 1
    
    signal_df's format：

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