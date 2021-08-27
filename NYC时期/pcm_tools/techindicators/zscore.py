import pandas

def rolling_z_score(series, look_back_window):
    mean_ = pandas.Series.ewm(series, span = look_back_window).mean()
    std = series.rolling(window = look_back_window).std()
    zs = (series - mean_)/std
    return zs
    # print ("Z-score: {}".format(zs))
    
  
def expanding_z_score(series, min_periods=1):
    def _z_score(ary):
        return ((ary-np.mean(ary)) / np.std(ary, ddof=1))[-1]

    exp = series.expanding(min_periods=min_periods)
    return exp.apply(expanding_zscore)
