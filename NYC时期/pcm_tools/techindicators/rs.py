import numpy as np

def resistance(pnl, how='high'):
    mark = np.maximum if how == 'high' else np.minimum
    return mark.accumulate(pnl.fillna(pnl.min()))

# resistance(pnl, how= 'high').plot()



# Create functions for drawdowns
def drawdown(pnl):
    dd = 1 - (pnl / water_mark(pnl, how='high').shift(1))
    dd.ix[dd < 0] = 0
    return dd




# Create functions for max drawdown
def drawdown(pnl):
    dd = 1 - (pnl / water_mark(pnl, how='high').shift(1))
    dd.ix[dd < 0] = 0
    return dd.max()


def rolling_max(series, window_span):
    mark = series.rolling(window_span).max()
    return mark


def rolling_min(series, window_span):
    mark = series.rolling(window_span).min()
    return mark



