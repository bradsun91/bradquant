def series_ema(series, window_span):
    expmovingavg = series.ewm(span = window_span).mean()
    return expmovingavg