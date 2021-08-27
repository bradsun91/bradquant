def macd(series, fast_line_lookback, slow_line_lookback, macd_ema_lookback):
    fast_ema = series.ewm(span = fast_line_lookback).mean()
    slow_ema = series.ewm(span = slow_line_lookback).mean()
    macd = fast_ema - slow_ema
    macd_ema = macd.ewm(span = macd_ema_lookback).mean()
    diff = macd - macd_ema
    
    return diff