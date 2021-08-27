def ma_signal(series, window_fast, window_mid, window_slow):
    
    price_ma_fast = series_ema(series, window_fast)
    price_ma_mid = series_ema(series, window_mid)
    price_ma_slow = series_ema(series, window_slow)
    
    if series[-1] >= price_ma_slow[-1]:
        if series[-1] >=price_ma_mid[-1]:
            if series[-1] >= price_ma_fast[-1]:
                return 2
#                 print ('Price is above/at {}-day EMA, strong up trend'.format(window_fast))
            else:
                return 1
#                 print ('Price is above/at {}-day EMA, price falling back'.format(window_mid))
        else:
            return 0
#             print ('Price is above/at {}-day EMA, price falling sharply'.format(window_slow))
    else:
        return -1
#         print('Price is below {}-day EMA, warning!'.format(window_slow))


def filtered_ma_tickers(list_, signal):
    """Documentation: The building block of this function is the function: ma_signal(series, window_fast, window_mid, window_slow); there are 
    4 inputs as signals for us to choose from in this filtered_ma_tickers function: 2, 1, 0, -1, respectively representing filtered stocks with 
    : 1) strong up trend, 2) price mildly falling back, 3) price falling sharply and 4) big drawdown"""
    filtered_list = []
    for ticker in list_:
        filtered_ticker_sig = ma_signal(list_[ticker], window1, window2, window3)
        if filtered_ticker_sig == signal:
            filtered_list.append(ticker)
        else:
            continue
    final_list = filtered_list

    return final_list