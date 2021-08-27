def bollinger_bands_mean(stock_price, window_size, num_of_std):

    rolling_mean = stock_price.ewm(span=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean
# m = bollinger_bands_mean(qqq_spy, 40, 2)


def bollinger_bands_upper(stock_price, window_size, num_of_std):

    rolling_mean = stock_price.ewm(span=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return upper_band
# u = bollinger_bands_upper(qqq_spy, 40, 2)


def bollinger_bands_lower(stock_price, window_size, num_of_std):

    rolling_mean = stock_price.ewm(span=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return lower_band
# l = bollinger_bands_lower(qqq_spy, 40, 2)

