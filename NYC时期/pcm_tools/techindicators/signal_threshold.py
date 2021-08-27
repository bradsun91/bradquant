import pandas as pd

def signal_threshold(values, min_threshold, max_threshold):
    signal_threshold_list = []
    for item in values:
        if item <= min_threshold:
            value = min_threshold
        elif item >= max_threshold:
            value = max_threshold
        else:
            value = item
        signal_threshold_list.append(value)
    filtered_list = pd.Series(signal_threshold_list, index = values.index)
    
    return filtered_list
#     filtered_list.plot(figsize = (25, 10))
    
# filtered_spread = large_spread(qs_exp_spread, -0.01, 0.01, 0)