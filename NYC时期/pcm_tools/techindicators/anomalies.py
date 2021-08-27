import pandas as pd

def anomalies(series, pct_lower_threshold, pct_upper_threshold, mid_value):
    anomaly_list = []
    for item in series:
        if item <= pct_lower_threshold:
            pct = item
        elif item >= pct_upper_threshold:
            pct = item
        else:
            pct = mid_value
        anomaly_list.append(pct)
    filtered_list = pd.Series(anomaly_list, index = series.index)
    
    return filtered_list 