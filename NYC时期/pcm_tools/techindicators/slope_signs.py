# Create a function that identifies slope's positive or negative signs
import pandas as pd
def show_slope(pct_chg):
    sign_list = []
    for item in pct_chg:
        if item < 0:
            sign = -1
        elif item > 0:
            sign = 1
        else:
            sign = 0
        sign_list.append(sign)
    signs_with_index = pd.Series(sign_list, index = pct_chg.index)
    return signs_with_index
    

# slope_signs = show_slope(exp_slope)
# three_graphs(qqq_spy, qs_exp_spread, slope_signs)