from datetime import datetime
from fredapi import Fred
import numpy as np
import pandas as pd
import scipy.stats as ss
from pcm_tools.pricing import bsm_pricing_call, bsm_pricing_put, bsm_put, bsm_call
from pcm_tools.pricing import bsm_pricing_call_return, bsm_pricing_put_return
import matplotlib.pyplot as plt
from pcm_tools.toolbox import fred_1r_ir_today
from scipy.optimize import newton
from pcm_tools.commisions import options_commissions


def put_option_expected_return(contract_num, S, K, V, T, Div, target_holding_days, target_stk_price_chg, target_im_chg_pct):
    option_stk_unit = 100
    returned = bsm_pricing_put_return(S, K, V, T, Div)
    
    current_option_price = returned[0] 
    current_delta = returned[1]
    current_gamma = returned[2]
    current_theta = returned[3]
    current_vega = returned[4]
    current_rho = returned[5]
    
    target_stk_price_chg = target_stk_price_chg
    target_im_chg_pct = target_im_chg_pct
    option_stk_unit = option_stk_unit
    
    # Calculate the impact of change of Implied Vol
    implied_value_change = current_vega*target_im_chg_pct*contract_num*option_stk_unit
    
    # Calculate the impact of change of Delta
    delta_value_change = current_delta * target_stk_price_chg * contract_num * option_stk_unit
    
    # Calculate the impact of Time Decay
    target_holding_days = target_holding_days
    subtract_list = []
    T = T
    for i in range(target_holding_days):
        days_left = T-i
        subtract_list.append(days_left)


    subtract_list = subtract_list
    theta_list = []
    for t in subtract_list:
        theta = bsm_pricing_put_return(S, K, V, t, Div)[3]
        theta_list.append(theta)
        

    total_theta_cost = pd.Series(theta_list).sum()
    total_theta_cost_value = total_theta_cost*contract_num*option_stk_unit
    
    expected_return = implied_value_change + delta_value_change + total_theta_cost_value
    
    premium = bsm_put(S, K, V, T, Div)
    commision_fee = options_commissions(premium, contract_num)
    net_return = expected_return - commision_fee*2
    return net_return 