def collision_rebalance(investing_capital, margin_multiple, spy_price, qqq_price):
    bp = investing_capital*margin_multiple
    spy_target_value =  bp/2
    qqq_target_value = bp-spy_target_value
    hedge_ratio = spy_price/qqq_price
    spy_qty = int(spy_target_value/spy_price)
    qqq_qty = int(qqq_target_value/qqq_price)
    
    print ("hedge_ratio: {}".format(hedge_ratio))
    print ("spy_qty: {}".format(spy_qty) )
    print ("qqq_qty: {}".format(qqq_qty))
    
    
    
def calc_trade_commissions(shares, price):
    fixed_fee = 0.005
    minimum_per_order = 1
    max_per_order = 0.005
    if shares*price*max_per_order < minimum_per_order:
        total_fees = shares*price*max_per_order  
    elif shares*fixed_fee < 1:
        total_fees = 1
    else:
        total_fees = shares*fixed_fee
        
    return total_fees

def pair_commissions_pct(execution_number): 
    pair_execution_comm_per_exec = 0.000055
    total_comm = pair_execution_comm_per_exec*execution_number
    print ("Total Commission Percentage: {}%".format(total_comm*100))