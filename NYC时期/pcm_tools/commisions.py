def options_commissions(premium, contract_qty, smart_routed = True):
    if premium >= 0.1:
        com = 0.7*contract_qty
        if com > 1:
            final_com = com
        else:
            final_com = 1
    elif premium < 0.1 and premium >= 0.05:
        com = 0.5*contract_qty
        if com > 1:
            final_com = com
        else:
            final_com = 1
    elif premium < 0.05:
        com = 0.25*contract_qty
        if com > 1:
            final_com = com
        else:
            final_com = 1       
    return final_com



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    