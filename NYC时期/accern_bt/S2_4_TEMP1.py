impact_cutoff =  [impact_cutoff]
templated_cutoff = [templated_cutoff]
event_relevance_cutoff = [event_relevance_cutoff]
event_impact_pct_change_avg_cutoff = [event_impact_pct_change_avg]
long_weights_setup = 0.05
short_weights_setup = 0.05
leverage_level = 1
data_file = [data_file]


impact_pos_string = 'event_impact_pos'
impact_neg_string = 'event_impact_neg'
templated_score_string = 'templated_story_score'
relevance_string = 'event_relevance'
event_impact_pct_avg_string = 'event_impact_pct_change_avg'




def initialize(context):
    context.leverage_buffer = 1

    # Set benchmark as SPY
    set_benchmark(symbol('SPY'))

    # Set slippage model in this backtesting strategy to be 1(or $0 if purely testing trading logic) cent to be the bid/ask spread for each trade 
    set_slippage(slippage.FixedSlippage(spread=0)) 
    
    # Set commission model to be $0.0075 (or $0 if purely testing trading logic) per share with each trade's minimum commission fee to be 1 dollor (or $0 if purely testing trading logic) no matter how small the trade value is.
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))  

    # Scheduling function to check if there's any morning_buy extry signal every day's morning 30 mins after the market opens. If there's any, execute morning_buy trade logic. 
    
    schedule_function(morning_execution,
                      date_rules.every_day(),
                      time_rules.market_open(hours=0, minutes=30))

    
    schedule_function(record_vars,
                      date_rules.every_day(),
                      time_rules.market_close(hours=0, minutes=1))   


    # Pulls the data from the csv file
    fetch_csv(data_file,
              date_column='time',  # Assigning the column label
              date_format='%m/%y/%d %H:%M:',  # Using date format from CSV file
              mask=False, 
              timezone='UTC')  # if the data csv file is UTC time, we set this as 'UTC', if EST, then we set the timezone = 'EST'
    

def morning_execution(context, data):
    current_bar = get_datetime()
    # Let's store the long and short positions that are in our portfolio
    long_ = [] 
    short_ = []
    try:
        for position in context.portfolio.positions:
            if context.portfolio.positions[position].amount>0:
                long_.append(position)
                # long_value += context.portfolio.positions[position].amount
        
            if context.portfolio.positions[position].amount<0:
                short_.append(position)
                # short_value += context.portfolio.positions[position].amount
                
    except:
        pass
    print ('current_gross_leverage: ', context.account.leverage)
    print ('current_net_leverage', context.account.net_leverage)
   
    selected_stock_to_long = [] 
    selected_stock_to_close = []
    selected_stock_to_short = []
    
    print ("***********************************Prepping to Start the Round***********************************")
    # data.current(stock, 'year/month/day/hour/minute') is our csv's timestamp data
    print ('total_universe_discovered_length: ', len(data.fetcher_assets))
    for stock in data.fetcher_assets: 
        if data.current(stock, 'year') == current_bar.year:
            if data.current(stock, 'month') == current_bar.month:
                if data.current(stock, 'day') == current_bar.day:
                    if data.current(stock, impact_pos_string) > impact_cutoff and data.current(stock, templated_score_string) < templated_cutoff and data.current(stock, event_impact_pct_avg_string) > event_impact_pct_change_avg_cutoff and data.current(stock, relevance_string) > event_relevance_cutoff: 
                        if data.can_trade(stock):
                            selected_stock_to_long.append(stock)    
                    if data.current(stock, impact_neg_string) > impact_cutoff and data.current(stock, templated_score_string) < templated_cutoff and data.current(stock, event_impact_pct_avg_string) > event_impact_pct_change_avg_cutoff and data.current(stock, relevance_string) > event_relevance_cutoff:
                        if data.can_trade(stock):
                            selected_stock_to_short.append(stock)
                    if data.current(stock, impact_pos_string) < impact_cutoff and stock in long_:
                        if data.can_trade(stock):
                            selected_stock_to_close.append(stock)
                    if data.current(stock, impact_neg_string) > impact_cutoff and stock in short_:
                        if data.can_trade(stock):
                            selected_stock_to_close.append(stock)
    
    final_selected_long_len = len(selected_stock_to_long)
    final_selected_short_len = len(selected_stock_to_short)
    final_selected_close_len = len(selected_stock_to_close)


    for s in data.fetcher_assets: 
        if data.current(s, 'year') == current_bar.year:
            if data.current(s, 'month') == current_bar.month:
                if data.current(s, 'day') == current_bar.day:
                    if s in selected_stock_to_long:
                        # Long trigger:
                        try:
                            # Each single long position should not be over 5% of the portfolio value by the time this single position is entered. 
                            long_weights = long_weights_setup
                            if s in long_:
                                pass
                            elif context.account.leverage < 1:
                                order_target_percent(s, long_weights)
                                context.account.leverage = context.account.leverage + long_weights
                            else:
                                continue 
                            print ('current position leverage: ', context.account.leverage)
                        except:
                            pass

                    if s in selected_stock_to_short:
                        # Short trigger:
                        try:
                            # Each single short position should not be over 5% of the portfolio value by the time this single position is entered. 
                            short_weights = short_weights_setup
                            if s in short_:
                                pass
                            elif context.account.leverage < 1: 
                                order_target_percent(s, -short_weights)
                                context.account.leverage = context.account.leverage + short_weights
                            else:
                                continue
                            print ('current position leverage: ', context.account.leverage)
                        except:
                            pass 

                    if s in selected_stock_to_close:
                        # Close trigger:
                        if s in context.portfolio.positions:
                            order_target_percent(s, 0)



def record_vars(context, data):
    long_count = 0
    short_count = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            long_count += 1
        elif position.amount < 0:
            short_count += 1
    record(gross_leverage=context.account.leverage)  # Plot leverage to chart
    record(net_leverage=context.account.net_leverage)
    record(num_longs = long_count)
    record(num_shorts = short_count)
    record(portfolio_size = len(context.portfolio.positions))