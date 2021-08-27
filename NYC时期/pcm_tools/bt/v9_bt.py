import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from datetime import timedelta
import statsmodels.api as sm
import csv



def get_raw_csv(file, col_strs, col_dict, target_file_name, sliced_rows):
    usecols = col_strs
    print("Parsing file: {}".format(file))
    df = pd.read_csv(file,
                     engine='c',
                     dtype='object',
                     skipinitialspace=True,
                     quoting=csv.QUOTE_ALL,
                     usecols=usecols,
                     nrows=sliced_rows)
    df.rename(columns=col_dict, inplace=True)
    df.to_csv(target_file_name, sep=',', index=False)



def clean_extracted(target_file_name, one_ticker, ticker_with_col_str_final):
    

    #### 4.1.2.1  Delete the UTC and space in the date strings 
    open_sample = pd.read_csv(target_file_name)
    str_dt_list = list(open_sample.date)
    for i, s in enumerate(str_dt_list):
        str_dt_list[i] = str_dt_list[i].replace(" UTC", "")
    

    #### 4.1.2.2 Convert strings into datetime
    for i, s in enumerate(str_dt_list):
        str_dt_list[i] = datetime.strptime(str_dt_list[i],'%Y-%m-%d %H:%M:%S')
    open_sample.index = str_dt_list
    del open_sample['date']

    #### 4.1.3  Select specific tickers from the sample dataframe
    ticker_filtered = open_sample[(open_sample.tickers == one_ticker)]
    
    #### 4.1.4  Round up the next minute for timestamps 
    list_ = []
    for time_index in ticker_filtered.index:
        minute_added = time_index + timedelta(minutes=1)
        changed_index = minute_added.replace(hour = minute_added.hour, minute = minute_added.minute, second=0, microsecond=0)
        list_.append(changed_index)
    ticker_filtered.index = list_
    
    #### 4.1.5  Drop rows with duplicated timestamps
    ticker_filtered['timestamp'] = ticker_filtered.index
    ticker_sentiment_minrounded_dropduplicated = ticker_filtered.drop_duplicates('timestamp', keep='last')
    
    #### 4.1.6  Trim timestamps into only trading hour data
    df = []
    ts = []
    for i, t in enumerate(ticker_sentiment_minrounded_dropduplicated.index):
        if ((9, 31) <= (ticker_sentiment_minrounded_dropduplicated.index[i].hour, ticker_sentiment_minrounded_dropduplicated.index[i].minute) < (16, 0)) == True:
            item = ticker_sentiment_minrounded_dropduplicated.sentiment[i]
            timestamp = ticker_sentiment_minrounded_dropduplicated.index[i]
        else:
            continue
        df.append(item)
        ts.append(timestamp)
    price_list = df  
    trading_ts = ts
    ticker_trimmed = pd.DataFrame(price_list, columns=[ticker_with_col_str_final], index = trading_ts)
    
    return ticker_trimmed


class NewsQuantSignalGeneration(object):

    def bool_signal_creation_above(df, df_col_str, ticker_with_signal_str_rdm, max_threshold, bracket_closed):
        sentiment_signal_df = pd.DataFrame()
        if bracket_closed == False:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: 1 if x > max_threshold else 0)
        else:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: 1 if x >= max_threshold else 0)
        return sentiment_signal_df


    def bool_signal_creation_below(df, df_col_str, ticker_with_signal_str_rdm, min_threshold, bracket_closed):
        sentiment_signal_df = pd.DataFrame()
        if bracket_closed == False:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: 1 if x < min_threshold else 0)
        else:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: 1 if x <= min_threshold else 0)
        return sentiment_signal_df


    def bool_signal_creation_between(df, df_col_str, ticker_with_signal_str_rdm, max_threshold, min_threshold, left_bracket_closed, right_bracket_closed):
        sentiment_signal_df = pd.DataFrame()
        if left_bracket_closed == True and right_bracket_closed == True:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: 1 if min_threshold <= x <= max_threshold else 0)
        elif left_bracket_closed == True and right_bracket_closed == False:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: 1 if min_threshold <= x < max_threshold else 0)
        elif left_bracket_closed == False and right_bracket_closed == True:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: 1 if min_threshold < x <= max_threshold else 0)
        else: # both are open brackets
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: 1 if min_threshold < x < max_threshold else 0)
        return sentiment_signal_df

    def bool_signal_creation_outside(df, df_col_str, ticker_with_signal_str_rdm, max_threshold, min_threshold, left_bracket_closed, right_bracket_closed):
        sentiment_signal_df = pd.DataFrame()
        if left_bracket_closed == True and right_bracket_closed == True:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: -1 if min_threshold >= x else (1 if x >= max_threshold else 0))
        elif left_bracket_closed == True and right_bracket_closed == False:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: -1 if min_threshold >= x else (1 if x > max_threshold else 0))
        elif left_bracket_closed == False and right_bracket_closed == True:
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: -1 if min_threshold > x else (1 if x >= max_threshold else 0))
        else: 
            sentiment_signal_df[ticker_with_signal_str_rdm] = df[df_col_str].apply(lambda x: -1 if min_threshold > x else (1 if x > max_threshold else 0))
        return sentiment_signal_df


def price_data_process(price_df):
    price_df["time_index"] = price_df["date"] + ' ' + price_df["timestamp"]
    del price_df['date'], price_df['timestamp']
    price_df.index = price_df['time_index']
    del price_df['time_index']
    price_df_sr = price_df['close_price']

    # Convert strings into datetime
    str_dt_list = list(price_df.index)
    for i, s in enumerate(str_dt_list):
        str_dt_list[i] = datetime.strptime(str_dt_list[i],'%Y-%m-%d %H:%M:%S')
    price_df.index = str_dt_list
    return price_df




def ema_crossover(asset_dataframe, slow_ema_window, fast_ema_window, close_price_str):
    slow_ema_window = slow_ema_window
    fast_ema_window = fast_ema_window
    asset_close_sr = asset_dataframe[close_price_str]
    slow_ema = series_ema(asset_close_sr, slow_ema_window)
    fast_ema = series_ema(asset_close_sr, fast_ema_window)
    asset_dataframe['slow_ema'] = slow_ema
    asset_dataframe['fast_ema'] = fast_ema
    asset_dataframe['diff'] = fast_ema - slow_ema
    asset_dataframe['signal'] = asset_dataframe['diff'].apply(lambda x: 1 if x > 0 else 0)
    return asset_dataframe


def merge_signal_price_files(df1, df2):
    df1['ts'] = df1.index
    df2['ts'] = df2.index
    merged_signal_df = df1.merge(df2, on = 'ts')
    merged_signal_df.index = merged_signal_df['ts']
    return merged_signal_df


def merged_final_signal(merged_signal_df, price_signal_str, sentiment_signal_str, iter_str):
    merged_signal_df['final_signal'] = 'Initializing'
    for i, item in enumerate(merged_signal_df[iter_str]):
        if merged_signal_df[price_signal_str][i] == 1 and merged_signal_df[sentiment_signal_str][i] == 1:
            merged_signal_df['final_signal'][i] = 1
            print (1, i)

        elif merged_signal_df[price_signal_str][i] == 0 and merged_signal_df[sentiment_signal_str][i] == 0:
            merged_signal_df['final_signal'][i] = 0
            print (0, i)

        else: # nothing changed, update this ts data as to be the previous one:
            # notice here, if i == 0 in this case, i-1 from below won't run, so analyze different scenarios:
            if i == 0: 
                continue
            else:
                merged_signal_df['final_signal'][i] = merged_signal_df['final_signal'][i-1]
                print ("{}".format(merged_signal_df['final_signal'][i-1]), i)
    return merged_signal_df




class PortfolioCalculation(object):

    """
    The purpose of building this class is to define the quantity and profit functions we will be calling in the backtesting
    process, where quantity(bp, ticker_price, port_pct) means the position size we hold in our strategy; while 
    profit(x0, x1, quantity, direction = 'L') means each time how much we make for each round of our trade, where x0 means 
    the previous stock price, x1 means the current stock price
    """
    
    def quantity(bp, ticker_price, port_pct): 
        ticker_quant = bp*port_pct/ticker_price
        return ticker_quant

    def profit(x0, x1, quantity, direction = 'L'):
        """direction = 'L' means when we calculate each profit, we go long, if we want to go short for each trade we put 
        direction = 'S'
        """
        p = (x1 - x0) * int(quantity)
        if direction == 'L':
            return p
        elif direction == 'S':
            return -p
        else:
            raise ValueError('At least put one strategy direction here!')


def ib_equity_commissions(quantity, fill_price):
        """
        Calculate the Interactive Brokers commission for
        a transaction. This is based on the US Fixed pricing,
        the details of which can be found here:
        https://www.interactivebrokers.co.uk/en/index.php?f=1590&p=stocks1
        """
        commission = min(0.005 * fill_price * quantity, max(1.0, 0.005 * quantity)) 
        return commission


class BT(object):
    """
    This is the meat of the whole backtesting infrastructure, mainly to prepare the final dataframe with all 
    important columns, including pnl and all stuff.
    
    In this version of updates, following bugs and problems are fixed from previous version of BT class:
    
    1. previous versions' signal change lags one row behind
    2. previous versions didn't factor into comissions and all other important trade and market impact stats 
    """
    
    def __init__(self, port_utilization, tradable_capital, margin_leverage):
        self.data = []
        self.tradable_capital = tradable_capital
        self.margin_leverage = margin_leverage
        self.port_utilization = port_utilization
        self.volume_limit = 0.025 ############# default = 0.025 as the quantopian standard when initializing.
        self.price_impact_constant = 0.1  ############# default = 0.1 as the quantopian standard when initializing.
        self.price_impact_constant = 0.1
        """
        The price impact constant (default 0.1) defines how large of an impact the order will have on the backtester's price calculation. 
        The slippage is calculated by multiplying the price impact constant (default = 0.1) by the square of the ratio of the order to the total volume. 
        In Quantopian previous example, for the 25-share orders, the price impact is 0.1 * (25/1000) * (25/1000), or 0.00625%. For the 10-share order, 
        the price impact is 0.1 * (10/1000) * (10/1000), or 0.001%.
        
        Reference page:
        https://www.quantopian.com/help#ide-slippage
        """

    def core_backtester(self, signal_dataframe, df_sgnl_str, df_tkr_price_str, strategy_drct, 
                        long_short_both, enter_signal, exit_signal, long_signal, short_signal):
        
        signal_updates = signal_dataframe[df_sgnl_str]
        ticker_price_updates = signal_dataframe[df_tkr_price_str]
        init_signal = 0
        init_ticker_qty = 0
        init_port_value = 0 
        init_ticker_price = 0
        init_mkt_qty = 0
        init_mkt_volume = 0
        cum_trades = 0
        cum_commissions = 0
        each_commission_cost = 0,
        volume_impact_pct = 0
        filled_price_impact_pct = 0
        filled_price_impact_abs = 0
        filled_price_impact_chg_dlr = 0 
        actual_filled_price = 0
        exited = False
        
########################
###### Scenario Situation 1/2 - single-dirction strategy

        if long_short_both == False: # there's only one trading direction:
        
            for i, (ts, signl) in enumerate(signal_updates.items()):            
                # update each profit and initital ticker price
                if i:
                    each_profit = PortfolioCalculation.profit(init_ticker_price, ticker_price_updates[ts], abs(init_ticker_qty), strategy_drct)
                else:
                    each_profit = 0  
                init_ticker_price = ticker_price_updates[ts]
                
                # discuss if the signal changes: 
                if signl != init_signal:   #signal starts to change, meaning we need to enter or exit positions
                    
                    if signl == enter_signal:  #only need to update position's qty here
                        updated_qty = signl*PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization) ################## 'ticker_qty' that the commission package needs
                        entered = True
                        init_ticker_qty = updated_qty 
                        cum_trades = 1 + cum_trades
                        filled_price_impact_pct = (self.price_impact_constant)*(abs(init_ticker_qty)/signal_dataframe['total_quantity'][ts])*(abs(init_ticker_qty)/signal_dataframe['total_quantity'][ts])
                        filled_price_impact_abs = (self.price_impact_constant)*(abs(init_ticker_qty)/signal_dataframe['total_quantity'][ts])*(abs(init_ticker_qty)/signal_dataframe['total_quantity'][ts])*init_ticker_price
                        filled_price_impact_chg_dlr = filled_price_impact_abs if strategy_drct == 'L' else (-filled_price_impact_abs if strategy_drct == 'S' else 0) 
                        actual_filled_price = filled_price_impact_chg_dlr + init_ticker_price
                        each_commission_cost = ib_equity_commissions(abs(init_ticker_qty), ticker_price_updates[ts])
                        cum_commissions += each_commission_cost 
                        
                        
                    elif signl == exit_signal:
                        updated_qty = PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization) ################## 'ticker_qty' that the commission package needs
                        exited = True
                        init_ticker_qty = updated_qty
                        cum_trades = 1 + cum_trades
                        filled_price_impact_pct = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])
                        filled_price_impact_abs = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*init_ticker_price
                        filled_price_impact_chg_dlr = filled_price_impact_abs if strategy_drct == 'L' else (-filled_price_impact_abs if strategy_drct == 'S' else 0) 
                        actual_filled_price = filled_price_impact_chg_dlr + init_ticker_price
                        each_commission_cost = ib_equity_commissions(abs(updated_qty), ticker_price_updates[ts])
                        cum_commissions += each_commission_cost                        
                        init_ticker_qty = 0 
                        
                    
                if exited:             # if there's no position, update qty back to 0
                    init_ticker_qty = 0
            
                    exited = False     # update exit status back to initialization of being False, cannot change to True or the traded qty in position will be all 0
#                     init_signal = signl
#                     cum_trades = cum_trades
#                     volume_impact_pct = init_ticker_qty/signal_dataframe['total_quantity'][ts]
#                     ################# Adding fourth trade stats: 
#                     filled_price_impact_pct = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])
#                     ################# Adding fifth trade stats: 
#                     filled_price_impact_abs = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*init_ticker_price
#                     ################# Adding sixth trade stats:
#                     filled_price_impact_chg_dlr = filled_price_impact_abs if strategy_drct == 'L' else (-filled_price_impact_abs if strategy_drct == 'S' else 0) 
#                     ################# Adding seventh trade stats:
#                     actual_filled_price = filled_price_impact_chg_dlr + init_ticker_price
                    
            
                each_row = {
                    'timestamp':ts, # save updated index
                    'ticker_price':ticker_price_updates[ts], # save updated price
                    'signal':signl, 
                    'ticker_qty':int(init_ticker_qty), # save updated qty
                    'profit':each_profit, # save updated profit
                    'cum_trades':cum_trades, 
                    'commissions':each_commission_cost,
                    'cum_commissions': cum_commissions,
                    'mkt_qty':signal_dataframe['total_quantity'][ts],
                    'mkt_volume':signal_dataframe['total_volume'][ts],
                    'volume_impact_pct':int(init_ticker_qty)/signal_dataframe['total_quantity'][ts],
                    'filled_price_impact_pct':filled_price_impact_pct,
                    'filled_price_impact_abs':filled_price_impact_abs,
                    'filled_price_impact_chg_dlr':filled_price_impact_chg_dlr, 
                    'actual_filled_price': actual_filled_price
                }

                self.data.append(each_row)
                init_signal = signl
                 # At last, update the signal_0 to this row's signal, which is 'signl'

            data_df = pd.DataFrame(self.data)
            data_df.index = data_df['timestamp']

            # Connect data_df with the later df2:



            # copy df1 to df2, creating a new dataframe for storing tearsheet data for printing
            df2 = data_df
#             df2['each_profit'] = data_df['profit']
            df2['cum_profit'] = data_df.profit.cumsum() + self.tradable_capital
            df2['each_return_pct'] = df2.cum_profit.pct_change().fillna(0)
            df2['cum_return_pct'] = (df2['each_return_pct'] + 1).cumprod()
            df2['net_cum_profit'] = df2['cum_profit'] - df2['cum_commissions'] - df2['filled_price_impact_abs'].cumsum()
            df2['net_each_return_pct'] = df2['net_cum_profit'].pct_change().fillna(0)
            df2['net_cum_return_pct'] = (df2['net_each_return_pct'] + 1).cumprod()
            df2['commission_fees_impact_level'] = df2['cum_commissions']/df2['cum_profit']
            
            df2 = df2.dropna(0)
            
            
            
########################
###### Strategy Scenario Situation 2/2 - both-direction strategy:  

        elif long_short_both == True:
            strategy_drct = None
            for i, (ts, signl) in enumerate(signal_updates.items()):
                if i and strategy_drct != None:
                    each_profit = PortfolioCalculation.profit(init_ticker_price, ticker_price_updates[ts], abs(init_ticker_qty), strategy_drct)
                else:
                    each_profit = 0  
                init_ticker_price = ticker_price_updates[ts]
                
                # Normally, the signal switches from 1 to 0 and then to -1, and vice versa, but another situation will not be avoided where signal will be changed directly changed from 1 to -1 or vice versa.
                
                if signl != init_signal: 
                    
                    if signl == long_signal: 
                        updated_qty = PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization)
                        entered = True
                        init_ticker_qty = updated_qty
                        cum_trades = 1 + cum_trades   
                        filled_price_impact_pct = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])
                        filled_price_impact_abs = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*init_ticker_price
                        drct = 'L'
                        filled_price_impact_chg_dlr = filled_price_impact_abs if strategy_drct == 'L' else 0
                        actual_filled_price = filled_price_impact_chg_dlr + init_ticker_price                    
                        each_commission_cost = ib_equity_commissions(abs(updated_qty), ticker_price_updates[ts])
                        cum_commissions += each_commission_cost
                           
                    
                    elif signl == short_signal:
                        updated_qty = signl*PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization)
                        entered = True
                        init_ticker_qty = updated_qty
                        cum_trades = 1 + cum_trades 
                        filled_price_impact_pct = (self.price_impact_constant)*(abs(init_ticker_qty)/signal_dataframe['total_quantity'][ts])*(abs(init_ticker_qty)/signal_dataframe['total_quantity'][ts])
                        filled_price_impact_abs = (self.price_impact_constant)*(abs(init_ticker_qty)/signal_dataframe['total_quantity'][ts])*(abs(init_ticker_qty)/signal_dataframe['total_quantity'][ts])*init_ticker_price
                        drct = 'S'
                        filled_price_impact_chg_dlr = -filled_price_impact_abs if strategy_drct == 'S' else 0
                        actual_filled_price = filled_price_impact_chg_dlr + init_ticker_price                    
                        each_commission_cost = ib_equity_commissions(abs(updated_qty), ticker_price_updates[ts])
                        cum_commissions += each_commission_cost
                        
                    
                    # Closing the position in this case is trickier since we need to dicuss which direction to need close against
                    elif signl == exit_signal:
                        if init_signal == 1: #  if we need to close the position by selling the long position, the trade direction is "S"
                            updated_qty = PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization)
                            exited = True
                            init_ticker_qty = updated_qty
                            cum_trades = 1 + cum_trades 
                            filled_price_impact_pct = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])
                            filled_price_impact_abs = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*init_ticker_price
                            drct = 'S'
                            filled_price_impact_chg_dlr = -filled_price_impact_abs if strategy_drct == 'S' else 0
                            actual_filled_price = filled_price_impact_chg_dlr + init_ticker_price
                            each_commission_cost = ib_equity_commissions(abs(updated_qty), ticker_price_updates[ts])
                            cum_commissions += each_commission_cost
                            init_ticker_qty = 0
                            
                            
                        elif init_signal == -1: #  if we need to close the position by buying back the short position, the trade direction is "L"
                            updated_qty = PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization)
                            exited = True
                            init_ticker_qty = updated_qty
                            cum_trades = 1 + cum_trades 
                            filled_price_impact_pct = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])
                            filled_price_impact_abs = (self.price_impact_constant)*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*(init_ticker_qty/signal_dataframe['total_quantity'][ts])*init_ticker_price
                            drct = 'L'
                            filled_price_impact_chg_dlr = filled_price_impact_abs if strategy_drct == 'L' else 0
                            actual_filled_price = filled_price_impact_chg_dlr + init_ticker_price
                            each_commission_cost = ib_equity_commissions(abs(updated_qty), ticker_price_updates[ts])
                            cum_commissions += each_commission_cost
                            init_ticker_qty = 0
                        
                    strategy_drct = drct
                    
                    

                if exited:             # if there's no position, update qty back to 0
                    exited = False 
                    init_ticker_qty = 0
                        # update exit status back to initialization of being False, cannot change to True or the traded qty in position will be all 0 
                    

                    
                each_row = {
                    'timestamp':ts, # save updated index
                    'ticker_price':ticker_price_updates[ts], # save updated price
                    'signal':signl, # save previous signal
                    'ticker_qty':int(init_ticker_qty), # save updated qty
                    'profit':each_profit, # save updated profit
                     ################## Add more details regarding the trading stats and market impact.
                    'cum_trades':cum_trades, 
                    'commissions':each_commission_cost,
                    'cum_commissions': cum_commissions,
                    'mkt_qty':signal_dataframe['total_quantity'][ts],
                    'mkt_volume':signal_dataframe['total_volume'][ts],
#                     'volume_impact_pct':volume_impact_pct,
                    'volume_impact_pct':int(init_ticker_qty)/signal_dataframe['total_quantity'][ts],
                    'filled_price_impact_pct':filled_price_impact_pct,
                    'filled_price_impact_abs':filled_price_impact_abs,
                    'filled_price_impact_chg_dlr':filled_price_impact_chg_dlr, 
                    'actual_filled_price': actual_filled_price,
                }

                self.data.append(each_row)
                init_signal = signl 
                 # At last, update the signal_0 to this row's signal, which is 'signl'

            data_df = pd.DataFrame(self.data)
            data_df.index = data_df['timestamp']


            # copy df1 to df2, creating a new dataframe for storing tearsheet data for printing
            df2 = data_df
#             df2['each_profit'] = data_df['profit']
            df2['cum_profit'] = data_df.profit.cumsum() + self.tradable_capital
            df2['each_return_pct'] = df2.cum_profit.pct_change().fillna(0)
            df2['cum_return_pct'] = (df2['each_return_pct'] + 1).cumprod()
            df2['net_cum_profit'] = df2['cum_profit'] - df2['cum_commissions'] - df2['filled_price_impact_abs'].cumsum()
            df2['net_each_return_pct'] = df2['net_cum_profit'].pct_change().fillna(0)
            df2['net_cum_return_pct'] = (df2['net_each_return_pct'] + 1).cumprod()
            df2['commission_fees_impact_level'] = df2['cum_commissions']/df2['cum_profit']

            df2 = df2.dropna(0)
            
        return df2



class RiskMetrics(object):
    '''
    The purpose of building this RiskMetrics class is to calculate and prep various risk metrics for us to call   
    '''
    ##### 1. Gross Sharpe & Net Sharpe (Done) #####   

    def annualized_sharpe_ratio(strategy_ret, freq_multiplier):
        """
        Documentation:
        ---------
        Strategy_ret: Series | Numpy Array
        Freq_multiplier explanation: below are the time series frequency we get for our data with its corresponding frequency
        multiplier we will be using, so that we can deal with any frequency of data, from second to daily data. Created by Brad. 
        
        Monthly_Data_Freq_multiplier: 12
        Weekly_Data_Freq_multiplier: 52
        Daily_Data_Freq_multiplier: 252, 
        Hourly_Data_Freq_multiplier: 252*6.5,  
        Minutely_Data_Freq_multipler: 252*6.5*60 
        Second_Data_Freq_multipler: 252*6.5*60*60
        """
        er = np.mean(strategy_ret)
        return np.sqrt(freq_multiplier) * er / np.std(strategy_ret, ddof=1)

    def sharpe_ratio_with_any_freq_ts(strategy_ret):
        er = np.mean(strategy_ret)
        return er / np.std(strategy_ret, ddof=1)


    def net_sharpe_ratio(net_each_return_pct, freq_multiplier):
        er = np.mean(net_each_return_pct)
        return np.sqrt(freq_multiplier) * er / np.std(net_each_return_pct, ddof=1)


    ##### 2. Gross Sortino & Net Sortino (Done) #####

    def sortino_ratio_with_any_freq_ts(strategy_ret):
        r = np.asarray(strategy_ret)
        diff = (0 - r).clip(min=0)
        downside_std =  ((diff**2).sum() / diff.shape[0])**(1/2)
        er = np.mean(strategy_ret)
        return er / downside_std

    def annualized_sortino_ratio(strategy_ret, freq_multiplier):
        """
        Documentation:
        ---------
        Strategy_ret: Series | Numpy Array  
        Freq_multiplier explanation: same as above in the sharpe ratio part: 
        Daily_Data_Freq_multiplier: 252, 
        Hourly_Data_Freq_multiplier: 252*6.5,  
        Minutely_Data_Freq_multipler: 252*6.5*60, 
        Second_Data_Freq_multipler: 252*6.5*60*60
        """

        r = np.asarray(strategy_ret)
        diff = (0 - r).clip(min=0)
        downside_std =  ((diff**2).sum() / diff.shape[0])**(1/2)
        er = np.mean(strategy_ret)
        return np.sqrt(freq_multiplier) * er / downside_std

    def net_sortino_ratio(net_each_return_pct, freq_multiplier):
        r = np.asarray(net_each_return_pct)
        diff = (0 - r).clip(min=0)
        downside_std =  ((diff**2).sum() / diff.shape[0])**(1/2)
        er = np.mean(net_each_return_pct)
        return np.sqrt(freq_multiplier) * er / downside_std



    ##### 3. Gross Alpha Beta & Net Alpha Beta (Done) #####

    def alpha_beta(strategy_ret, benchmark_ret, benchmark_str, show_all):
        # set benchmark's constant
        X = sm.add_constant(benchmark_ret) 
        # y is the values of returns of the strategy
        y = strategy_ret

        # creating the Ordinary Least Square model to get beta and alpha between strategy and benchmark return
        model = sm.OLS(y,X).fit()
        beta = model.params[benchmark_str]
        alpha = model.params["const"]
        # If we want to see all stats summary table and graph, we make show_all == True, 
        # otherwise it will only return alpha and beta.
        if show_all == False:
            print ("Gross Jensen's Constant: {:0.04f}".format(alpha))
            print ('Gross Beta Against Benchmark: {:0.04f}'.format(beta))
        else:
            print ("Gross Jensen's Constant: {:0.04f}".format(alpha))
            print ('Gross Beta Against Benchmark: {:0.04f}'.format(beta))
            print (model.summary())

            # If show_all == True, prep for the alpha beta graph
            fig, ax1 = plt.subplots(1,figsize=(30,6))
            ax1.scatter(benchmark_ret, strategy_ret,label= "Gross Strategy Returns", color='blue', edgecolors='none', alpha=0.7)
            ax1.grid(True)
            ax1.set_xlabel("Benchmark's Returns")
            ax1.set_ylabel("Gross Strategy's Returns")

            # create X points of the line, by using the min and max points to generate sequence
            line_x = np.linspace(benchmark_ret.min(), benchmark_ret.max())

            # generate y points by multiplying x by the slope
            ax1.plot(line_x, line_x*model.params[benchmark_str], color="red", label="beta")

            # add legend
            ax1.legend(loc='upper center', ncol=2, fontsize='large')

            plt.show()


    def net_alpha_beta(net_each_return_pct, benchmark_ret, benchmark_str, show_all):

        # set benchmark's constant
        X = sm.add_constant(benchmark_ret) 
        # y is the values of returns of the strategy
        y = net_each_return_pct

        # creating the Ordinary Least Square model to get beta and alpha between strategy and benchmark return
        model = sm.OLS(y,X).fit()
        beta = model.params[benchmark_str]
        alpha = model.params["const"]
        # If we want to see all stats summary table and graph, we make show_all == True, 
        # otherwise it will only return alpha and beta.
        if show_all == False:
            print ("Net Jensen's Constant: {:0.04f}".format(alpha))
            print ('Net Beta Against Benchmark: {:0.04f}'.format(beta))
        else:
            print ("Net Jensen's Constant: {:0.04f}".format(alpha))
            print ('Net Beta Against Benchmark: {:0.04f}'.format(beta))
            print (model.summary())

            # If show_all == True, prep for the alpha beta graph
            fig, ax1 = plt.subplots(1,figsize=(30,6))
            ax1.scatter(benchmark_ret, net_each_return_pct,label= "Net Strategy Returns", color='blue', edgecolors='none', alpha=0.7)
            ax1.grid(True)
            ax1.set_xlabel("Benchmark's Returns")
            ax1.set_ylabel("Net Strategy's Returns")

            # create X points of the line, by using the min and max points to generate sequence
            line_x = np.linspace(benchmark_ret.min(), benchmark_ret.max())

            # generate y points by multiplying x by the slope
            ax1.plot(line_x, line_x*model.params[benchmark_str], color="red", label="beta")

            # add legend
            ax1.legend(loc='upper center', ncol=2, fontsize='large')

            plt.show()


    ##### 4. Gross gain loss ratio & Net gain loss ratio (Done) #####

    def gain_loss_ratio(strategy_ret):
        """Upside Performnace vs. Downside Performance

        Lower than 1: Upside movement is less than Downside movement
        Higher than 1: Upside movement is less than Downside movement
        """
        r = np.asarray(strategy_ret)
        diff_dn = (0 - r).clip(min = 0)
        diff_up = (r - 0).clip(min = 0)
        downside_std =  ((diff_dn**2).sum() / diff_dn.shape[0])**(1/2)
        upside_std = ((diff_up**2).sum() / diff_up.shape[0])**(1/2)
        return upside_std / downside_std


    def net_gain_loss_ratio(net_each_return_pct):
        """Upside Performnace vs. Downside Performance

        Lower than 1: Upside movement is less than Downside movement
        Higher than 1: Upside movement is less than Downside movement
        """
        r = np.asarray(net_each_return_pct)
        diff_dn = (0 - r).clip(min = 0)
        diff_up = (r - 0).clip(min = 0)
        downside_std =  ((diff_dn**2).sum() / diff_dn.shape[0])**(1/2)
        upside_std = ((diff_up**2).sum() / diff_up.shape[0])**(1/2)
        return upside_std / downside_std


    ##### 5. Gross water mark & Net water mark (Done) #####

    def water_mark(pnl, how='high'):
        """Accumulative Maximum/Minimum of a Series

        Parameter
        ---------
        pnl: Pandas Series
            - index of timestamp
            - period percentagized returns

        Return
        ------
        Series
        """
        mark = np.maximum if how == 'high' else np.minimum
        return mark.accumulate(pnl.fillna(pnl.min()))


    def net_water_mark(net_cum_return_pct, how='high'):
        """Accumulative Maximum/Minimum of a Series

        Parameter
        ---------
        pnl: Pandas Series
            - index of timestamp
            - period percentagized returns

        Return
        ------
        Series
        """
        mark = np.maximum if how == 'high' else np.minimum
        return mark.accumulate(net_cum_return_pct.fillna(net_cum_return_pct.min()))


    ##### 6. Gross DD & Net DD (Done) #####

    def drawdown(pnl, how='high'):
        """
        - Calcualte the largest peak-to-through drawdown of the PnL Curve
        - As well as the Duration of the drawdown

        Parameter
        ---------
        pnl: Pandas Series 
            - index of timestamp
            - period percentagized returns

        Return
        ------
        drawdown
        """
        mark = np.maximum if how == 'high' else np.minimum
        watermark = mark.accumulate(pnl.fillna(pnl.min()))
        dd = 1 - (pnl / watermark.shift(1))
        dd.ix[dd < 0] = 0
        return dd
 
    def net_drawdown_dur(net_cum_return_pct, how = 'high'):    
        """Accumulative Drawdown Duration

        Parameter
        ---------
        pnl: Pandas Series 
        - index of timestamp
        - period percentagized returns


        Theory
        ------
        1. Get the Drawdown percentage series
        2. Convert anything that is not 0 to 1 as boolean type
        3. Then find periodic drawdown period
        - by comparising on changing in boolean value
        - cumsum to get a series that is labeled with drawdown period
        4. Group by each drawdown period using count size

        Return
        -------
        Integer, maximum period of drawdown period length
        """
        # Get Drawdown Max Duration
        mark = np.maximum if how == 'high' else np.minimum
        watermark = mark.accumulate(net_cum_return_pct.fillna(net_cum_return_pct.min()))
        dd = 1 - (net_cum_return_pct / watermark.shift(1))
        dd.ix[dd < 0] = 0
        drawdown = dd
        sr = drawdown[1:].astype(bool)
        return sr.groupby((sr != sr.shift()).cumsum()).size()

    ##### 7. Gross max_DD & Net max_DD (Done) #####

    def max_drawdown(pnl, how ='high'):
        """Max drawdown Percentage in the trading period

        Parameter
        ---------
        pnl: Pandas Series 
            - index of timestamp
            - period percentagized returns

        Rerturn
        -------
        FLoat, maximum drawdown percentage
        """
        mark = np.maximum if how == 'high' else np.minimum
        watermark = mark.accumulate(pnl.fillna(pnl.min()))
        dd = 1 - (pnl / watermark.shift(1))
        dd.ix[dd < 0] = 0
        drawdown = dd
        return drawdown.max()


    def net_max_drawdown(net_cum_return_pct, how ='high'):
        """Max drawdown Percentage in the trading period

        Parameter
        ---------
        pnl: Pandas Series 
            - index of timestamp
            - period percentagized returns

        Rerturn
        -------
        FLoat, maximum drawdown percentage
        """
        mark = np.maximum if how == 'high' else np.minimum
        watermark = mark.accumulate(net_cum_return_pct.fillna(net_cum_return_pct.min()))
        dd = 1 - (net_cum_return_pct / watermark.shift(1))
        dd.ix[dd < 0] = 0
        drawdown = dd
        return drawdown.max()



    ##### 8. Gross max_DD_duration & Net max_DD_duration (Done) #####

    def max_drawdown_dur(pnl, how='high'):
        # Get Drawdown Max Duration
        mark = np.maximum if how == 'high' else np.minimum
        watermark = mark.accumulate(pnl.fillna(pnl.min()))
        dd = 1 - (pnl / watermark.shift(1))
        dd.ix[dd < 0] = 0
        drawdown = dd
        sr = drawdown[1:].astype(bool)
        drawdown_dur =  sr.groupby((sr != sr.shift()).cumsum()).size()
        return drawdown_dur.max()


    def net_max_drawdown_dur(net_cum_return_pct, how='high'):
        # Get Drawdown Max Duration
        mark = np.maximum if how == 'high' else np.minimum
        watermark = mark.accumulate(net_cum_return_pct.fillna(net_cum_return_pct.min()))
        dd = 1 - (net_cum_return_pct / watermark.shift(1))
        dd.ix[dd < 0] = 0
        drawdown = dd
        sr = drawdown[1:].astype(bool)
        drawdown_dur =  sr.groupby((sr != sr.shift()).cumsum()).size()
        return drawdown_dur.max()


    ##### 9. Gross vols & Net vols (Done) #####

    def annual_volatility(strategy_ret, freq_multiplier):
        """
        Documentation:
        ---------
        Strategy_ret: Series | Numpy Array  
        Freq_multiplier explanation: 
        Daily_Data_Freq_multiplier: 252, 
        Hourly_Data_Freq_multiplier: 252*6.5,  
        Minutely_Data_Freq_multipler: 252*6.5*60, 
        Second_Data_Freq_multipler: 252*6.5*60*60
        """
        annual_std = np.std(strategy_ret)*np.sqrt(freq_multiplier)
        daily_std = annual_std/np.sqrt(252)
        return annual_std

    def net_annual_volatility(net_each_return_pct, freq_multiplier):
        """
        Documentation:
        ---------
        Strategy_ret: Series | Numpy Array  
        Freq_multiplier explanation: 
        Daily_Data_Freq_multiplier: 252, 
        Hourly_Data_Freq_multiplier: 252*6.5,  
        Minutely_Data_Freq_multipler: 252*6.5*60, 
        Second_Data_Freq_multipler: 252*6.5*60*60
        """
        annual_std = np.std(net_each_return_pct)*np.sqrt(freq_multiplier)
        daily_std = annual_std/np.sqrt(252)
        return annual_std


    def daily_volatility(strategy_ret, freq_multiplier):
        annual_std = np.std(strategy_ret)*np.sqrt(freq_multiplier)
        daily_std = annual_std/np.sqrt(252)
        return daily_std


    def net_daily_volatility(net_each_return_pct, freq_multiplier):
        annual_std = np.std(net_each_return_pct)*np.sqrt(freq_multiplier)
        daily_std = annual_std/np.sqrt(252)
        return daily_std


    def vol_with_random_ts(strategy_ret):
        return np.std(strategy_ret)


    def net_vol_with_random_ts(net_each_return_pct):
        return np.std(net_each_return_pct)



class TearSheet(object):

    """
    The class helps return all strategy risk metrcis stats as well as stats plots
    """
    def __init__(self, core_bt_df2_bm, freq_multiplier, show_all, benchmark_rdnm_str, benchmark_str, rolling_window):
        
        self.core_bt_df2_bm = core_bt_df2_bm
        self.ticker_qty = core_bt_df2_bm.ticker_qty
        self.net_each_return_pct = core_bt_df2_bm.net_each_return_pct
        self.cum_profit = core_bt_df2_bm.cum_profit
        self.cum_commissions = core_bt_df2_bm.cum_commissions
        self.net_cum_profit = core_bt_df2_bm.net_cum_profit
        self.net_cum_return_pct = core_bt_df2_bm.net_cum_return_pct
        self.cum_trades = core_bt_df2_bm.cum_trades
        self.volume_impact_pct = core_bt_df2_bm.volume_impact_pct
        self.filled_price_impact_pct = core_bt_df2_bm.filled_price_impact_pct
        self.strategy_ret = core_bt_df2_bm['each_return_pct']
        self.freq_multiplier = freq_multiplier
        self.pnl = core_bt_df2_bm['cum_return_pct']
        self.benchmark_ret = core_bt_df2_bm['benchmark_ret']
        self.benchmark_str = benchmark_str
        self.show_all = show_all
        self.how = 'high'
        self.cum_slippage = core_bt_df2_bm.filled_price_impact_abs.cumsum()
        self.volume_limit = 0.025
        self.benchmark_cum_ret = core_bt_df2_bm['benchmark_cum_ret']
        self.benchmark_rdnm_str = benchmark_rdnm_str
        self.rolling_window = rolling_window
        self.commission_fees_impact_level = core_bt_df2_bm.commission_fees_impact_level
        # Federal funds rate set as risk-free rate: https://fred.stlouisfed.org/series/FEDFUNDS


    def print_trade_risk_metrics(self): 
               
##### Calculate gross watermark and gross drawdown #####
        mark = np.maximum if self.how == 'high' else np.minimum
        # Calculating water_mark_sr
        water_mark_sr = mark.accumulate(self.pnl.fillna(self.pnl.min()))
        # Calculating drawdown_sr
        dd = 1 - (self.pnl / water_mark_sr.shift(1))
        dd.ix[dd < 0] = 0
        drawdown_sr = dd

##### Calculate net watermark and net drawdown #####
        net_mark = np.maximum if self.how == 'high' else np.minimum
        # Calculating water_mark_sr
        net_water_mark_sr = mark.accumulate(self.net_cum_return_pct.fillna(self.net_cum_return_pct.min()))
        # Calculating drawdown_sr
        net_dd = 1 - (self.net_cum_return_pct / net_water_mark_sr.shift(1))
        net_dd.ix[net_dd < 0] = 0
        net_drawdown_sr = net_dd


        fig = plt.figure(figsize = (30, 100))
        ax1 = fig.add_subplot(14, 1, 1)
        ax2 = fig.add_subplot(14, 1, 2)
        ax3 = fig.add_subplot(14, 1, 3)
        ax4 = fig.add_subplot(14, 1, 4)
        ax5 = fig.add_subplot(14, 1, 5)
        ax6 = fig.add_subplot(14, 1, 6)
        ax7 = fig.add_subplot(14, 1, 7)
        ax8 = fig.add_subplot(14, 1, 8)
        ax9 = fig.add_subplot(14, 1, 9)
        ax10 = fig.add_subplot(14, 1, 10)
        ax11 = fig.add_subplot(14, 1, 11)
        ax12 = fig.add_subplot(14, 1, 12)
        ax13 = fig.add_subplot(14, 1, 13)
        ax14 = fig.add_subplot(14, 1, 14)


##### Gross rolling std plotting & Net rolling std plotting #####

        # calculate gross rolling volatility
        rolling_std = self.strategy_ret.rolling(self.rolling_window).std()
        annual_rolling_std = rolling_std*np.sqrt(self.freq_multiplier)

        # calculate net rolling volatility
        net_rolling_std = self.net_each_return_pct.rolling(self.rolling_window).std()
        net_annual_rolling_std = net_rolling_std*np.sqrt(self.freq_multiplier)

##### Gross rolling sharpe plotting & Net rolling sharpe plotting #####

        # calculate gross rolling annual sharpe
        rolling_er = self.strategy_ret.ewm(self.rolling_window).mean()
        rolling_sharpe = rolling_er / rolling_std

        # Calculate net rolling annual sharpe
        net_rolling_er = self.net_each_return_pct.ewm(self.rolling_window).mean()
        net_rolling_sharpe = net_rolling_er / net_rolling_std

##### Calculate prepping for gross strategy's Alpha, not Jensen's constant #####
        # set benchmark's constant
        X1 = sm.add_constant(self.benchmark_ret) 
        # y is the values of returns of the strategy
        y1 = self.strategy_ret

        # creating the Ordinary Least Square model to get beta and alpha between strategy and benchmark return
        model1 = sm.OLS(y1,X1).fit()
        beta1 = model1.params[self.benchmark_str]
        alpha1 = model1.params["const"]

        
##### Calculate prepping for net strategy's Alpha, not Jensen's constant #####
        # set benchmark's constant
        X2 = sm.add_constant(self.benchmark_ret) 
        # y is the values of returns of the strategy
        y2 = self.net_each_return_pct

        # creating the Ordinary Least Square model to get beta and alpha between strategy and benchmark return
        model2 = sm.OLS(y2,X2).fit()
        beta2 = model2.params[self.benchmark_str]
        alpha2 = model2.params["const"]

        gross_strategy_alpha = (self.pnl.values[-1]/self.pnl.values[0]-1) - (self.benchmark_cum_ret[-1]-1)*beta1
        net_strategy_alpha = (self.net_cum_return_pct[-1]/self.net_cum_return_pct[0]-1) - (self.benchmark_cum_ret[-1]-1)*beta2


        print ("-"*60)
        print ("Trading Metrics and Stats")
        print ("-"*30)

        self.core_bt_df2_bm.cum_profit.plot(ax = ax1, color = 'green', lw=1)
        (self.benchmark_cum_ret*self.net_cum_profit[0]).plot(ax = ax1, color = 'red', lw = 1)
        ax1.set_ylabel('Cumulative Profits ($)', fontsize = 16)
        ax1.set_title('Gross Backtested Equity Curve', fontsize = 16)
        ax1.set_xlabel('none').set_visible(False)
        print ("Total Gross Net Liquid Asset, in $: {:0.02f}".format(self.core_bt_df2_bm.cum_profit[-1]))
        print ("Total Gross Return: {:0.02f}%".format((self.core_bt_df2_bm.cum_return_pct[-1]-1)*100))
        print ("Total Net Return: {:0.02f}%".format(100*(self.net_cum_return_pct[-1]/self.net_cum_return_pct[0]-1)))
        print ("Total Benchmark Return: {:0.02f}%".format(100*(self.benchmark_cum_ret[-1]-1)))

        self.core_bt_df2_bm.cum_commissions.plot(ax = ax2, color = 'black', lw=1)
        ax2.set_ylabel('Cumulative Commissions ($)', fontsize = 16)
        ax2.set_title('Trading Commissions', fontsize = 16)
        ax2.set_xlabel('none').set_visible(False)
        print ("Total Commission Fees, in $: {:0.02f}".format(self.core_bt_df2_bm.cum_commissions[-1]))

        self.core_bt_df2_bm.net_cum_profit.plot(ax = ax3, color = 'purple', lw=1)
        (self.benchmark_cum_ret*self.net_cum_profit[0]).plot(ax = ax3, color = 'red', lw = 1)
        ax3.set_ylabel('Net Cumulative Profits ($)', fontsize = 16)
        ax3.set_title('Net Backtested Equity Curve', fontsize = 16)
        ax3.set_xlabel('none').set_visible(False)
        print ("Total Net Liquid Asset after Trading Costs, in $: {:0.02f}".format(self.core_bt_df2_bm.net_cum_profit[-1]))
        print ("Trading Costs Impact Level (Commissions + Slippage): {:0.02f}%".format(abs((self.core_bt_df2_bm.cum_commissions[-1] + self.cum_slippage[-1])/(self.core_bt_df2_bm.cum_profit[-1]-self.core_bt_df2_bm.cum_profit[0])*100)))

        self.core_bt_df2_bm.cum_trades.plot(ax = ax4, lw=1)
        ax4.set_ylabel('Cumulative Trades', fontsize = 16)
        ax4.set_title('Cumulative Filled Trades', fontsize = 16)
        ax4.set_xlabel('none').set_visible(False)
        print ("Total Trades: {}".format(self.core_bt_df2_bm.cum_trades[-1])) 

        self.core_bt_df2_bm.volume_impact_pct.plot(ax = ax5, color = 'orange', lw=1)
        ax5.set_ylabel('Trade Volume Impact', fontsize = 16)
        ax5.set_title('Market Impact Per Trade', fontsize = 16)
        ax5.set_xlabel('none').set_visible(False)
        print ("Average Volume Impact Per Trade: {:0.02f}%".format(abs(self.core_bt_df2_bm.volume_impact_pct.mean()*100))) 


        self.core_bt_df2_bm.filled_price_impact_pct.plot(ax = ax6, color = 'blue', lw=1)
        ax6.set_ylabel('Trade Slippage Impact', fontsize = 16)
        ax6.set_title('Slippage Impact Per Trade', fontsize = 16)
        ax6.set_xlabel('none').set_visible(False)
        print ("Average Slippage Impact Per Trade: {:0.04f}%".format(self.core_bt_df2_bm.filled_price_impact_pct.mean()*100))

        # plot gross drawdown
        (-1 * drawdown_sr).plot(ax=ax7, kind='area', color='red', alpha=0.3, lw=1)
        ax7.set_title('Gross Underwater/Drawdown Graph', fontsize = 16)
        ax7.set_xlabel('none').set_visible(False)


        # plot net drawdown
        (-1 * net_drawdown_sr).plot(ax = ax8, kind = 'area', color = 'grey', lw = 1)
        ax8.set_title('Net Underwater/Drawdown Graph', fontsize = 16)
        ax8.set_xlabel('none').set_visible(False)


        # plot gross watermark
        water_mark_sr.plot(ax = ax9, color = 'green', lw = 1)
        ax9.set_title('Gross Watermark Curve', fontsize = 16)
        ax9.set_xlabel('none').set_visible(False)

        # plot net watermark
        net_water_mark_sr.plot(ax = ax10, color = 'magenta', lw = 1)
        ax10.set_title('Net Watermark Curve', fontsize = 16)
        ax10.set_xlabel('none').set_visible(False)

        # plot gross annual rolling volatility
        annual_rolling_std.plot(ax = ax11, color = 'purple', lw = 1)
        ax11.set_title('Annual Rolling Volatility', fontsize = 16)
        ax11.set_xlabel('none').set_visible(False)


        # plot net annual rolling volatility
        net_annual_rolling_std.plot(ax = ax12, color = 'orange', lw = 1)
        ax12.set_title('Net Annual Rolling Volatility', fontsize = 16)
        ax12.set_xlabel('none').set_visible(False)


        self.core_bt_df2_bm.commission_fees_impact_level.plot(ax = ax13, color = 'black', lw = 1)
        ax13.set_title('Commission Fees Impact Level Risk Monitor', fontsize = 16)
        ax13.set_xlabel('none').set_visible(False)

        self.ticker_qty.plot(ax = ax14, color = 'teal', lw = 1)
        ax14.set_title('Position Monitor', fontsize = 16)
        ax14.set_xlabel('none').set_visible(False)


        print ("-"*60)
        print ("Risk Metrics, excluding Trading Costs")
        print ("-"*30)
        # Gross Stats
        RiskMetrics.alpha_beta(self.strategy_ret, self.benchmark_ret, self.benchmark_str, self.show_all)
        print ("Gross Strategy Alpha: {:0.04f}".format(gross_strategy_alpha))
        print ("Gross Sharpe Ratio: {:0.02f}".format(RiskMetrics.annualized_sharpe_ratio(self.strategy_ret, self.freq_multiplier)))
        print ("Gross Sortino Ratio: {:0.02f}".format(RiskMetrics.annualized_sortino_ratio(self.strategy_ret, self.freq_multiplier)))
        print ("Gross Gain Loss Ratio: {:0.02f}".format(RiskMetrics.gain_loss_ratio(self.strategy_ret)))
        print ("Gross Max Drawdown: {:0.02f}%".format(100*RiskMetrics.max_drawdown(self.pnl, self.how)))
        print ("Gross Max Drawdown Duration: {} time unit(s)".format(RiskMetrics.max_drawdown_dur(self.pnl, self.how)))
        print ("Gross Time-Unit Volatility: {:0.02f}%".format(100*RiskMetrics.vol_with_random_ts(self.strategy_ret)))
        print ("Gross Estimated Annual Volatility: {:0.02f}%".format(100*RiskMetrics.annual_volatility(self.strategy_ret, self.freq_multiplier)))
        # print ("Gross Total Strategy Return: {:0.02f}%".format(100*(self.pnl.values[-1]/self.pnl.values[0]-1)))
        print ("Gross Most Recent Time-Range Pnl: {:0.02f}%".format(100*(self.pnl.values[-1]/self.pnl.values[-2]-1)))
        print ("Liquidity Risk Monitor: Average Risk Level: {:0.02f}%".format(0))

        print ("-"*60)
        print ("Net Risk Metrics, including Trading Costs")
        print ("-"*30)
        # Net Stats
        RiskMetrics.net_alpha_beta(self.net_each_return_pct, self.benchmark_ret, self.benchmark_str, self.show_all)
        print ("Net Strategy Alpha: {:0.04f}".format(net_strategy_alpha))
        print ("Net Sharpe Ratio: {:0.02f}".format(RiskMetrics.net_sharpe_ratio(self.net_each_return_pct, self.freq_multiplier)))
        print ("Net Sortino Ratio: {:0.02f}".format(RiskMetrics.net_sortino_ratio(self.net_each_return_pct, self.freq_multiplier)))
        print ("Net Gain Loss Ratio: {:0.02f}".format(RiskMetrics.net_gain_loss_ratio(self.net_each_return_pct)))
        print ("Net Max Drawdown: {:0.02f}%".format(100*RiskMetrics.net_max_drawdown(self.net_cum_return_pct, self.how)))
        print ("Net Max Drawdown Duration: {} time unit(s)".format(RiskMetrics.net_max_drawdown_dur(self.net_cum_return_pct, self.how)))
        print ("Net Time-Unit Volatility: {:0.02f}%".format(100*RiskMetrics.vol_with_random_ts(self.net_each_return_pct)))
        print ("Net Estimated Annual Volatility: {:0.02f}%".format(100*RiskMetrics.net_annual_volatility(self.net_each_return_pct, self.freq_multiplier)))
        # print ("Net Total Strategy Return: {:0.02f}%".format(100*(self.net_cum_return_pct[-1]/self.net_cum_return_pct[0]-1)))
        print ("Net Most Recent Time-Range Pnl: {:0.02f}%".format(100*(self.net_cum_return_pct[-1]/self.net_cum_return_pct[-2]-1)))
        print ("Liquidity Risk Monitor: Average Risk Level: {:0.02f}%".format(abs(self.core_bt_df2_bm.volume_impact_pct.mean()*100)/self.volume_limit))