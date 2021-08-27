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




class BT(object):
    """
    This is the meat of the whole backtesting infrastructure, mainly to prepare the final dataframe with all 
    important columns, including pnl
    
    """
    
    def __init__(self, port_utilization, tradable_capital, margin_leverage):
        self.data = []
        self.tradable_capital = tradable_capital
        self.margin_leverage = margin_leverage
        self.port_utilization = port_utilization
        

    def core_backtester(self, signal_dataframe, df_sgnl_str, df_tkr_price_str, strategy_drct, 
                        long_short_both, enter_signal, exit_signal, long_signal, short_signal):
        
        signal_updates = signal_dataframe[df_sgnl_str]
        ticker_price_updates = signal_dataframe[df_tkr_price_str]
        init_signal = 0
        init_ticker_qty = 0
        init_port_value = 0
        init_ticker_price = None
        exited = False
        
        if long_short_both == False: # if there's only one trading direction:
        
            for i, (ts, signl) in enumerate(signal_updates.items()):
                if i:
                    each_profit = PortfolioCalculation.profit(init_ticker_price, ticker_price_updates[ts], init_ticker_qty, strategy_drct)
                else:
                    each_profit = 0  

                if signl != init_signal:   #signal starts to change, meaning we need to enter or exit positions
                    if signl == enter_signal:  #only need to update position's qty here
                        updated_qty = PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization)
                        entered = True
                        init_ticker_qty = updated_qty
                        
                    elif signl == exit_signal:
                        exited = True

                # After signal changes, after looping through each row, we need to update stock price:
                init_ticker_price = ticker_price_updates[ts]

                if exited:             # if there's no position, update qty back to 0
                    init_ticker_qty = 0
                    exited = False     # update exit status back to initialization of being False

                each_row = {
                    'timestamp':ts, # save updated index
                    'ticker_price':ticker_price_updates[ts], # save updated price
                    'signal':init_signal, # save previous signal
                    'ticker_qty':init_ticker_qty, # save updated qty
                    'profit':each_profit # save updated profit
                }

                self.data.append(each_row)
                init_signal = signl # At last, update the signal_0 to this row's signal, which is 'signl'

            data_df = pd.DataFrame(self.data)
            data_df.index = data_df['timestamp']

            # Connect data_df with the later df2:



            # copy df1 to df2, creating a new dataframe for storing tearsheet data for printing
            df2 = data_df
            df2['each_profit'] = data_df['profit']
            df2['cum_profit'] = data_df.profit.cumsum() + self.tradable_capital
            df2['each_return_pct'] = df2.cum_profit.pct_change().fillna(0)
            df2['cum_return_pct'] = (df2['each_return_pct'] + 1).cumprod()

            df2 = df2.dropna(0)
            
            
        elif long_short_both == True:
            strategy_drct = None
            for i, (ts, signl) in enumerate(signal_updates.items()):
                
                if i and strategy_drct != None:
                    each_profit = PortfolioCalculation.profit(init_ticker_price, ticker_price_updates[ts], init_ticker_qty, strategy_drct)
                else:
                    each_profit = 0  

                # Step 1. Signal Generation
                if signl != init_signal:   #signal starts to change, meaning we need to enter or exit positions
                    if signl == long_signal:  
                        updated_qty = PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization)
                        entered = True
                        init_ticker_qty = updated_qty
                        drct = 'L'   
                    
                    elif signl == short_signal:
                        updated_qty = PortfolioCalculation.quantity(self.tradable_capital, ticker_price_updates[ts], self.port_utilization)
                        entered = True
                        init_ticker_qty = updated_qty
                        drct = 'S'  
                        
                    elif signl == exit_signal:
                        exited = True
                    # Update the most signal and cover previous signal
                    strategy_drct = drct

                # After signal changes, after looping through each row, we need to update stock price:
                init_ticker_price = ticker_price_updates[ts]

                if exited:             # if there's no position, update qty back to 0
                    init_ticker_qty = 0
                    exited = False     # update exit status back to initialization of being False

                each_row = {
                    'timestamp':ts, # save updated index
                    'ticker_price':ticker_price_updates[ts], # save updated price
                    'signal':init_signal, # save previous signal
                    'ticker_qty':init_ticker_qty, # save updated qty
                    'profit':each_profit # save updated profit
                }

                self.data.append(each_row)
                init_signal = signl # At last, update the signal_0 to this row's signal, which is 'signl'

            data_df = pd.DataFrame(self.data)
            data_df.index = data_df['timestamp']

            # Connect data_df with the later df2:



            # copy df1 to df2, creating a new dataframe for storing tearsheet data for printing
            df2 = data_df
            df2['each_profit'] = data_df['profit']
            df2['cum_profit'] = data_df.profit.cumsum() + self.tradable_capital
            df2['each_return_pct'] = df2.cum_profit.pct_change().fillna(0)
            df2['cum_return_pct'] = (df2['each_return_pct'] + 1).cumprod()

            df2 = df2.dropna(0)
            
    # later on connect this BT's result, which is the df2, to RiskMetrics class:
        return df2



class RiskMetrics(object):
    '''
    The purpose of building this RiskMetrics class is to calculate and prep various risk metrics for us to call   
    '''

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
            print ('Strategy Alpha: {:0.02f}'.format(alpha))
            print ('Strategy Beta: {:0.02f}'.format(beta))
        else:
            print ('Strategy Alpha: {:0.02f}'.format(alpha))
            print ('Strategy Beta: {:0.02f}'.format(beta))
            print (model.summary())

            # If show_all == True, prep for the alpha beta graph
            fig, ax1 = plt.subplots(1,figsize=(15,6))
            ax1.scatter(benchmark_ret, strategy_ret,label= "Strategy Returns", color='blue', edgecolors='none', alpha=0.7)
            ax1.grid(True)
            ax1.set_xlabel("Benchmark's Returns")
            ax1.set_ylabel("Strategy's Returns")

            # create X points of the line, by using the min and max points to generate sequence
            line_x = np.linspace(benchmark_ret.min(), benchmark_ret.max())

            # generate y points by multiplying x by the slope
            ax1.plot(line_x, line_x*model.params[benchmark_str], color="red", label="beta")

            # add legend
            ax1.legend(loc='upper center', ncol=2, fontsize='large')

            plt.show()

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
 
    def drawdown_dur(pnl, how = 'high'):    
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
        watermark = mark.accumulate(pnl.fillna(pnl.min()))
        dd = 1 - (pnl / watermark.shift(1))
        dd.ix[dd < 0] = 0
        drawdown = dd
        sr = drawdown[1:].astype(bool)
        return sr.groupby((sr != sr.shift()).cumsum()).size()

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

    def daily_volatility(strategy_ret, freq_multiplier):
        annual_std = np.std(strategy_ret)*np.sqrt(freq_multiplier)
        daily_std = annual_std/np.sqrt(252)
        return daily_std

    def vol_with_random_ts(strategy_ret):
        return np.std(strategy_ret)



class TearSheet(object):

    """
    The class helps return all strategy risk metrcis stats as well as stats plots
    """
    def __init__(self, core_bt_df2_bm, freq_multiplier, show_all, ts_random, benchmark_rdnm_str, benchmark_str, rolling_window):
        
        self.strategy_ret = core_bt_df2_bm['each_return_pct']
        self.freq_multiplier = freq_multiplier
        self.pnl = core_bt_df2_bm['cum_return_pct']
        self.benchmark_ret = core_bt_df2_bm['benchmark_ret']
        self.benchmark_str = benchmark_str
        self.show_all = show_all
        self.how = 'high'
        self.ts_random = ts_random
        self.benchmark_cum_ret = core_bt_df2_bm['benchmark_cum_ret']
        self.benchmark_rdnm_str = benchmark_rdnm_str
        self.rolling_window = rolling_window
        

    def print_risk_metrics(self): 
        # We have two scenarios to deal with: either our timestamp interval interval is the same or different
        if self.ts_random == False:
            print (RiskMetrics.alpha_beta(self.strategy_ret, self.benchmark_ret, self.benchmark_str, self.show_all))
            print ("Annual Sharpe Ratio: {:0.02f}".format(RiskMetrics.annualized_sharpe_ratio(self.strategy_ret, self.freq_multiplier)))
            print ("Annual Sortino Ratio: {:0.02f}".format(RiskMetrics.annualized_sortino_ratio(self.strategy_ret, self.freq_multiplier)))
            print ("Gain Loss Ratio: {:0.02f}".format(RiskMetrics.gain_loss_ratio(self.strategy_ret)))
            print ("Max Drawdown: {:0.02f}".format(RiskMetrics.max_drawdown(self.pnl, self.how)))
            print ("Max Drawdown Duration: {} time unit(s)".format(RiskMetrics.max_drawdown_dur(self.pnl, self.how)))
            print ("Daily Volatility: {:0.02f}%".format(100*RiskMetrics.daily_volatility(self.strategy_ret, self.freq_multiplier)))
            print ("Annual Volatility: {:0.02f}%".format(100*RiskMetrics.annual_volatility(self.strategy_ret, self.freq_multiplier)))
            print ("Total Strategy Return: {:0.02f}%".format(100*(self.pnl.values[-1]/self.pnl.values[0]-1)))
            print ("Most Recent Time-Range Pnl: {:0.02f}%".format(100*(self.pnl.values[-1]/self.pnl.values[-2]-1)))


        else:
            print (RiskMetrics.alpha_beta(self.strategy_ret, self.benchmark_ret, self.benchmark_str, self.show_all))
            print ("Sharpe ratio with any-freq timestamp: {:0.02f}".format(RiskMetrics.sharpe_ratio_with_any_freq_ts(self.strategy_ret)))
            print ("Sortino ratio with any-freq timestamp: {:0.02f}".format(RiskMetrics.sortino_ratio_with_any_freq_ts(self.strategy_ret)))
            print ("Gain Loss Ratio: {:0.02f}".format(RiskMetrics.gain_loss_ratio(self.strategy_ret)))
            print ("Max Drawdown: {:0.02f}".format(RiskMetrics.max_drawdown(self.pnl, self.how)))
            print ("Max Drawdown Duration: {} time unit(s)".format(RiskMetrics.max_drawdown_dur(self.pnl, self.how)))
            print ("Volatility with any-freq timestamp: {:0.02f}%".format(100*RiskMetrics.vol_with_random_ts(self.strategy_ret)))
            print ("Total Strategy Return: {:0.02f}%".format(100*(self.pnl.values[-1]/self.pnl.values[0]-1)))
            print ("Most Recent Time-Range Pnl: {:0.02f}%".format(100*(self.pnl.values[-1]/self.pnl.values[-2]-1)))
            

    def plot_risk_metrics(self):
        
        mark = np.maximum if self.how == 'high' else np.minimum
        # Calculating water_mark_sr
        water_mark_sr = mark.accumulate(self.pnl.fillna(self.pnl.min()))
        
        # Calculating drawdown_sr
        dd = 1 - (self.pnl / water_mark_sr.shift(1))
        dd.ix[dd < 0] = 0
        drawdown_sr = dd

        # plot strategy equity performance curve 
        fig, ax = plt.subplots(figsize = (30,6))
        self.pnl.plot(ax = ax, color = 'blue', lw = 1, label="Strategy Performance")
        self.benchmark_cum_ret.plot(ax = ax, color = 'red', lw =1, label = self.benchmark_rdnm_str)
        ax.set_title('Backtest Equity Curve')

        # plot drawdown
        fig, ax = plt.subplots(figsize = (30, 6))
        (-1 * drawdown_sr).plot(kind='area', color='red', alpha=0.3, lw=1, ax=ax)
        ax.set_title('Underwater/Drawdown Graph')

        # plot watermark
        fig, ax = plt.subplots(figsize = (30, 6))
        water_mark_sr.plot(ax = ax, color = 'green', lw = 1, label = "Strategy Watermark")
        ax.set_title('Watermark Curve')

    def plot_rolling_risk_monitor(self):

        # calculate rolling volatility
        rolling_std = self.strategy_ret.rolling(self.rolling_window).std()
        annual_rolling_std = rolling_std*np.sqrt(self.freq_multiplier)
        daily_rolling_std = annual_rolling_std/np.sqrt(self.freq_multiplier)

        # calculate rolling annual sharpe
        rolling_er = self.strategy_ret.ewm(self.rolling_window).mean()
        rolling_sharpe = np.sqrt(self.freq_multiplier) * rolling_er / rolling_std

        # calculate rolling sharpe with infrequent timestamp and sortino ratio
        rolling_sharpe_random_ts = rolling_er / rolling_std


        # We have two scenarios to deal with: either our timestamp interval interval is the same or different
        if self.ts_random == False: # plot the normal version of graph
            # plot rolling daily volatility
            fig, ax = plt.subplots(figsize = (30, 6))
            daily_rolling_std.plot(ax = ax, color = 'purple', lw = 1, label = "Rolling Volatility")
            ax.set_title('Rolling Volatility')

            # calculate rolling annual sharpe
            fig, ax = plt.subplots(figsize = (30, 6))
            rolling_sharpe.plot(ax = ax, color = 'black', lw = 1, label = "Rolling Annual Sharpe")
            ax.set_title('Rolling Annual Sharpe')


        else:
            # plot rolling random-ts volatility
            fig, ax = plt.subplots(figsize = (30, 6))
            rolling_std.plot(ax = ax, color = 'purple', lw = 1, label = "Rolling Random-Ts Volatility")
            ax.set_title('Rolling Random-Ts Volatility')

            # calculate rolling random-ts sharpe
            fig, ax = plt.subplots(figsize = (30, 4))
            rolling_sharpe_random_ts.plot(ax = ax, color = 'black', lw = 1, label = "Rolling Random-Ts Sharpe")
            ax.set_title('Rolling Random-Ts Sharpe')