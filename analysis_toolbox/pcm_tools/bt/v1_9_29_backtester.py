import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import statsmodels.api as sm

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





class BackTester(object):
    """
    The purpose of building this BactTester is to prepare a system where we can just feed the signal dataframe into and then it 
    can return a dataframe with all the strategy performance columns that we need, including strategy returns, cumulative
    returns, benchmark returns, etc., or the preparation to generate the tearsheet later.   
    """



    # Initialization:
    def param_initialization(df, 
                             df_sgnl_str,
                             df_tkr_price_str,
                             port_utilization, 
                             tradable_capital, 
                             margin_leverage, 
                             ):

        """
        return (data,
                init_signal,
                init_ticker_qty,
                signal_updates,
                init_port_value, 
                ticker_price_updates,
                init_ticker_price, 
                tradable_capital,
                exited,
                margin_leverage, 
                port_utilization
                )
        """

        data = []
        init_signal = 0
        init_ticker_qty = 0
        signal_updates = df[df_sgnl_str]
        init_port_value = 0
        ticker_price_updates = df[df_tkr_price_str]
        init_ticker_price = None
        exited = False

        return (data,
                init_signal,
                init_ticker_qty,
                signal_updates,
                init_port_value, 
                ticker_price_updates,
                init_ticker_price, 
                tradable_capital,
                exited,
                margin_leverage, 
                port_utilization
                )
        
            
    def core_backtester(data,
                    init_signal,
                    init_ticker_qty,
                    signal_updates,
                    init_port_value, 
                    ticker_price_updates,
                    init_ticker_price, 
                    tradable_capital,
                    exited,
                    margin_leverage, 
                    port_utilization):
    
        for i, (ts, signl) in enumerate(signal_updates.items()):
            if i:
                each_profit = PortfolioCalculation.profit(init_ticker_price, ticker_price_updates[ts], init_ticker_qty, 'L')
            else:
                each_profit = 0  

            if signl != init_signal:   #signal starts to change, meaning we need to enter or exit positions
                if signl == 'Enter at Adjusted Close':  #only need to update position's qty here
                    updated_qty = PortfolioCalculation.quantity(tradable_capital, ticker_price_updates[ts], port_utilization)
                    entered = True
                    init_ticker_qty = updated_qty

                elif signl == 'Clear Position by Adjusted Close':
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

            data.append(each_row)
            init_signal = signl # At last, update the signal_0 to this row's signal, which is 'signl'

        return data



class TearSheet(object):

    """
    The class helps return all strategy risk metrcis stats as well as stats plots
    """
    
    def print_risk_metrics(strategy_ret, freq_multiplier, pnl, benchmark_ret, benchmark_str, show_all, how, ts_random): 
        # We have two scenarios to deal with: either our timestamp interval interval is the same or different
        if ts_random == False:
            print (RiskMetrics.alpha_beta(strategy_ret, benchmark_ret, benchmark_str, show_all))
            print ("Annual Sharpe Ratio: {:0.02f}".format(RiskMetrics.annualized_sharpe_ratio(strategy_ret, freq_multiplier)))
            print ("Annual Sortino Ratio: {:0.02f}".format(RiskMetrics.annualized_sortino_ratio(strategy_ret, freq_multiplier)))
            print ("Gain Loss Ratio: {:0.02f}".format(RiskMetrics.gain_loss_ratio(strategy_ret)))
            print ("Max Drawdown: {:0.02f}".format(RiskMetrics.max_drawdown(pnl, how)))
            print ("Max Drawdown Duration: {} days".format(RiskMetrics.max_drawdown_dur(pnl, how)))
            print ("Daily Volatility: {:0.02f}%".format(100*RiskMetrics.daily_volatility(strategy_ret, freq_multiplier)))
            print ("Annual Volatility: {:0.02f}%".format(100*RiskMetrics.annual_volatility(strategy_ret, freq_multiplier)))
            print ("Total Strategy Return: {:0.02f}%".format(100*(pnl[-1]/pnl[0]-1)))
            print ("Most Recent Time-Range Pnl: {:0.02f}%".format(100*(pnl[-1]/pnl[-2]-1)))

        else:
            print (RiskMetrics.alpha_beta(strategy_ret, benchmark_ret, benchmark_str, show_all))
            print ("Sharpe ratio with any-freq timestamp: {:0.02f}".format(RiskMetrics.sharpe_ratio_with_any_freq_ts(strategy_ret)))
            print ("Sortino ratio with any-freq timestamp: {:0.02f}".format(RiskMetrics.sortino_ratio_with_any_freq_ts(strategy_ret)))
            print ("Gain Loss Ratio: {:0.02f}".format(RiskMetrics.gain_loss_ratio(strategy_ret)))
            print ("Max Drawdown: {:0.02f}".format(RiskMetrics.max_drawdown(pnl, how)))
            print ("Max Drawdown Duration: {} days".format(RiskMetrics.max_drawdown_dur(pnl, how)))
            print ("Volatility with any-freq timestamp: {:0.02f}%".format(100*RiskMetrics.vol_with_random_ts(strategy_ret)))
            print ("Total Strategy Return: {:0.02f}%".format(100*(pnl[-1]/pnl[0]-1)))
            print ("Most Recent Time-Range Pnl: {:0.02f}%".format(100*(pnl[-1]/pnl[-2]-1)))
            


    def plot_risk_metrics(pnl, drawdown_sr, water_mark_sr, benchmark_cum_ret, benchmark_rdnm_str):

        # plot strategy equity performance curve 
        fig, ax = plt.subplots(figsize = (30,6))
        pnl.plot(ax = ax, color = 'blue', lw = 1, label="Strategy Performance")
        benchmark_cum_ret.plot(ax = ax, color = 'red', lw =1, label = benchmark_rdnm_str)
        ax.set_title('Backtest Equity Curve')

        # plot drawdown
        fig, ax = plt.subplots(figsize = (30, 6))
        (-1 * drawdown_sr).plot(kind='area', color='red', alpha=0.3, lw=1, ax=ax)
        ax.set_title('Underwater/Drawdown Graph')

        # plot watermark
        fig, ax = plt.subplots(figsize = (30, 6))
        water_mark_sr.plot(ax = ax, color = 'green', lw = 1, label = "Strategy Watermark")
        ax.set_title('Watermark Curve')


    def plot_rolling_risk_monitor(strategy_ret, freq_multiplier, benchmark_ret, rolling_window, ts_random):

        # calculate rolling volatility
        rolling_std = strategy_ret.rolling(rolling_window).std()
        annual_rolling_std = rolling_std*np.sqrt(freq_multiplier)
        daily_rolling_std = annual_rolling_std/np.sqrt(freq_multiplier)

        # calculate rolling annual sharpe
        rolling_er = strategy_ret.ewm(rolling_window).mean()
        rolling_sharpe = np.sqrt(freq_multiplier) * rolling_er / rolling_std

        # calculate rolling sharpe with infrequent timestamp and sortino ratio
        rolling_sharpe_random_ts = rolling_er / rolling_std


        # We have two scenarios to deal with: either our timestamp interval interval is the same or different
        if ts_random == False: # plot the normal version of graph
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










