import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def annualized_sharpe_ratio(strategy_ret, freq_multiplier):
    """
    Documentation:
    ---------
    Strategy_ret: Series | Numpy Array
    Freq_multiplier explanation:
    Daily_Data_Freq_multiplier: 252, 
    Hourly_Data_Freq_multiplier: 252*6.5,  
    Minutely_Data_Freq_multipler: 252*6.5*60 
    Second_Data_Freq_multipler: 252*6.5*60*60
    """
    er = np.mean(strategy_ret)
    return np.sqrt(freq_multiplier) * er / np.std(strategy_ret, ddof=1)




def alpha_beta(strategy_ret, benchmark_ret, bench_str, show_all):
    X = sm.add_constant(benchmark_ret) # set spy's constant
    # y is the values of the monthly returns of the stock = the first column
    y = strategy_ret

    # creating the model
    model = sm.OLS(y,X).fit()
    beta = model.params[bench_str]
    alpha = model.params["const"]
    # lets see what we got so far
    if show_all == False:
        print ('Strategy Alpha: {:0.02f}'.format(alpha))
        print ('Strategy Beta: {:0.02f}'.format(beta))
    else:
        print ('Strategy Alpha: {:0.02f}'.format(alpha))
        print ('Strategy Beta: {:0.02f}'.format(beta))
        print (model.summary())
        
        fig, ax1 = plt.subplots(1,figsize=(15,6))
        ax1.scatter(benchmark_ret, strategy_ret,label= "Strategy Returns", color='blue', edgecolors='none', alpha=0.7)
        ax1.grid(True)
        ax1.set_xlabel("Benchmark's Returns")
        ax1.set_ylabel("Strategy's Returns")

        # create X points of the line, by using the min and max points to generate sequence
        line_x = np.linspace(benchmark_ret.min(), benchmark_ret.max())

        # generate y points by multiplying x by the slope
        ax1.plot(line_x, line_x*model.params[bench_str], color="red", label="beta")

        # add legend
        ax1.legend(loc='upper center', ncol=2, fontsize='large')
#         # add Beta and R2 stats to plot
#         ax1.text(-0.12, 0.12, 'Beta: %.2f' % beta, fontsize=12)
#         ax1.text(-0.12, 0.09, 'Alpha: %.2f' % alpha, fontsize=12)
#         ax1.text(-0.12, 0.06, 'r^2: %.2f' % rsqr, fontsize=12)


        plt.show()



def annualized_sortino_ratio(strategy_ret, freq_multiplier):
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
    
    r = np.asarray(strategy_ret)
    diff = (0 - r).clip(min=0)
    downside_std =  ((diff**2).sum() / diff.shape[0])**(1/2)
    er = np.mean(strategy_ret)
    return np.sqrt(freq_multiplier) * er / downside_std




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





def drawdown(pnl):
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
    dd = 1 - (pnl / water_mark(pnl, how='high').shift(1))
    dd.ix[dd < 0] = 0
    return dd



def drawdown_dur(pnl):    
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
    - cumsum to get a series that labeled with drawdown period
    4. Group by each drawdown period using count size

    Return
    -------
    Integer, maximum period of drawdown period length
    """
    # Get Drawdown Max Duration
    sr = drawdown(pnl)[1:].astype(bool)
    return sr.groupby((sr != sr.shift()).cumsum()).size()



def max_drawdown(pnl):
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
    return drawdown(pnl).max()




def max_drawdown_dur(pnl):
    # Get Drawdown Max Duration
    return drawdown_dur(pnl).max()



def volatility(strategy_ret, freq_multiplier):
    """
    Documentation:
    ---------
    Strategy_ret: Series | Numpy Array  
    Freq_multiplier explanation: 
    Daily_Data_Freq_multiplier: 252, (annual volatility based on daily return)
    Hourly_Data_Freq_multiplier: 252*6.5,  
    Minutely_Data_Freq_multipler: 252*6.5*60, 
    Second_Data_Freq_multipler: 252*6.5*60*60
    """
    # first let's assume we trade on a daily basis
    
    annual_std = np.std(strategy_ret)*np.sqrt(freq_multiplier)
    daily_std = annual_std/np.sqrt(252)
    return daily_std, annual_std


