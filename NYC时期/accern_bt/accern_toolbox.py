import fix_yahoo_finance as yf
from pcm_tools.toolbox import get_yahoo_data
from pandas.tseries.offsets import BMonthEnd
from pykalman import KalmanFilter
import pandas as pd
import glob
import csv


relevance_metrics = [
       'avg_entity_relevance',
       'avg_event_relevance']

probability_metrics = [
       'avg_event_impact_gt_1pct_pos', 
       'avg_event_impact_gt_mu_add_sigma',
       'avg_event_impact_gt_mu_pos_add_2sigma_pos',
       'avg_event_impact_gt_mu_pos_add_sigma_pos',
       'avg_event_impact_lt_1pct_neg',
       'avg_event_impact_lt_mu_neg_sub_2sigma_neg',
       'avg_event_impact_lt_mu_neg_sub_sigma_neg',
       'avg_event_impact_lt_mu_sub_sigma', 
       'avg_event_impact_neg',
       'avg_event_impact_pct_change_avg', 
       'avg_event_impact_pct_change_stdev',
       'avg_event_impact_pos']

exposure_metrics = [
       'avg_story_group_count', 
       'avg_story_group_traffic_sum', 
       'avg_story_traffic']

timeliness_metrics = [
       'avg_entity_author_timeliness_score', 'avg_entity_source_timeliness_score',
       'avg_event_author_timeliness_score','avg_event_source_timeliness_score',
       'avg_overall_author_timeliness_score','avg_overall_source_timeliness_score']

credibility_metrics = [
       'avg_entity_author_republish_score', 'avg_entity_source_republish_score',
       'avg_event_author_republish_score','avg_event_source_republish_score',
       'avg_overall_author_republish_score','avg_overall_source_republish_score',
       'avg_templated_story_score']

directional_metrics = [
       'avg_entity_sentiment', 
       'avg_event_sentiment',
       'avg_story_group_sentiment_avg',
       'avg_story_sentiment']

template_metric = [
       'avg_story_group_sentiment_stdev']



# Make sure in the dataframe the columne 'time' type is string
def execution_QA(df, score_col, check_days, time_col, num_stocks_to_trade, symbol_col):
    # Get the sorted datetime strings:
    timestamps = df[time_col].sort_values(ascending = True).unique()
    
    # get the full stock number lengths of the first three days:
    len_first_day = len(df[df[time_col] == timestamps[:check_days][0]].sort_values(score_col, ascending = False))
    len_second_day = len(df[df[time_col] == timestamps[:check_days][1]].sort_values(score_col, ascending = False))
    len_third_day = len(df[df[time_col] == timestamps[:check_days][2]].sort_values(score_col, ascending = False))
    
    df_len_dict_three_days = {timestamps[:check_days][0]:len_first_day, 
                          timestamps[:check_days][1]: len_second_day, 
                          timestamps[:check_days][2]: len_third_day}
    
    for key, value in sorted(df_len_dict_three_days.items()):
        print(key,'- Total Stock Num:', value)
        
    # Pay attention here, every time we run this, if we want to rerun, need to run the num_stocks_to_trade = the parameter we want to set
    actual_num_stocks_to_trade_dict = {}

    for key, value in sorted(df_len_dict_three_days.items()):
        num_to_trade = num_stocks_to_trade
        if value < 2*num_to_trade:
            num_to_trade = int(value/2)
            print ('{} stock num insufficient! Actual num to trade is: '.format(key), num_to_trade)
        else:
            print ('{} stock num sufficient. Actual num to trade is: '.format(key), num_to_trade)
        actual_num_stocks_to_trade_dict[key] = num_to_trade
        
    day_1_actual_num = actual_num_stocks_to_trade_dict[timestamps[:check_days][0]]
    day_2_actual_num = actual_num_stocks_to_trade_dict[timestamps[:check_days][1]]
    day_3_actual_num = actual_num_stocks_to_trade_dict[timestamps[:check_days][2]]
    
    ticker_to_long_day_1 = list(df[df[time_col] == timestamps[:check_days][0]].sort_values(score_col, ascending = False)[:day_1_actual_num][symbol_col])
    ticker_to_long_day_2 = list(df[df[time_col] == timestamps[:check_days][1]].sort_values(score_col, ascending = False)[:day_2_actual_num][symbol_col])
    ticker_to_long_day_3 = list(df[df[time_col] == timestamps[:check_days][2]].sort_values(score_col, ascending = False)[:day_3_actual_num][symbol_col])
    
    ticker_to_short_day_1 = list(df[df[time_col] == timestamps[:check_days][0]].sort_values(score_col, ascending = False)[-day_1_actual_num:][symbol_col])
    ticker_to_short_day_2 = list(df[df[time_col] == timestamps[:check_days][1]].sort_values(score_col, ascending = False)[-day_2_actual_num:][symbol_col])
    ticker_to_short_day_3 = list(df[df[time_col] == timestamps[:check_days][2]].sort_values(score_col, ascending = False)[-day_3_actual_num:][symbol_col])
    
    # For day 1, the executions include either long or short only
    positions_to_long_day_1 = ticker_to_long_day_1
    positions_to_short_day_1 = ticker_to_short_day_1


    # For day 2, the executions include 1) long, 2) short, 3) close long, 4) close short positions
    positions_to_long_day_2 = ticker_to_long_day_2
    positions_to_short_day_2 = ticker_to_short_day_2

    positions_to_close_long_day_2 = []
    for ticker in positions_to_long_day_1:
        if ticker not in positions_to_long_day_2:
            positions_to_close_long_day_2.append(ticker)

    positions_to_close_short_day_2 = []
    for ticker in positions_to_short_day_1:
        if ticker not in positions_to_short_day_2:
            positions_to_close_short_day_2.append(ticker)


    # For day 3, the executions include 1) long, 2) short, 3) close long, 4) close short positions
    positions_to_long_day_3 = ticker_to_long_day_3
    positions_to_short_day_3 = ticker_to_short_day_3

    positions_to_close_long_day_3 = []
    for ticker in positions_to_long_day_2:
        if ticker not in positions_to_long_day_3:
            positions_to_close_long_day_3.append(ticker)

    positions_to_close_short_day_3 = []
    for ticker in positions_to_short_day_2:
        if ticker not in positions_to_short_day_3:
            positions_to_close_short_day_3.append(ticker)
            
    # Printing the log:

    print ('='*60)
    print ('Positions to long on {}:\n{}'.format(timestamps[:check_days][0], sorted(positions_to_long_day_1)))
    print ('Number of stocks: {}'.format(len(positions_to_long_day_1)))
    print ('-'*60)
    print ('Positions to short on {}:\n{}'.format(timestamps[:check_days][0], sorted(positions_to_short_day_1)))
    print ('Number of stocks: {}'.format(len(positions_to_short_day_1)))

    print ('='*60)
    print ('Positions to long on {}:\n{}'.format(timestamps[:check_days][1], sorted(positions_to_long_day_2)))
    print ('Number of stocks: {}'.format(len(positions_to_long_day_2)))
    print ('-'*60)
    print ('Positions to short on {}:\n{}'.format(timestamps[:check_days][1], sorted(positions_to_short_day_2)))
    print ('Number of stocks: {}'.format(len(positions_to_short_day_2)))
    print ('-'*60)
    print ('Close long positions on {}:\n{}'.format(timestamps[:check_days][1], sorted(positions_to_close_long_day_2)))
    print ('Number of stocks: {}'.format(len(positions_to_close_long_day_2)))
    print ('-'*60)
    print ('Close short positions on {}:\n{}'.format(timestamps[:check_days][1], sorted(positions_to_close_short_day_2)))
    print ('Number of stocks: {}'.format(len(positions_to_close_short_day_2)))

    print ('='*60)
    print ('Positions to long on {}:\n{}'.format(timestamps[:check_days][2], sorted(positions_to_long_day_3)))
    print ('Number of stocks: {}'.format(len(positions_to_long_day_3)))
    print ('-'*60)
    print ('Positions to short on {}:\n{}'.format(timestamps[:check_days][2], sorted(positions_to_short_day_3)))
    print ('Number of stocks: {}'.format(len(positions_to_short_day_3)))
    print ('-'*60)
    print ('Close long positions on {}:\n{}'.format(timestamps[:check_days][2], sorted(positions_to_close_long_day_3)))
    print ('Number of stocks: {}'.format(len(positions_to_close_long_day_3)))
    print ('-'*60)
    print ('Close short positions on {}:\n{}'.format(timestamps[:check_days][2], sorted(positions_to_close_short_day_3)))
    print ('Number of stocks: {}'.format(len(positions_to_close_short_day_3)))
    print ('='*60)



def create_alphatrend_dst_score(df):
    df['ent_sent'] = df.groupby(['day'])['avg_entity_sentiment'].rank(ascending=True)
    df['eve_sent'] = df.groupby(['day'])['avg_event_sentiment'].rank(ascending=True)
    df['timeliness'] = df.groupby(['day'])['avg_entity_source_timeliness_score'].rank(ascending=True)
    df['traffic_sum'] = df.groupby(['day'])['avg_story_group_traffic_sum'].rank(ascending=True)
    df['alphatrend_dst_score'] = df['ent_sent']+df['eve_sent']+df['timeliness']+df['traffic_sum']
    return df


def generate_missing_backtests(result_file, config_file):
    results = pd.read_csv(result_file)
    sharpe_null_df = results[results['sharpe'].isnull()]
    rerun_result_df = sharpe_null_df[~sharpe_null_df.title.isnull()]
    rerun_algo_names = list(rerun_result_df.title)
    config = pd.read_csv(config_file)
    rerun_config_df = config[config['algo_name'].isin(rerun_algo_names)]
    return rerun_result_df

# Example:
# result_file = '/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/results/2_11_initialize_algorithms_table_results.csv'
# config_file = '/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/configs/2_11_avg_ent_sentiment_low_beta_rus_growth_daily_long_short_IS_OOS_config.csv'


# path = r'/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/configs/'              
# inputs = sorted(glob.glob(os.path.join(path, "*.csv")))  

# fieldnames = []
# for filename in inputs:
#     with open(filename, "r", newline="") as f_in:
#         reader = csv.reader(f_in)
#         headers = next(reader)
#         for h in headers:
#             if h not in fieldnames:
#                 fieldnames.append(h)
                
# # Then copy the data
# with open("out.csv", "w", newline="") as f_out:   # Comment 2 below
#     writer = csv.DictWriter(f_out, fieldnames=fieldnames)
#     for filename in inputs:
#         with open(filename, "r", newline="") as f_in:
#             reader = csv.DictReader(f_in)  # Uses the field names in this file
#             for line in reader:
#             # Comment 3 below
#             writer.writerow(line)



def get_price_df_from_liquid_russell_open_csv():
    price_df = pd.read_csv('Russell_top_mkt_cap_larger_than_1000mil_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df


def get_price_df_from_growth_stock_russell_open_csv():
    price_df = pd.read_csv('Russell_2000_renaissance_growth_stocks_daily_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df


def get_price_df_from_all_stock_russell_open_csv():
    price_df = pd.read_csv('2_8_Russell_2000_daily_all_data_all_tickers_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df


# ------------------- Make sure we don't have any forward-looking bias for daily strategy. meaning that we need to shift down all the dates using shift(1) -------------------
def get_daily_accern_for_Q(file_path):
    df = pd.read_csv(file_path)
    df['shifted_time'] = df.groupby('entity_ticker')['day'].shift(-1)
    df['time'] = pd.to_datetime(df['shifted_time'])
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['day'] = df['time'].dt.day
    df['symbol'] = df['entity_ticker']
    del df['entity_ticker']
    del df['shifted_time']
    return df.sort_values('time')
# ------------------- Make sure we don't have any forward-looking bias for daily strategy. meaning that we need to shift down all the dates using shift(1) -------------------


def get_price_df_from_liquid_russell_open_csv():
    price_df = pd.read_csv('Russell_top_mkt_cap_larger_than_1000mil_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df

def get_price_df_from_growth_russell_open_csv():
    price_df = pd.read_csv('Russell_growth_stocks_daily_open_price.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df
    
def get_price_df_from_sp500_open_csv():
    price_df = pd.read_csv('sp500_open_price_daily.csv')
    price_df.index = pd.to_datetime(price_df.Date)
    del price_df['Date']
    pricing_index = price_df.index.tz_localize('UTC')
    price_df.index = pricing_index
    return price_df


# def convert_daily_io_data_to_next_day(io_data):
#     df = pd.read_csv(io_data)
#     df['shifted_time'] = df.groupby('entity_ticker')['day'].shift(-1)
#     df['time'] = pd.to_datetime(df['shifted_time'])  
#     df['datetime'] = pd.to_datetime(df['time'])
#     df = df.set_index(['datetime', 'entity_ticker'])
#     factor_index = df.index.get_level_values(0).tz_localize("UTC")
#     df.set_index([factor_index, df.index.get_level_values(1)], inplace=True) 
#     return df.sort_values('time')


def get_accern_factor_daily_from_Q_for_open(df):
    df['datetime'] = pd.to_datetime(df['time'])  
    df = df.sort_values('datetime')
    factor_index = pd.DatetimeIndex(df['datetime']).tz_localize('UTC')
    df.set_index([factor_index, df['symbol']], inplace=True) 
    return df

def get_accern_factor_daily_from_io_for_open(file_path):
    df = pd.read_csv(file_path)
    df['shifted_time'] = df.groupby('entity_ticker')['day'].shift(-1)
    df['datetime'] = pd.to_datetime(df['shifted_time'])  
    df = df.sort_values('datetime')
    factor_index = pd.DatetimeIndex(df['datetime']).tz_localize('UTC')
    df.set_index([factor_index, df['entity_ticker']], inplace=True) 
    return df


def convert_monthly_io_data_to_next_month_first_day(io_data):
    df = pd.read_csv(io_data)

    df['next_month'] = df['month'].apply(lambda x: pd.bdate_range(start=x, periods=2, freq='BMS')[1])
    df['next_month'] = df['next_month'].apply(lambda x: str(x))
    df['next_month'] = df['next_month'].apply(lambda x: x.replace(' 00:00:00+00:00', ''))

    df.replace('2013-09-02', '2013-09-03', inplace=True)
    df.replace('2014-01-01', '2014-01-02', inplace=True)
    df.replace('2014-09-01', '2014-09-02', inplace=True)
    df.replace('2015-01-01', '2015-01-02', inplace=True)
    df.replace('2016-01-01', '2016-01-04', inplace=True)
    df.replace('2017-01-02', '2017-01-03', inplace=True)

    del df['month']
    return df



def get_Q_data_converting_monthly_io_data_to_next_month_first(file_path):
    
    df = pd.read_csv(file_path)

    df['next_month'] = df['month'].apply(lambda x: pd.bdate_range(start=x, periods=2, freq='BMS')[1])
    df['next_month'] = df['next_month'].apply(lambda x: str(x))
    df['next_month'] = df['next_month'].apply(lambda x: x.replace(' 00:00:00+00:00', ''))

    df.replace('2013-09-02', '2013-09-03', inplace=True)
    df.replace('2014-01-01', '2014-01-02', inplace=True)
    df.replace('2014-09-01', '2014-09-02', inplace=True)
    df.replace('2015-01-01', '2015-01-02', inplace=True)
    df.replace('2016-01-01', '2016-01-04', inplace=True)
    df.replace('2017-01-02', '2017-01-03', inplace=True)
    
    del df['month']
    
    df['time'] = pd.to_datetime(df['next_month'])
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['day'] = df['time'].dt.day
    df['symbol'] = df['entity_ticker']
    del df['next_month']
    
    return df


################ Official Version for Monthly Alphalens ################
def get_accern_factor_converting_monthly_io_data_to_next_month_first(io_data):
    
    df = pd.read_csv(io_data)

    df['next_month'] = df['month'].apply(lambda x: pd.bdate_range(start=x, periods=2, freq='BMS')[1])
    df['next_month'] = df['next_month'].apply(lambda x: str(x))
    df['next_month'] = df['next_month'].apply(lambda x: x.replace(' 00:00:00+00:00', ''))

    df.replace('2013-09-02', '2013-09-03', inplace=True)
    df.replace('2014-01-01', '2014-01-02', inplace=True)
    df.replace('2014-09-01', '2014-09-02', inplace=True)
    df.replace('2015-01-01', '2015-01-02', inplace=True)
    df.replace('2016-01-01', '2016-01-04', inplace=True)
    df.replace('2017-01-02', '2017-01-03', inplace=True)

    del df['month']

    df['datetime'] = pd.to_datetime(df['next_month'])
    df = df.set_index(['datetime', 'entity_ticker'])
    factor_index = df.index.get_level_values(0).tz_localize("UTC")
    df.set_index([factor_index, df.index.get_level_values(1)], inplace=True)
    return df




def last_trading_day_of_month_Quantopian():
  last_trading_day_of_month = \
  ['2013-08-30',
 '2013-09-30',
 '2013-10-31',
 '2013-11-29',
 '2013-12-31',
 '2014-01-31',
 '2014-02-28',
 '2014-03-31',
 '2014-04-30',
 '2014-05-30',
 '2014-06-30',
 '2014-07-31',
 '2014-08-29',
 '2014-09-30',
 '2014-10-31',
 '2014-11-28',
 '2014-12-31',
 '2015-01-30',
 '2015-02-27',
 '2015-03-31',
 '2015-04-30',
 '2015-05-29',
 '2015-06-30',
 '2015-07-31',
 '2015-08-31',
 '2015-09-30',
 '2015-10-30',
 '2015-11-30',
 '2015-12-31',
 '2016-01-29',
 '2016-02-29',
 '2016-03-31',
 '2016-04-29',
 '2016-05-31',
 '2016-06-30',
 '2016-07-29',
 '2016-08-31',
 '2016-09-30',
 '2016-10-31',
 '2016-11-30',
 '2016-12-30',
 '2017-01-31',
 '2017-02-28',
 '2017-03-31',
 '2017-04-28',
 '2017-05-31',
 '2017-06-30',
 '2017-07-31',
 '2017-08-31',
 '2017-09-29',
 '2017-10-31',
 '2017-11-30']
  return last_trading_day_of_month

  def first_trading_day_of_month_Quantopian():
    first_trading_day_of_month = \
    [
 '2013-09-03',
 '2013-10-01',
 '2013-11-01',
 '2013-12-02',
 '2014-01-02',
 '2014-02-03',
 '2014-03-03',
 '2014-04-01',
 '2014-05-01',
 '2014-06-02',
 '2014-07-01',
 '2014-08-01',
 '2014-09-02',
 '2014-10-01',
 '2014-11-03',
 '2014-12-01',
 '2015-01-02',
 '2015-02-02',
 '2015-03-02',
 '2015-04-01',
 '2015-05-01',
 '2015-06-01',
 '2015-07-01',
 '2015-08-03',
 '2015-09-01',
 '2015-10-01',
 '2015-11-02',
 '2015-12-01',
 '2016-01-04',
 '2016-02-01',
 '2016-03-01',
 '2016-04-01',
 '2016-05-02',
 '2016-06-01',
 '2016-07-01',
 '2016-08-01',
 '2016-09-01',
 '2016-10-03',
 '2016-11-01',
 '2016-12-01',
 '2017-01-03',
 '2017-02-01',
 '2017-03-01',
 '2017-04-03',
 '2017-05-01',
 '2017-06-01',
 '2017-07-03',
 '2017-08-01',
 '2017-09-01',
 '2017-10-02',
 '2017-11-01',
 '2017-12-01']
  return first_trading_day_of_month


# this list is by 12-20-2017
def sp500():
	sp500_list = ['MMM', 'ABT', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AMD', 'HII', 'AAP', 'AES', 'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ARNC', 'AJG', 'AIZ', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BHGE', 'BLL', 'BAC', 'BK', 'BAX', 'BBT', 'BDX', 'BRK.B', 'BBY', 'BIIB', 'BLK', 'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BHF', 'BMY', 'AVGO', 'BF.B', 'CHRW', 'CA', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'CBOE', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNC', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHTR', 'CHK', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO', 'GLW', 'COST', 'COTY', 'CCI', 'CSRA', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DOV', 'DWDP', 'DPS', 'DTE', 'DRE', 'DUK', 'DXC', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EVHC', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FISV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FL', 'F', 'FTV', 'FBHS', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT', 'GWW', 'HAL', 'HBI', 'HOG', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HP', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HPQ', 'HUM', 'HBAN', 'IDXX', 'INFO', 'ITW', 'ILMN', 'IR', 'INTC', 'ICE', 'IBM', 'INCY', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IQV', 'IRM', 'JEC', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KHC', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LEG', 'LEN', 'LUK', 'LLY', 'LNC', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'KORS', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'PCAR', 'PKG', 'PH', 'PDCO', 'PAYX', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RJF', 'RTN', 'O', 'RHT', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RCL', 'CRM', 'SBAC', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIG', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SPGI', 'SWK', 'SBUX', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYF', 'SNPS', 'SYY', 'TROW', 'TPR', 'TGT', 'TEL', 'FTI', 'TXN', 'TXT', 'TMO', 'TIF', 'TWX', 'TJX', 'TMK', 'TSS', 'TSCO', 'TDG', 'TRV', 'TRIP', 'FOXA', 'FOX', 'TSN', 'UDR', 'ULTA', 'USB', 'UA', 'UAA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'HCN', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS']
	return sp500_list

# Use the alphalens to filter out worse companies based on sum of entity_sentiment (rerun PA_0)
def sp500_non_predictive_sum_entity_sentiment_monthly():
    bad_list = [
    'AES', 'CF', 'CHD', 'EMN', 'FBHS', 'LYB', 'FMC', 'JEC', ]


# Use the alphalens to filter out worse companies based on sum of entity_sentiment (rerun PA_0)
def sp500_predictive_sum_entity_sentiment_monthly():
    good_list = [
    'ZBH', 'FCX', 'COL', 'ILMN', 'CBS', 'CCI', 'MGM', 'ORLY', 'TGT', 'VIAB', 'ED', 'HPQ',
    'SNPS', 'TRIP', 'TXN']
    return good_list


def v4_metrics():
	metrics = ['signal_id', 'story_id', 'story_group_id', 'new_story_group',
       'story_group_count', 'source_id', 'author_id', 'story_type',
       'story_source', 'templated_story_score', 'story_traffic',
       'story_group_traffic_sum', 'story_group_exposure', 'entity_sentiment',
       'event_sentiment', 'story_sentiment', 'story_group_sentiment_avg',
       'story_group_sentiment_stdev', 'entity_name', 'entity_ticker',
       'entity_exchange', 'entity_relevance', 'entity_country',
       'entity_indices', 'entity_industry', 'entity_region', 'entity_sector',
       'entity_competitors', 'entity_type', 'entity_composite_figi',
       'entity_exch_code', 'entity_figi', 'entity_market_sector',
       'entity_security_description', 'entity_security_type',
       'entity_share_class_figi', 'entity_unique_id',
       'entity_unique_id_fut_opt', 'entity_author_republish_score',
       'entity_author_timeliness_score', 'entity_source_republish_score',
       'entity_source_timeliness_score', 'event', 'event_group',
       'event_relevance', 'event_author_republish_score',
       'event_author_timeliness_score', 'event_source_republish_score',
       'event_source_timeliness_score', 'event_impact_pct_change_avg',
       'event_impact_pct_change_stdev', 'event_impact_pos', 'event_impact_neg',
       'event_impact_gt_mu_add_sigma', 'event_impact_lt_mu_sub_sigma',
       'event_impact_gt_mu_pos_add_sigma_pos',
       'event_impact_lt_mu_neg_sub_sigma_neg',
       'event_impact_gt_mu_pos_add_2sigma_pos',
       'event_impact_lt_mu_neg_sub_2sigma_neg', 'event_impact_gt_1pct_pos',
       'event_impact_lt_1pct_neg', 'overall_source_timeliness_score',
       'overall_source_republish_score', 'overall_author_republish_score',
       'overall_author_timeliness_score', 'harvested_at']
	return metrics


def v4_quant_metrics():
	quant_metrics = ['story_group_count','templated_story_score', 'story_traffic',
		'story_group_traffic_sum', 'story_group_exposure', 'entity_sentiment',
       'event_sentiment', 'story_sentiment', 'story_group_sentiment_avg',
       'story_group_sentiment_stdev', 'entity_relevance', 'entity_author_republish_score', 
       'entity_author_timeliness_score', 'entity_source_republish_score',
       'entity_source_timeliness_score','event_relevance', 'event_author_republish_score',
       'event_author_timeliness_score', 'event_source_republish_score',
       'event_source_timeliness_score', 'event_impact_pct_change_avg',
       'event_impact_pct_change_stdev', 'event_impact_pos', 'event_impact_neg',
       'event_impact_gt_mu_add_sigma', 'event_impact_lt_mu_sub_sigma',
       'event_impact_gt_mu_pos_add_sigma_pos',
       'event_impact_lt_mu_neg_sub_sigma_neg',
       'event_impact_gt_mu_pos_add_2sigma_pos',
       'event_impact_lt_mu_neg_sub_2sigma_neg', 'event_impact_gt_1pct_pos',
       'event_impact_lt_1pct_neg', 'overall_source_timeliness_score',
       'overall_source_republish_score', 'overall_author_republish_score',
       'overall_author_timeliness_score']
	return quant_metrics




    
    # 'story_group_count',
    # 'templated_story_score', 
    # 'story_traffic',

#-------------------------------------------
    # 'story_group_traffic_sum', 
#-------------------------------------------

    # 'story_group_exposure', 
    # 'entity_sentiment',
       # 'event_sentiment', 
       # 'story_sentiment', 
       # 'story_group_sentiment_avg',
       # 'story_group_sentiment_stdev', 
       # 'entity_relevance', 
       # 'entity_author_republish_score', 
       # 'entity_author_timeliness_score', 
       # 'entity_source_republish_score',


#-------------------------------------------
       # 'entity_source_timeliness_score',
#-------------------------------------------


       # 'event_relevance', 
       # 'event_author_republish_score',
       # 'event_author_timeliness_score', 
       # 'event_source_republish_score',
       # 'event_source_timeliness_score', 
       # 'event_impact_pct_change_avg',
       # 'event_impact_pct_change_stdev', 
       # 'event_impact_pos', 
       # 'event_impact_neg',
       # 'event_impact_gt_mu_add_sigma', 
       # 'event_impact_lt_mu_sub_sigma',
       # 'event_impact_gt_mu_pos_add_sigma_pos',
       # 'event_impact_lt_mu_neg_sub_sigma_neg',
       # 'event_impact_gt_mu_pos_add_2sigma_pos',
       # 'event_impact_lt_mu_neg_sub_2sigma_neg', 
       # 'event_impact_gt_1pct_pos',
       # 'event_impact_lt_1pct_neg', 
       # 'overall_source_timeliness_score',
       # 'overall_source_republish_score', 
       # 'overall_author_republish_score',
       # 'overall_author_timeliness_score']





def v4_qual_metrics():
	qual_metrics = ['entity_composite_figi','entity_sector',
					 'entity_name','entity_exch_code',
					 'story_type','event','signal_id',
					 'entity_indices','entity_security_type',
					 'entity_security_description','entity_unique_id_fut_opt',
					 'entity_exchange','entity_type','entity_market_sector',
					 'entity_country','entity_figi','story_source',
					 'harvested_at','entity_industry',
					 'event_group','entity_unique_id',
					 'entity_region','new_story_group',
					 'entity_ticker',
					 'source_id',
					 'entity_competitors',
					 'entity_share_class_figi',
					 'story_id',
					 'author_id',
					 'story_group_id']

	return qual_metrics 

def get_yahoo_multiple_tickers(ticker_list, start_date, end_date):
	df = yf.download(ticker_list, start = start_date, end = end_date)
	df_unstacked = df.to_frame().unstack()
	return df_unstacked


def get_yahoo_single_ticker(ticker, start_date, end_date):
	df = yf.download(ticker, start = start_date, end = end_date)
	return df


def series_ema(series, window_span):
    expmovingavg = series.ewm(span = window_span).mean()
    return expmovingavg


def combine_csv_files(path_drct, output_file_name):
	path = path_drct
	interesting_files = glob.glob(path) 
	header_saved = False
	with open(output_file_name,'wb') as fout:
	    for filename in interesting_files:
	        with open(filename) as fin:
	            header = next(fin)
	            if not header_saved:
	                fout.write(header.encode('utf-8'))
	                header_saved = True
	            for line in fin:
	                fout.write(line.encode('utf-8'))


# Note that metric_columns is a list containig the metric strings.
def trim_columns(read_path_name, metric_columns, export_path_name):
    df = pd.read_csv(read_path_name)
    default_columns = ['time', 'month', 'year', 'day', 'hour', 'symbol']
    metric_columns = metric_columns
    columns_to_keep =  set(default_columns) | set(metric_columns)
    columns_to_drop = list(set(df.columns) - set(columns_to_keep))
    df_trimmed = df.drop(columns_to_drop, axis = 1)
    df_trimmed.to_csv(export_path_name)
    return df_trimmed


def sector_names():

    sector_category = ['Real Estate', 'Industrials', 'Materials', 'Information Technology', 'Financials', 'Utilities', 'Telecommunication Services', 'Health Care', 'Energy', 'Consumer Staples', 'Consumer Discretionary']

    return sector_category


# def ticker_sector():
#   ticker_sector = {
#     "ACN" : 0, "ATVI" : 0, "ADBE" : 0, "AMD" : 0, "AKAM" : 0, "ADS" : 0, "GOOGL" : 0, "GOOG" : 0, 
#     "APH" : 0, "ADI" : 0, "ANSS" : 0, "AAPL" : 0, "AMAT" : 0, "ADSK" : 0, "ADP" : 0, "AVGO" : 0,
#     "AMG" : 1, "AFL" : 1, "ALL" : 1, "AXP" : 1, "AIG" : 1, "AMP" : 1, "AON" : 1, "AJG" : 1, "AIZ" : 1, "BAC" : 1,
#     "BK" : 1, "BBT" : 1, "BRK.B" : 1, "BLK" : 1, "HRB" : 1, "BHF" : 1, "COF" : 1, "CBOE" : 1, "SCHW" : 1, "CB" : 1,
#     "ABT" : 2, "ABBV" : 2, "AET" : 2, "A" : 2, "ALXN" : 2, "ALGN" : 2, "AGN" : 2, "ABC" : 2, "AMGN" : 2, "ANTM" : 2,
#     "BCR" : 2, "BAX" : 2, "BDX" : 2, "BIIB" : 2, "BSX" : 2, "BMY" : 2, "CAH" : 2, "CELG" : 2, "CNC" : 2, "CERN" : 2,
#     "MMM" : 3, "AYI" : 3, "ALK" : 3, "ALLE" : 3, "AAL" : 3, "AME" : 3, "AOS" : 3, "ARNC" : 3, "BA" : 3, "CHRW" : 3,
#     "CAT" : 3, "CTAS" : 3, "CSX" : 3, "CMI" : 3, "DE" : 3, "DAL" : 3, "DOV" : 3, "ETN" : 3, "EMR" : 3, "EFX" : 3,
#     "AES" : 4, "LNT" : 4, "AEE" : 4, "AEP" : 4, "AWK" : 4, "CNP" : 4, "CMS" : 4, "ED" : 4, "D" : 4, "DTE" : 4,
#     "DUK" : 4, "EIX" : 4, "ETR" : 4, "ES" : 4, "EXC" : 4, "FE" : 4, "NEE" : 4, "NI" : 4, "NRG" : 4, "PCG" : 4,
#     "ARE" : 5, "AMT" : 5, "AIV" : 5, "AVB" : 5, "BXP" : 5, "CBG" : 5, "CCI" : 5, "DLR" : 5, "DRE" : 5,
#     "EQIX" : 5, "EQR" : 5, "ESS" : 5, "EXR" : 5, "FRT" : 5, "GGP" : 5, "HCP" : 5, "HST" : 5, "IRM" : 5, "KIM" : 5,
#     "APD" : 6, "ALB" : 6, "AVY" : 6, "BLL" : 6, "CF" : 6, "DWDP" : 6, "EMN" : 6, "ECL" : 6, "FMC" : 6, "FCX" : 6,
#     "IP" : 6, "IFF" : 6, "LYB" : 6, "MLM" : 6, "MON" : 6, "MOS" : 6, "NEM" : 6, "NUE" : 6, "PKG" : 6, "PPG" : 6,
#     "T" : 7, "CTL" : 7, "VZ" : 7, 
#     "MO" : 8, "ADM" : 8, "BF.B" : 8, "CPB" : 8, "CHD" : 8, "CLX" : 8, "KO" : 8, "CL" : 8, "CAG" : 8,
#     "STZ" : 8, "COST" : 8, "COTY" : 8, "CVS" : 8, "DPS" : 8, "EL" : 8, "GIS" : 8, "HSY" : 8, "HRL" : 8,
#     "AAP" : 9, "AMZN" : 9, "APTV" : 9, "AZO" : 9, "BBY" : 9, "BWA" : 9, "KMX" : 9, "CCL" : 9, 
#     "APC" : 10, "ANDV" : 10, "APA" : 10, "BHGE" : 10, "COG" : 10, "CHK" : 10, "CVX" : 10, "XEC" : 10, "CXO" : 10,
#     "COP" : 10, "DVN" : 10, "EOG" : 10, "EQT" : 10, "XOM" : 10, "HAL" : 10, "HP" : 10, "HES" : 10, "KMI" : 10}
#   return ticker_sector


def sector_category_dict():
    c = pd.read_csv('/Users/workspace/Accern/Project1_backtesting_shared_on_github/Backtesting/sp500_csv_list.csv')
    sector_list = list(set(open_sector_df['GICS Sector']))
    iter_range = range(11)
    single_sector_dict = {}
    for i in iter_range:    
      all_info = open_sector_df[open_sector_df['GICS Sector'] == sector_list[i]]
      single_sector_dict[str(sector_list[i])] = all_info['Ticker symbol']
    return single_sector_dict


# Push the first day of that month to the last business trading day of that month and meanwhile convert to the Q data format. 
def convert_monthly_format_for_Q(file_path):
    open_accern_reninv = pd.read_csv(file_path)
    accern_factor_ = open_accern_reninv
    accern_factor_['time'] = pd.to_datetime(accern_factor_['month']) 

    offset = BMonthEnd()
    for i, date in enumerate(accern_factor_['time']):
        d = accern_factor_['time'][i].date()
        d = offset.rollforward(d)
        accern_factor_['time'][i] = d
    del accern_factor_['month']

    accern_factor_['time'] = pd.to_datetime(accern_factor_['time'])
    accern_factor_['month'] = accern_factor_['time'].dt.month
    accern_factor_['year'] = accern_factor_['time'].dt.year
    accern_factor_['day'] = accern_factor_['time'].dt.day
    accern_factor_['symbol'] = accern_factor_['entity_ticker']

    return accern_factor_

# def monthly_io_shift_


"""
This is the function that converts the raw Accern data's format to alphalens's required factor dataframe format, 
specifically for daily data. 'factor_csv_file' looks like: e.g. '1_16_10_ETFs_daily.csv'(raw data file downloaded from Accern's io platform).
"""

def process_factor_daily_data(factor_csv_file):
    accern_factor = pd.read_csv(factor_csv_file)
    accern_factor['datetime'] = pd.to_datetime(accern_factor['day'])
    accern_factor = accern_factor.set_index(['datetime', 'entity_ticker'])
    factor_index = accern_factor.index.get_level_values(0).tz_localize("UTC")
    accern_factor.set_index([factor_index, accern_factor.index.get_level_values(1)], inplace=True)
    del accern_factor['day']
    return accern_factor


# Look at the percentage change for the N forward periods in a dataframe for a column
def forward_looking_pct_periodwise(df, df_col, periods):
    df['{}_periods_away_pct_chg'.format(periods)] = df[df_col].pct_change(periods).shift(-periods)
    df['{}_periods_away_value'.format(periods)] = df[df_col].shift(-periods)
    return df


# def process_Q_formatted_data_to_accern_factor_df():





"""
This is the function that downloads yahoo finance adjusted close data based on Accern factor file's tickers and also converts 
to Alphalens's price dataframe format. 'factor_csv_file' looks like: e.g. '1_16_10_ETFs_daily.csv' (raw data file downloaded 
from Accern's io platform).

"""


def process_daily_adj_close_price_data(factor_csv_file):
    factor_df = pd.read_csv(factor_csv_file)
    factor_ticker_list = list(factor_df.entity_ticker.unique())
    price_data = get_yahoo_data(factor_ticker_list, '2013-08-01', '2017-11-30')
    close_price_reninv = price_data['Adj Close']
    price_index = close_price_reninv.index.tz_localize("UTC")
    close_price_reninv.set_index(price_index, inplace=True)
    
    return close_price_reninv



def process_monthly_adjclose_price_data(factor_csv_file):
    factor_df = pd.read_csv(factor_csv_file)
    factor_ticker_list = list(factor_df.entity_ticker.unique())
    price_data = get_yahoo_data(factor_ticker_list, '2013-08-01', '2017-11-30')
    close_price_reninv = price_data['Adj Close']
    
    close_price_reninv['datetime'] = pd.to_datetime(close_price_reninv.index)
    close_price_reninv['date'] = close_price_reninv['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))

    close_price_reninv['year'] = close_price_reninv.datetime.dt.year
    close_price_reninv['month'] = close_price_reninv.datetime.dt.month
    close_price_reninv['day'] = close_price_reninv.datetime.dt.day
    
    last_day_price_date_list = []
    for i, month in enumerate(close_price_reninv['month']):
        try: 
            if close_price_reninv['month'][i] != close_price_reninv['month'][i-1]:
                last_day_price_date_list.append(close_price_reninv['date'][i-1])
            else:
                continue
        except:
            continue
            
    close_price_reninv.index = close_price_reninv.datetime
    rows_str_to_drop = []
    for i, date in enumerate(close_price_reninv['date']):
        if close_price_reninv['date'][i] not in last_day_price_date_list:
            rows_str_to_drop.append(close_price_reninv['date'][i])
        else:
            continue

    close_price_reninv['to_be_dropped'] = close_price_reninv['date'].apply(lambda x: x in rows_str_to_drop)
    close_price_reninv.drop(close_price_reninv[close_price_reninv.to_be_dropped == True].index, inplace=True)
    
    pricing_index = close_price_reninv.index.tz_localize('UTC')
    close_price_reninv.index = pricing_index
    close_price_reninv = close_price_reninv.drop(['datetime', 'date', 'year', 'month', 'day'], axis=1)
    del close_price_reninv['to_be_dropped']
    return close_price_reninv



# We already have the monthly aggragted data from io platform, and convert to the last trading day of the month for the timestamp
def process_monthly_end_factor_data(factor_csv_file):
    open_accern_reninv = pd.read_csv(factor_csv_file)
    accern_factor_ = open_accern_reninv
    accern_factor_['time'] = pd.to_datetime(accern_factor_['month']) 
    
    offset = BMonthEnd()
    for i, date in enumerate(accern_factor_['time']):
        d = accern_factor_['time'][i].date()
        d = offset.rollforward(d)
        accern_factor_['time'][i] = d
    del accern_factor_['month']
    
    open_accern_reninv = accern_factor_  
    open_accern_reninv['datetime'] = pd.to_datetime(open_accern_reninv.time)
    accern_factor = open_accern_reninv
    accern_factor['datetime'] = pd.to_datetime(accern_factor['time'])
    accern_factor = accern_factor.set_index(['datetime', 'entity_ticker'])
    factor_index = accern_factor.index.get_level_values(0).tz_localize("UTC")
    accern_factor.set_index([factor_index, accern_factor.index.get_level_values(1)], inplace=True)
    del accern_factor['time']
    
    return accern_factor

"""
This function draws the period-wise mean return quantile factor charts of only one ticker, for a specific 
"""
def single_ticker_quantile_return_analysis(factor_csv_file, target_metric_str):
    single_df = pd.read_csv(factor_csv_file)
    single_df.index = pd.to_datetime(single_df['day'])
    single_df_index = single_df.index.tz_localize("UTC")
    single_df.set_index(single_df_index, inplace=True)
    del single_df['day']
    
    single_ticker = list(single_df.entity_ticker.unique())
    price_df = yf.download(single_ticker, start = '2013-08-01', end = '2017-11-30')
    price_df_adjclose =pd.DataFrame(price_df['Adj Close'])
    price_df_adjclose.columns = single_ticker
    price_df_index = price_df_adjclose.index.tz_localize("UTC")
    price_df_adjclose.set_index(price_df_index, inplace=True)
    
    single_df['merged_ts'] = single_df.index
    price_df_adjclose['merged_ts'] = price_df_adjclose.index
    merged_daily = price_df_adjclose.merge(single_df, on = 'merged_ts')
    target_metric_str = target_metric_str
    merged_daily['quantiles'] = pd.qcut(merged_daily[target_metric_str], 5, labels=np.arange(1, 6, 1))
    
    merged_daily['1_period_return_forward'] = merged_daily[single_ticker].pct_change(periods = 1).shift(-1)
    merged_daily['5_period_return_forward'] = merged_daily[single_ticker].pct_change(periods = 5).shift(-5)
    merged_daily['10_period_return_forward'] = merged_daily[single_ticker].pct_change(periods = 10).shift(-10)
    
    one_period_mean_return_forward = merged_daily.groupby(['quantiles'])['1_period_return_forward'].mean()
    five_period_mean_return_forward = merged_daily.groupby(['quantiles'])['5_period_return_forward'].mean()
    ten_period_mean_return_forward = merged_daily.groupby(['quantiles'])['10_period_return_forward'].mean()
    
    mean_return_quantile_analysis_df = pd.DataFrame()
    mean_return_quantile_analysis_df['1_period_mean_return_forward'] = one_period_mean_return_forward
    mean_return_quantile_analysis_df['5_period_mean_return_forward'] = five_period_mean_return_forward
    mean_return_quantile_analysis_df['10_period_mean_return_forward'] = ten_period_mean_return_forward

    mean_return_quantile_analysis_df.plot(kind='bar', figsize=(18, 6))

def sector_names_mapping():
  sector_names = {
    0 : "Information Technology",
    1 : "Financials",
    2 : "Health Care",
    3 : "Industrials",
    4 : "Utilities", 
    5 : "Real Estate", 
    6 : "Materials", 
    7 : "Telecommunication Services", 
    8 : "Consumer Staples", 
    9 : "Consumer Discretionary", 
    10 : "Energy" 
    }
  return sector_names

def sector_names_mapping_rus():
    sector_names = {
    0 : "Information Technology",
    1 : "Financials",
    2 : "Health Care",
    3 : "Industrials",
    4 : "Utilities", 
    5 : "Real Estate", 
    6 : "Materials", 
    7 : "Telecommunication Services", 
    8 : "Consumer Staples", 
    9 : "Consumer Discretionary", 
    10 : "Energy",
    11 : "Cash and/or Derivatives"
    }
    return sector_names
# def sp500_ticker_sector_mapping():
#     ticker_sector = {'CTL': 7, 'COTY': 8, 'DISCK': 9, 'AIV': 5, 'FRT': 5, 'CF': 6, 'CA': 0, 'MHK': 9, 'FLIR': 0, 'GS': 1, 'HD': 9, 'AMD': 0, 'MAA': 5, 'KR': 8, 'RTN': 3, 'MKC/V': 8, 'BLL': 6, 'UAA': 9, 'CCI': 5, 'IRM': 5, 'NVDA': 0, 'SYMC': 0, 'DUK': 4, 'IBM': 0, 'LNC': 1, 'LMT/WD': 3, 'EBAY': 0, 'EXC': 4, 'ARNC': 3, 'MON': 6, 'TAP/A': 8, 'PXD': 10, 'FL': 9, 'USB': 1, 'TIF': 9, 'GGP': 5, 'KSS': 9, 'CTXS': 0, 'RMDCD': 2, 'CAH': 2, 'MCO': 1, 'PH': 3, 'IP': 6, 'HRS': 0, 'SYY': 8, 'BIIB': 2, 'TWX': 9, 'MS': 1, 'UDR': 5, 'WMT': 8, 'OMC': 9, 'NFX': 10, 'NRG': 4, 'KHC': 8, 'SPGI': 1, 'AXP': 1, 'TRV': 1, 'ZION': 1, 'PKG': 6, 'HON-W': 3, 'WBA': 8, 'EVHC': 2, 'NCLH': 9, 'ITW': 3, 'FB': 0, 'D': 4, 'EA': 0, 'CAG-W': 8, 'CTSH': 0, 'COO': 2, 'NSC': 3, 'JNJ': 2, 'XRX-W': 0, 'DGX': 2, 'DHR-W': 2, 'BA/': 3, 'SPG': 5, 'IFF': 6, 'PM': 8, 'GE': 3, 'ABC': 2, 'ETR': 4, 'ABBV': 2, 'JEC': 3, 'ROST': 9, 'CINF': 1, 'NWS': 9, 'INFO': 3, 'MTB': 1, 'SBAC': 5, 'ADM': 8, 'STZ/B': 8, 'SWKS': 0, 'AAPL': 0, 'GT': 9, 'HRL': 8, 'NUE': 6, 'INCY': 2, 'KMX': 9, 'CB': 1, 'MCD': 9, 'GOOG': 0, 'PPG': 6, 'F': 9, 'TSN': 8, 'MRO': 10, 'V': 0, 'DFS': 1, 'UHS': 2, 'SCG': 4, 'O': 5, 'EOG': 10, 'BBT': 1, 'AVGO': 0, 'XEC': 10, 'AMAT': 0, 'LLL': 3, 'MDT': 2, 'RHI': 3, 'ES': 4, 'ROK': 3, 'TMK': 1, 'TGT': 9, 'EXR': 5, 'EL': 8, 'GLW': 0, 'REG': 5, 'GIS': 8, 'ROP': 3, 'BMY': 2, 'ADBE': 0, 'GPS': 9, 'CI': 2, 'HUM': 2, 'ECL': 6, 'APD-W': 6, 'BRK.B': 1, 'MCK': 2, 'RJF': 1, 'PX': 6, 'KO': 8, 'CBS': 9, 'VAR-W': 2, 'TROW': 1, 'WYNN': 9, 'KEY': 1, 'YUM-W': 9, 'KMB': 8, 'RHT': 0, 'CNC': 2, 'DISCA': 9, 'APC': 10, 'SLG': 5, 'HSIC': 2, 'ACN': 0, 'MU': 0, 'ANTM': 2, 'ATVI': 0, 'IDXX': 2, 'DOV': 3, 'T': 7, 'XEL': 4, 'CDNS': 0, 'K': 8, 'NOC': 3, 'FOXA': 9, 'PPL': 4, 'CMG': 9, 'PNW': 4, 'PSX': 10, 'ABT': 2, 'SIG': 9, 'QCOM': 0, 'CHD': 8, 'AMG': 1, 'ORLY': 9, 'WY': 5, 'PLD': 5, 'PRGO': 2, 'GILD': 2, 'NTRS': 1, 'HAS': 9, 'XOM': 10, 'CMI': 3, 'GPC': 9, 'ALL': 1, 'JWN': 9, 'FTI': 10, 'SNPS': 0, 'LH': 2, 'FISV': 0, 'MAT': 9, 'DVN': 10, 'AEE': 4, 'STT': 1, 'GOOGL': 0, 'AAP': 9, 'NLSN': 3, 'PHM': 9, 'ALB': 6, 'UNM': 1, 'BDX': 2, 'JCI-W': 3, 'CMA': 1, 'PRU': 1, 'PCLN': 9, 'SLB': 10, 'PVH': 9, 'ADSK': 0, 'DVA': 2, 'ZBH': 2, 'AAL': 3, 'MLM': 6, 'MAC': 5, 'XRAY': 2, 'DHI': 9, 'IT': 0, 'IR': 3, 'MYL': 2, 'VTR': 5, 'CHRW': 3, 'CXO': 10, 'WLTW': 1, 'MAR': 9, 'AMGN': 2, 'LUK-W': 1, 'XLNX': 0, 'NFLX': 0, 'UAL': 3, 'PKI': 2, 'ALXN': 2, 'HCP-W': 5, 'PAYX': 0, 'COL': 3, 'LEG': 9, 'MDLZ': 8, 'LYB': 6, 'INTC': 0, 'UNP': 3, 'STI': 1, 'PEP': 8, 'ILMN': 2, 'DE': 3, 'WM': 3, 'JNPR': 0, 'MOS': 6, 'TDG': 3, 'AZO': 9, 'DIS': 9, 'DTE': 4, 'VLO': 10, 'HAL': 10, 'TEL': 0, 'EIX': 4, 'CMS': 4, 'WU': 0, 'CTAS': 3, 'ISRG': 2, 'RSG': 3, 'PBCT': 1, 'LKQ': 9, 'ORCL': 0, 'TMO': 2, 'SWK': 9, 'DXC': 0, 'BK': 1, 'SBUX': 9, 'XYL': 3, 'WAT': 2, 'TSCO': 9, 'KMI': 10, 'CSCO': 0, 'BHF': 1, 'ALGN': 2, 'DRE': 5, 'ETN': 3, 'CFG': 1, 'XL': 1, 'SHW': 6, 'SYF': 1, 'WDC': 0, 'WFC': 1, 'SO': 4, 'HP': 10, 'NAVI': 1, 'TPR': 9, 'CHK': 10, 'AJG': 1, 'COG': 10, 'ARE': 5, 'AWK': 4, 'DG': 9, 'LRCX': 0, 'HCN': 5, 'APH': 0, 'DPS': 8, 'EMN': 6, 'FFIV': 0, 'INTU': 0, 'LUV': 3, 'HOG': 9, 'AME': 3, 'FLR': 3, 'CSRA': 0, 'HES': 10, 'SJM': 8, 'ALLE': 3, 'UA': 9, 'PFG': 1, 'NI': 4, 'OKE': 10, 'NWL': 9, 'WHR': 9, 'NWSA': 9, 'AMP': 1, 'ANSS': 0, 'CME': 1, 'QRVO': 0, 'COF': 1, 'ULTA': 9, 'EXPD': 3, 'ALK': 3, 'NTAP': 0, 'C': 1, 'NEM': 6, 'MMC': 1, 'APA': 10, 'COP': 10, 'NBL': 10, 'FITB': 1, 'AVB': 5, 'BBY': 9, 'FCX': 6, 'VNO-W': 5, 'CSX': 3, 'FDX': 3, 'PSA': 5, 'PGR': 1, 'PG/WD': 8, 'PNR': 3, 'HBI': 9, 'CVX': 10, 'MCHP': 0, 'CAT': 3, 'LNT': 4, 'RF': 1, 'L': 1, 'FOX': 9, 'PDCO': 2, 'AET': 2, 'BCR': 2, 'CELG': 2, 'MET': 1, 'DWDP': 6, 'BSX': 2, 'EMR': 3, 'LEN/B': 9, 'GPN': 0, 'DLTR': 9, 'RCL': 9, 'DRI': 9, 'TSS': 0, 'FLS': 3, 'GRMN': 9, 'BWA': 9, 'VMC': 6, 'TRIP': 9, 'ESRX': 2, 'MMM': 3, 'OXY': 10, 'BLK': 1, 'BXP': 5, 'SYK': 2, 'DAL': 3, 'BAC': 1, 'CNP': 4, 'FIS': 0, 'PFE': 2, 'ADP': 0, 'AOS': 3, 'DLR': 5, 'EQR': 5, 'LLY': 2, 'MRK': 2, 'AYI': 3, 'WYN': 9, 'CERN': 2, 'LB': 9, 'KIM': 5, 'PCAR': 3, 'KSU': 3, 'AON': 1, 'GM': 9, 'TJX': 9, 'GD': 3, 'MPC/C': 10, 'UTX': 3, 'CHTR': 9, 'APTV': 9, 'VZ': 7, 'HIG': 1, 'VRSK': 3, 'AVY': 6, 'LOW': 9, 'SNI': 9, 'HPE-W': 0, 'CVS': 8, 'EFX': 3, 'MGM': 9, 'HOLX': 2, 'CL': 8, 'SCHW': 1, 'KORS': 9, 'TXN': 0, 'BF.B': 8, 'HRB': 1, 'ADI': 0, 'HST': 5, 'PEG': 4, 'FBHS': 3, 'ED': 4, 'VRSN': 0, 'MAS': 3, 'EW': 2, 'M': 9, 'ADS': 0, 'JBHT': 3, 'CBOE': 1, 'NEE': 4, 'UNH': 2, 'RRC': 10, 'IVZ': 1, 'BEN': 1, 'HCA': 2, 'VIAB': 9, 'MA': 0, 'IPG': 9, 'VFC': 9, 'AKAM': 0, 'HSY': 8, 'CMCSA': 9, 'CBG': 5, 'ICE': 1, 'PWR': 3, 'DISH': 9, 'AFL': 1, 'COST': 8, 'WEC': 4, 'NKE': 9, 'AMZN': 9, 'PNC': 1, 'NDAQ': 1, 'REGN': 2, 'URI': 3, 'SNA': 9, 'CCL': 9, 'MTD': 2, 'SRCL': 3, 'AMT': 5, 'HLT': 9, 'FE': 4, 'A': 2, 'SEE': 6, 'CPB': 8, 'MSFT': 0, 'KLAC': 0, 'ANDV': 10, 'PCG': 4, 'AES': 4, 'EXPE': 9, 'WMB': 10, 'FTV': 3, 'PYPL': 0, 'ETFC': 1, 'WRK-W': 6, 'ZTS': 2, 'HPQ': 0, 'MO': 8, 'CRM': 0, 'FMC': 6, 'ESS': 5, 'BHGE': 10, 'STX': 0, 'RL': 9, 'NOV': 10, 'AIZ': 1, 'FAST': 3, 'AIG': 1, 'TXT': 3, 'VRTX': 2, 'AGN': 2, 'BAX': 2, 'EQT': 10, 'CLX': 8, 'IQV': 2, 'EQIX': 5, 'RE': 1, 'JPM': 1, 'SRE': 4, 'GWW': 3, 'HBAN': 1, 'AEP': 4, 'UPS': 3, 'MSI': 0, 'MNST': 8}
#     return ticker_sector

def rus_growth_ticker_sector_mapping():
    ticker_sector = {'ITRI': 0, 'TTEC': 0, 'KTWO': 2, 'CHFN': 1, 'XONE': 3, 'GLDD': 3, 'NGHC': 1, 'BLKB': 0, 'HRTG': 1, 'EVRI': 0, 'SRPT': 2, 'MGEE': 4, 'GTN': 9, 'HOMB': 1, 'XNCR': 2, 'NAV': 3, 'ILG': 9, 'LNTH': 2, 'FBK': 1, 'CRDB': 1, 'CORE': 9, 'ATRI': 2, 'UVE': 1, 'BSTC': 2, 'BMI': 0, 'NKTR': 2, 'ONCE': 2, 'MLHR': 3, 'CAKE': 9, 'KPTI': 2, 'MYE': 6, 'NLNK': 2, 'BIG': 9, 'CROX': 9, 'SUPN': 2, 'PLNT': 9, 'OPTN': 2, 'VGR': 8, 'HCHC': 3, 'BJRI': 9, 'FRGI': 9, 'XOXO': 0, 'CHUBK': 0, 'EXTR': 0, 'TPRE': 1, 'EMKR': 0, 'CWT': 4, 'SRT': 0, 'COLB': 1, 'LLNW': 0, 'WEB': 0, 'QTM': 0, 'COLL': 2, 'FLDM': 2, 'NX': 3, 'GST': 10, 'NUVA': 2, 'CTLT': 2, 'UNB': 1, 'PSMT': 8, 'UPL': 10, 'MCB': 1, 'JBSS': 8, 'COTV': 2, 'PLSE': 2, 'HIL': 3, 'WVE': 2, 'RGR': 9, 'OBLN': 2, 'YEXT': 0, 'PXLW': 0, 'NSTG': 2, 'SOI': 10, 'BOLD': 2, 'LTC': 5, 'TR': 8, 'TWOU': 0, 'HLNE': 1, 'GWRS': 4, 'GRUB': 0, 'BCO': 3, 'DIN': 9, 'MHO': 9, 'NSIT': 0, 'NTRI': 9, 'DDD': 0, 'SXT': 6, 'IMMR': 0, 'MNR': 5, 'YORW': 4, 'HSKA': 2, 'SND': 10, 'HIIQ': 1, 'PVBC': 1, 'NFBK': 1, 'ARCB': 3, 'OCLR': 0, 'SAMG': 1, 'COWN': 1, 'CASS': 0, 'HIVE': 0, 'MOGA': 3, 'CWH': 9, 'HEES': 3, 'BCC': 6, 'NVEE': 3, 'HALO': 2, 'ADES': 6, 'SXI': 3, 'RRD': 3, 'WLB': 10, 'EBS': 2, 'WLDN': 3, 'BLD': 9, 'TREE': 1, 'VIAV': 0, 'FLOW': 3, 'SUM': 6, 'AIMT': 2, 'RDFN': 5, 'TSE': 6, 'AMBA': 0, 'WTW': 9, 'DLTH': 9, 'REXR': 5, 'INOV': 2, 'ABAX': 2, 'CSII': 2, 'LMAT': 2, 'STRL': 3, 'AT': 4, 'KALA': 2, 'AGX': 3, 'AGEN': 2, 'GPX': 3, 'GBL': 1, 'AXGN': 2, 'LITE': 0, 'TPC': 3, 'HBMD': 1, 'XBIT': 2, 'SELB': 2, 'CW': 3, 'GHDX': 2, 'LHCG': 2, 'PEGI': 4, 'PCH': 5, 'SLP': 2, 'EGHT': 0, 'MPX': 9, 'CVRS': 2, 'PTGX': 2, 'SNHY': 3, 'KIN': 2, 'NRCIA': 2, 'APAM': 1, 'USNA': 8, 'LTS': 1, 'INO': 2, 'LSCC': 0, 'HSTM': 2, 'VRNT': 0, 'GEO': 5, 'SSTK': 0, 'BRSS': 3, 'CYBE': 0, 'CBM': 2, 'OLBK': 1, 'OLLI': 9, 'TRUE': 0, 'LNDC': 8, 'DIOD': 0, 'MGI': 0, 'IRTC': 2, 'LIND': 9, 'OFED': 1, 'XPER': 0, 'PAYC': 0, 'OKTA': 0, 'PATK': 3, 'PAHC': 2, 'CTMX': 2, 'DORM': 9, 'CVA': 3, 'CLPR': 5, 'EVI': 3, 'TWNK': 8, 'SSD': 3, 'TRHC': 2, 'NBHC': 1, 'RNET': 10, 'KERX': 2, 'HUBS': 0, 'EYE': 9, 'KOP': 6, 'ASMB': 2, 'GMS': 3, 'AQ': 0, 'FOE': 6, 'OCX': 2, 'SEND': 0, 'ROIC': 5, 'CTO': 5, 'SHLO': 9, 'LL': 9, 'IPCC': 1, 'AAOI': 0, 'CHEF': 8, 'WK': 0, 'LTXB': 1, 'MMSI': 2, 'VHC': 0, 'MATX': 3, 'RLI': 1, 'VREX': 2, 'GEN': 2, 'CENTA': 8, 'WWW': 9, 'UBNT': 0, 'RIGL': 2, 'APPF': 0, 'EROS': 9, 'BPMC': 2, 'HOFT': 9, 'REV': 8, 'STMP': 0, 'AXDX': 2, 'SSYS': 0, 'ELF': 8, 'VERI': 0, 'TILE': 3, 'EBIX': 0, 'BIOS': 2, 'BCPC': 6, 'JILL': 9, 'SPAR': 3, 'KBAL': 3, 'DBD': 0, 'FNGN': 1, 'CHCT': 5, 'PRSC': 2, 'ROSE': 10, 'CRBP': 2, 'REI': 10, 'QADA': 0, 'SMCI': 0, 'UE': 5, 'CCS': 9, 'TRUP': 1, 'BPI': 9, 'MLAB': 0, 'ADUS': 2, 'MLNT': 2, 'RMTI': 2, 'CWST': 3, 'HTLD': 3, 'GTS': 2, 'OFIX': 2, 'AEIS': 0, 'FWRD': 3, 'MTDR': 10, 'NPO': 3, 'KIDS': 2, 'SGH': 0, 'STC': 1, 'HRI': 3, 'IMMU': 2, 'NXST': 9, 'MCFT': 9, 'HLI': 1, 'CPF': 1, 'MODN': 0, 'KRA': 6, 'CTRL': 0, 'DERM': 2, 'MTSI': 0, 'SPSC': 0, 'BLMN': 9, 'BREW': 8, 'COHU': 0, 'UTMD': 2, 'ZGNX': 2, 'INAP': 0, 'FLWS': 9, 'PVAC': 10, 'GBNK': 1, 'CHGG': 9, 'CRMT': 9, 'IART': 2, 'EXPO': 3, 'PRAH': 2, 'PFGC': 8, 'RARX': 2, 'TNC': 3, 'WWE': 9, 'HOME': 9, 'RP': 0, 'MC': 1, 'ATKR': 3, 'HIFS': 1, 'HONE': 1, 'MSEX': 4, 'WETF': 1, 'DHIL': 1, 'ASGN': 3, 'NYNY': 9, 'ADSW': 3, 'SAFE': 5, 'PLOW': 3, 'MRSN': 2, 'NSA': 5, 'CPLA': 9, 'PNK': 9, 'NATH': 9, 'HELE': 9, 'FSS': 3, 'MWA': 3, 'WING': 9, 'PTLA': 2, 'TMHC': 9, 'SFLY': 9, 'AXAS': 10, 'FRPT': 8, 'WRE': 5, 'MACK': 2, 'FTK': 6, 'PLUG': 3, 'INSM': 2, 'TAX': 9, 'SLAB': 0, 'PQG': 6, 'DXPE': 3, 'MB': 0, 'AIN': 3, 'AQUA': 4, 'DSKE': 3, 'EPM': 10, 'EVTC': 0, 'AAXN': 3, 'WAGE': 3, 'ADXS': 2, 'PTCT': 2, 'WMS': 3, 'CYTK': 2, 'HDSN': 3, 'AFH': 1, 'GDOT': 1, 'CRCM': 0, 'KBH': 9, 'HAIR': 2, 'SAIA': 3, 'GTHX': 2, 'PCYG': 0, 'RMBS': 0, 'TELL': 10, 'BBSI': 3, 'RAVN': 3, 'FBM': 3, 'SAIC': 0, 'BL': 0, 'PHX': 10, 'SNX': 0, 'ITI': 0, 'ESPR': 2, 'CORI': 2, 'ADRO': 2, 'LOXO': 2, 'ICHR': 0, 'KEM': 0, 'LJPC': 2, 'MG': 3, 'ISRL': 10, 'TTEK': 3, 'FUL': 6, 'JELD': 3, 'FMI': 2, 'MLP': 5, 'FIVN': 0, 'LMNR': 8, 'SSB': 1, 'CSGS': 0, 'IMGN': 2, 'IRWD': 2, 'ENSG': 2, 'SWM': 6, 'COUP': 0, 'RTYH8': 11, 'EHTH': 1, 'ZYNE': 2, 'CUTR': 2, 'RVLT': 3, 'PRTK': 2, 'MGLN': 2, 'GNBC': 1, 'MRLN': 1, 'NSSC': 0, 'IIIN': 3, 'SCS': 3, 'GFF': 3, 'PGTI': 3, 'KTOS': 3, 'RVNC': 2, 'BY': 1, 'ALGT': 3, 'LNN': 3, 'BRKS': 0, 'LORL': 9, 'ANIK': 2, 'ABCD': 9, 'MKSI': 0, 'SN': 10, 'LDL': 3, 'SLCA': 10, 'MTNB': 2, 'SGMS': 9, 'PFBC': 1, 'IDTI': 0, 'ALX': 5, 'JCOM': 0, 'OPB': 1, 'CASH': 1, 'RHP': 5, 'CAR': 3, 'ACXM': 0, 'ARRY': 2, 'QTWO': 0, 'AMKR': 0, 'EEX': 9, 'ASPS': 5, 'HAWK': 0, 'COLM': 9, 'WD': 1, 'NVCR': 2, 'VRNS': 0, 'ZIXI': 0, 'DFIN': 1, 'VEC': 3, 'FFIN': 1, 'CNCE': 2, 'POWI': 0, 'CVNA': 9, 'MLI': 3, 'ENV': 0, 'TLRD': 9, 'QTNT': 2, 'CRUS': 0, 'NVAX': 2, 'MBIN': 1, 'MRCY': 3, 'COBZ': 1, 'NYT': 9, 'FORM': 0, 'MGEN': 2, 'MTH': 9, 'NHTC': 8, 'EDGE': 2, 'EFII': 0, 'ZIOP': 2, 'MULE': 0, 'MHLD': 1, 'ULH': 3, 'OMCL': 2, 'EBSB': 1, 'CULP': 9, 'AWR': 4, 'LXRX': 2, 'CSTR': 1, 'CATM': 0, 'NPK': 3, 'VHI': 6, 'WLH': 9, 'GCBC': 1, 'PCRX': 2, 'SGC': 9, 'KOPN': 0, 'CHUY': 9, 'WNC': 3, 'REPH': 2, 'AMOT': 3, 'TGTX': 2, 'DS': 9, 'RGCO': 4, 'PFPT': 0, 'MSTR': 0, 'CSU': 2, 'DMRC': 0, 'RSYS': 0, 'CPK': 4, 'VVI': 3, 'ENTG': 0, 'HSC': 3, 'PSB': 5, 'NEO': 2, 'GEF': 6, 'BMTC': 1, 'MJCO': 0, 'IDCC': 0, 'IIVI': 0, 'MATW': 3, 'AVXS': 2, 'GVA': 3, 'OMAM': 1, 'NP': 6, 'VRTS': 1, 'CDE': 6, 'SPXC': 3, 'MCRB': 2, 'MEI': 0, 'BECN': 3, 'FLIC': 1, 'UIHC': 1, 'NVTA': 2, 'EPAM': 0, 'APOG': 3, 'SHOO': 9, 'SGRY': 2, 'VAC': 9, 'NSP': 3, 'SRDX': 2, 'BNFT': 0, 'APTI': 0, 'GOGO': 0, 'CSWI': 3, 'SEAS': 9, 'PSDO': 0, 'HRG': 8, 'CHE': 2, 'ONVO': 2, 'OMNT': 0, 'CFMS': 2, 'ORA': 4, 'WSBF': 1, 'RGEN': 2, 'VDSI': 0, 'IMDZ': 2, 'AFAM': 2, 'POL': 6, 'AJRD': 3, 'FOLD': 2, 'CLSD': 2, 'JONE': 10, 'SRCI': 10, 'BATRA': 9, 'PENN': 9, 'AZPN': 0, 'NEOS': 2, 'DEPO': 2, 'NCOM': 1, 'FFNW': 1, 'AMED': 2, 'FRAC': 10, 'KNX': 3, 'JBT': 3, 'EXXI': 10, 'FARO': 0, 'CIO': 5, 'BCOV': 0, 'VSAT': 0, 'RDUS': 2, 'SMTC': 0, 'GNMK': 2, 'PLUS': 0, 'MSA': 3, 'TNTR': 0, 'CNS': 1, 'BLUE': 2, 'TISI': 3, 'IBP': 9, 'USLM': 6, 'RUSHB': 3, 'TTD': 0, 'AMBR': 0, 'QTS': 5, 'HABT': 9, 'UEIC': 9, 'MNOV': 2, 'TRNC': 9, 'DECK': 9, 'BGC': 3, 'PLT': 0, 'VCYT': 2, 'PRAA': 1, 'BLKFDS': 11, 'PSTG': 0, 'UFPI': 3, 'GPRO': 9, 'RST': 0, 'HCI': 1, 'FATE': 2, 'SYNT': 0, 'DRRX': 2, 'KODK': 0, 'ATRC': 2, 'IPAR': 8, 'MDLY': 1, 'LFUS': 0, 'PCTY': 0, 'ELLI': 0, 'LQ': 9, 'III': 0, 'KWR': 6, 'ICUI': 2, 'LWAY': 8, 'VIVO': 2, 'NTNX': 0, 'PBH': 2, 'ACRS': 2, 'SBOW': 10, 'XCRA': 0, 'FIX': 3, 'LGIH': 9, 'AKTS': 0, 'CSV': 9, 'PETQ': 2, 'MYRG': 3, 'BLDR': 3, 'WTBA': 1, 'PMTS': 0, 'FSCT': 0, 'WTI': 10, 'CLVS': 2, 'WMGI': 2, 'USCR': 6, 'RLGT': 3, 'LLEX': 10, 'PAY': 0, 'KAMN': 3, 'EPAY': 0, 'ACLS': 0, 'ISTR': 1, 'SAGE': 2, 'BRC': 3, 'WLFC': 3, 'PZN': 1, 'MDCO': 2, 'ROG': 0, 'BHVN': 2, 'OCUL': 2, 'ZAGG': 9, 'ABM': 3, 'TPHS': 5, 'SYX': 0, 'FR': 5, 'TRNO': 5, 'GEFB': 6, 'AMN': 2, 'RRR': 9, 'INSY': 2, 'MTZ': 3, 'ARA': 2, 'TLGT': 2, 'KURA': 2, 'TPB': 8, 'JACK': 9, 'MPWR': 0, 'SBBP': 2, 'KBR': 3, 'YRCW': 3, 'MCS': 9, 'ENTL': 2, 'CSOD': 0, 'CSLT': 2, 'ABCB': 1, 'WNEB': 1, 'EXLS': 0, 'YELP': 0, 'HRTX': 2, 'ITIC': 1, 'RTIX': 2, 'ADMS': 2, 'WWD': 3, 'CLFD': 0, 'RM': 1, 'ALNA': 2, 'CRVL': 2, 'LKFN': 1, 'TRVN': 2, 'TSC': 1, 'ANAB': 2, 'PEN': 2, 'AERI': 2, 'REN': 10, 'PRIM': 3, 'SNDX': 2, 'NTB': 1, 'GKOS': 2, 'RMR': 5, 'SGYP': 2, 'UMH': 5, 'BMCH': 3, 'ATEN': 0, 'CIR': 3, 'SYNH': 2, 'HCKT': 0, 'TWLO': 0, 'THRM': 9, 'GNTY': 1, 'EHC': 2, 'TDOC': 2, 'FC': 3, 'SNDR': 3, 'CZR': 9, 'BHBK': 1, 'PETS': 9, 'HF': 5, 'SWX': 4, 'CSTE': 3, 'MSL': 1, 'CHUBA': 0, 'USPH': 2, 'VCRA': 2, 'NCSM': 10, 'KAI': 3, 'PLAY': 9, 'LZB': 9, 'EVR': 1, 'CELC': 2, 'BLMT': 1, 'UPLD': 0, 'CUDA': 0, 'PJC': 1, 'CALD': 0, 'BGSF': 3, 'RGNX': 2, 'RARE': 2, 'AYX': 0, 'DVAX': 2, 'FBNK': 1, 'CARO': 1, 'ARNA': 2, 'WINA': 9, 'VBIV': 2, 'TCMD': 2, 'QUOT': 0, 'HMSY': 2, 'EVC': 9, 'ECOM': 0, 'AXON': 2, 'APLS': 2, 'MMI': 5, 'NJR': 4, 'FONR': 2, 'CNAT': 2, 'AVID': 0, 'UCTT': 0, 'AQMS': 3, 'CPRX': 2, 'NDLS': 9, 'AMSWA': 0, 'RUTH': 9, 'FIZZ': 8, 'QLYS': 0, 'AVXL': 2, 'BCRX': 2, 'ABG': 9, 'PRTY': 9, 'SYRS': 2, 'PRI': 1, 'AAT': 5, 'WDFC': 8, 'CMP': 6, 'DCPH': 2, 'XLRN': 2, 'HOV': 9, 'IDRA': 2, 'ACBI': 1, 'FOXF': 9, 'FBIO': 2, 'SMP': 9, 'IRBT': 9, 'PEGA': 0, 'ITG': 1, 'ENVA': 1, 'LGND': 2, 'AIT': 3, 'RUSHA': 3, 'UIS': 0, 'EGBN': 1, 'SPA': 3, 'BEAT': 2, 'TVPT': 0, 'WATT': 3, 'MBUU': 9, 'NHI': 5, 'USAT': 0, 'BLBD': 3, 'CEVA': 0, 'RDI': 9, 'CRZO': 10, 'JNCE': 2, 'BABY': 2, 'TTGT': 0, 'AOBC': 9, 'NMIH': 1, 'MITK': 0, 'DAKT': 0, 'MDGL': 2, 'TBPH': 2, 'MNRO': 9, 'MDXG': 2, 'VIVE': 2, 'BELFB': 0, 'USD': 11, 'DF': 8, 'PGEM': 3, 'VALU': 1, 'BID': 9, 'NGVT': 6, 'DY': 3, 'FIVE': 9, 'SBGI': 9, 'QDEL': 2, 'MXL': 0, 'RYAM': 6, 'EVBG': 0, 'MOH': 2, 'BCOR': 0, 'FRBK': 1, 'PIRS': 2, 'TREX': 3, 'INGN': 2, 'NXTM': 2, 'TEN': 9, 'AST': 2, 'FRED': 9, 'SHLD': 9, 'AOSL': 0, 'FFWM': 1, 'GPT': 5, 'INST': 0, 'NTRA': 2, 'AMPH': 2, 'TNET': 3, 'CCXI': 2, 'SYNA': 0, 'SPKE': 4, 'ANCX': 1, 'RBB': 1, 'RH': 9, 'MYOK': 2, 'CVGW': 8, 'CMPR': 0, 'PODD': 2, 'AKBA': 2, 'LAWS': 3, 'BYD': 9, 'AKAO': 2, 'MEET': 0, 'RNG': 0, 'SFBS': 1, 'EAT': 9, 'FNKO': 9, 'PDFS': 0, 'ETSY': 0, 'PUB': 1, 'ZOES': 9, 'LPSN': 0, 'EDIT': 2, 'VIRT': 1, 'GERN': 2, 'OFLX': 3, 'QTNA': 0, 'BOJA': 9, 'PRMW': 8, 'CHDN': 9, 'PI': 0, 'PACB': 2, 'WOR': 6, 'PICO': 9, 'LMNX': 2, 'VSAR': 2, 'SCMP': 2, 'XENT': 2, 'CRY': 2, 'WOW': 9, 'TCBI': 1, 'ATNX': 2, 'EQBK': 1, 'REVG': 3, 'ELVT': 1, 'FLXN': 2, 'RPD': 0, 'WHG': 1, 'ESNT': 1, 'JJSF': 8, 'CDXS': 6, 'MLR': 3, 'GRPN': 9, 'DENN': 9, 'ASIX': 6, 'AIMC': 3, 'CPS': 9, 'FCPT': 5, 'TVTY': 2, 'PETX': 2, 'TYPE': 0, 'INWK': 3, 'DEL': 6, 'FNSR': 0, 'CMD': 2, 'LPX': 6, 'ALRM': 0, 'PCMI': 0, 'GTT': 0, 'RTEC': 0, 'COKE': 8, 'REIS': 0, 'PRO': 0, 'EGP': 5, 'VRAY': 2, 'NOVT': 0, 'ALRN': 2, 'NVRO': 2, 'TCX': 0, 'OMN': 6, 'VTVT': 2, 'CLCT': 9, 'CVGI': 3, 'ATRS': 2, 'PARR': 10, 'OSIS': 0, 'PRGS': 0, 'RYTM': 2, 'MVIS': 0, 'VRTU': 0, 'ETM': 9, 'INVA': 2, 'BSF': 1, 'NANO': 0, 'EGRX': 2, 'ALDR': 2, 'LOPE': 9, 'MED': 8, 'HAE': 2, 'CIVI': 2, 'EVH': 2, 'INSE': 9, 'GNRC': 3, 'CNOB': 1, 'HTBK': 1, 'HZN': 9, 'STRA': 9, 'TECD': 0, 'CCF': 6, 'FCFS': 1, 'NXRT': 5, 'MMS': 0, 'RCM': 2, 'CENX': 6, 'CALA': 2, 'CPSI': 2, 'GBCI': 1, 'KRO': 6, 'TTMI': 0, 'LC': 1, 'ENT': 9, 'BOX': 0, 'DLX': 3, 'SHAK': 9, 'ERI': 9, 'RDNT': 2, 'CAMP': 0, 'KFRC': 3, 'SPRO': 2, 'BGS': 8, 'B': 3, 'HWKN': 6, 'HCSG': 3, 'BFS': 5, 'THC': 2, 'GMRE': 5, 'RICK': 9, 'UEC': 10, 'HBP': 3, 'SONC': 9, 'PUMP': 10, 'GBT': 2, 'BWFG': 1, 'HQY': 2, 'STAA': 2, 'MBFI': 1, 'MTX': 6, 'QUAD': 3, 'ROX': 8, 'OCN': 1, 'OXFD': 2, 'HMTV': 9, 'SAM': 8, 'NLS': 9, 'DOC': 5, 'TXMD': 2, 'PRFT': 0, 'CCMP': 0, 'MDCA': 9, 'ASC': 10, 'ATSG': 3, 'CORT': 2, 'VRS': 6, 'CCC': 6, 'TPIC': 3, 'LAD': 9, 'WGO': 9, 'STML': 2, 'KW': 5, 'RETA': 2, 'HDP': 0, 'MRT': 5, 'EGOV': 0, 'BLCM': 2, 'KMG': 6, 'KNSL': 1, 'CVLT': 0, 'PRLB': 3, 'CBPX': 3, 'CARB': 0, 'LABL': 3, 'ABTX': 1, 'FRAN': 9, 'JAG': 10, 'PZZA': 9, 'MDC': 9, 'AKCA': 2, 'AZZ': 3, 'SREV': 0, 'CAI': 3, 'WTS': 3, 'CBTX': 1, 'NNBR': 3, 'VBTX': 1, 'DOOR': 3, 'LANC': 8, 'MCRN': 3, 'IMAX': 9, 'PRTA': 2, 'SHLM': 6, 'PLCE': 9, 'SITE': 3, 'CLXT': 2, 'HI': 3, 'SGMO': 2, 'FELE': 3, 'CTRE': 5, 'ACIW': 0, 'IVAC': 0, 'VNDA': 2, 'OVID': 2, 'ZEN': 0, 'EIGI': 0, 'ECOL': 3, 'HMHC': 9, 'SMBC': 1, 'MSBI': 1, 'WSFS': 1, 'KNL': 3, 'BOFI': 1, 'CIEN': 0, 'IOVA': 2, 'TXRH': 9, 'TBI': 3, 'NEWR': 0, 'KS': 6, 'NYMX': 2, 'OMER': 2, 'HZO': 9, 'OSTK': 9, 'ANIP': 2, 'ATU': 3, 'DAN': 9, 'AHH': 5, 'SPWH': 9, 'PGNX': 2, 'ASTE': 3, 'PPBI': 1, 'NVEC': 0, 'CCRN': 2, 'UHT': 5, 'CARA': 2, 'OSUR': 2, 'BATRK': 9, 'FRTA': 6, 'GMED': 2, 'FN': 0, 'AAON': 3, 'HNI': 3, 'FORR': 3, 'HA': 3, 'FGEN': 2, 'GDEN': 9, 'ERII': 3, 'PCYO': 4, 'CHRS': 2, 'VICR': 3, 'ACOR': 2, 'ENZ': 2, 'CLDR': 0, 'QSII': 2, 'MASI': 2, 'LCII': 9, 'IPHI': 0, 'NERV': 2, 'SEM': 2, 'PBYI': 2, 'OXM': 9, 'TNAV': 0, 'CBRL': 9, 'SNBR': 9, 'KLDX': 6, 'AMWD': 3, 'RRGB': 9, 'ARAY': 2, 'BCEI': 10, 'ACIA': 0, 'TOCA': 2, 'SNNA': 2, 'SP': 3, 'DYAX': 2, 'ENS': 3, 'GNCA': 2, 'CENT': 8, 'NCS': 3, 'MNTA': 2, 'FSB': 1, 'TPH': 9, 'GLUU': 0, 'KMT': 3, 'ELGX': 2, 'DOVA': 2, 'SBRA': 5, 'ATHX': 2, 'ALG': 3, 'CVCO': 9, 'FICO': 0, 'FMSA': 10, 'ROLL': 3, 'PBPB': 9, 'MOBL': 0, 'TTS': 9, 'RNGR': 10, 'SYKE': 0, 'NEOG': 2, 'ORN': 3, 'MGPI': 8, 'CERS': 2, 'EPZM': 2, 'MGNX': 2, 'HCCI': 3, 'MSFUT': 11, 'ATRO': 3, 'LOB': 1, 'EXAS': 2, 'HY': 3, 'MDSO': 2, 'SCL': 6, 'TMP': 1, 'CRIS': 2, 'EME': 3, 'CVI': 10, 'WTTR': 10, 'MGRC': 3, 'IMPV': 0}
    return ticker_sector


def sp500_ticker_sector_mapping():
    ticker_sector = {'CTL': 7, 'COTY': 8, 'DISCK': 9, 'AIV': 5, 'FRT': 5, 'CF': 6, 'CA': 0, 'MHK': 9, 'FLIR': 0, 'GS': 1, 'HD': 9, 'AMD': 0, 'MAA': 5, 'KR': 8, 'RTN': 3, 'MKC': 8, 'BLL': 6, 'UAA': 9, 'CCI': 5, 'IRM': 5, 'NVDA': 0, 'SYMC': 0, 'DUK': 4, 'IBM': 0, 'LNC': 1, 'LMT': 3, 'EBAY': 0, 'EXC': 4, 'ARNC': 3, 'MON': 6, 'TAP': 8, 'PXD': 10, 'FL': 9, 'USB': 1, 'TIF': 9, 'GGP': 5, 'KSS': 9, 'CTXS': 0, 'RMD': 2, 'CAH': 2, 'MCO': 1, 'PH': 3, 'IP': 6, 'HRS': 0, 'SYY': 8, 'BIIB': 2, 'TWX': 9, 'MS': 1, 'UDR': 5, 'WMT': 8, 'OMC': 9, 'NFX': 10, 'NRG': 4, 'KHC': 8, 'SPGI': 1, 'AXP': 1, 'TRV': 1, 'ZION': 1, 'PKG': 6, 'HON': 3, 'WBA': 8, 'EVHC': 2, 'NCLH': 9, 'ITW': 3, 'FB': 0, 'D': 4, 'EA': 0, 'CAG': 8, 'CTSH': 0, 'COO': 2, 'NSC': 3, 'JNJ': 2, 'XRX': 0, 'DGX': 2, 'DHR': 2, 'BA/': 3, 'SPG': 5, 'IFF': 6, 'PM': 8, 'GE': 3, 'ABC': 2, 'ETR': 4, 'ABBV': 2, 'JEC': 3, 'ROST': 9, 'CINF': 1, 'NWS': 9, 'INFO': 3, 'MTB': 1, 'SBAC': 5, 'ADM': 8, 'STZ': 8, 'SWKS': 0, 'AAPL': 0, 'GT': 9, 'HRL': 8, 'NUE': 6, 'INCY': 2, 'KMX': 9, 'CB': 1, 'MCD': 9, 'GOOG': 0, 'PPG': 6, 'F': 9, 'TSN': 8, 'MRO': 10, 'V': 0, 'DFS': 1, 'UHS': 2, 'SCG': 4, 'O': 5, 'EOG': 10, 'BBT': 1, 'AVGO': 0, 'XEC': 10, 'AMAT': 0, 'LLL': 3, 'MDT': 2, 'RHI': 3, 'ES': 4, 'ROK': 3, 'TMK': 1, 'TGT': 9, 'EXR': 5, 'EL': 8, 'GLW': 0, 'REG': 5, 'GIS': 8, 'ROP': 3, 'BMY': 2, 'ADBE': 0, 'GPS': 9, 'CI': 2, 'HUM': 2, 'ECL': 6, 'APD': 6, 'BRK.B': 1, 'MCK': 2, 'RJF': 1, 'PX': 6, 'KO': 8, 'CBS': 9, 'VAR': 2, 'TROW': 1, 'WYNN': 9, 'KEY': 1, 'YUM': 9, 'KMB': 8, 'RHT': 0, 'CNC': 2, 'DISCA': 9, 'APC': 10, 'SLG': 5, 'HSIC': 2, 'ACN': 0, 'MU': 0, 'ANTM': 2, 'ATVI': 0, 'IDXX': 2, 'DOV': 3, 'T': 7, 'XEL': 4, 'CDNS': 0, 'K': 8, 'NOC': 3, 'FOXA': 9, 'PPL': 4, 'CMG': 9, 'PNW': 4, 'PSX': 10, 'ABT': 2, 'SIG': 9, 'QCOM': 0, 'CHD': 8, 'AMG': 1, 'ORLY': 9, 'WY': 5, 'PLD': 5, 'PRGO': 2, 'GILD': 2, 'NTRS': 1, 'HAS': 9, 'XOM': 10, 'CMI': 3, 'GPC': 9, 'ALL': 1, 'JWN': 9, 'FTI': 10, 'SNPS': 0, 'LH': 2, 'FISV': 0, 'MAT': 9, 'DVN': 10, 'AEE': 4, 'STT': 1, 'GOOGL': 0, 'AAP': 9, 'NLSN': 3, 'PHM': 9, 'ALB': 6, 'UNM': 1, 'BDX': 2, 'JCI': 3, 'CMA': 1, 'PRU': 1, 'PCLN': 9, 'SLB': 10, 'PVH': 9, 'ADSK': 0, 'DVA': 2, 'ZBH': 2, 'AAL': 3, 'MLM': 6, 'MAC': 5, 'XRAY': 2, 'DHI': 9, 'IT': 0, 'IR': 3, 'MYL': 2, 'VTR': 5, 'CHRW': 3, 'CXO': 10, 'WLTW': 1, 'MAR': 9, 'AMGN': 2, 'LUK': 1, 'XLNX': 0, 'NFLX': 0, 'UAL': 3, 'PKI': 2, 'ALXN': 2, 'HCP': 5, 'PAYX': 0, 'COL': 3, 'LEG': 9, 'MDLZ': 8, 'LYB': 6, 'INTC': 0, 'UNP': 3, 'STI': 1, 'PEP': 8, 'ILMN': 2, 'DE': 3, 'WM': 3, 'JNPR': 0, 'MOS': 6, 'TDG': 3, 'AZO': 9, 'DIS': 9, 'DTE': 4, 'VLO': 10, 'HAL': 10, 'TEL': 0, 'EIX': 4, 'CMS': 4, 'WU': 0, 'CTAS': 3, 'ISRG': 2, 'RSG': 3, 'PBCT': 1, 'LKQ': 9, 'ORCL': 0, 'TMO': 2, 'SWK': 9, 'DXC': 0, 'BK': 1, 'SBUX': 9, 'XYL': 3, 'WAT': 2, 'TSCO': 9, 'KMI': 10, 'CSCO': 0, 'BHF': 1, 'ALGN': 2, 'DRE': 5, 'ETN': 3, 'CFG': 1, 'XL': 1, 'SHW': 6, 'SYF': 1, 'WDC': 0, 'WFC': 1, 'SO': 4, 'HP': 10, 'NAVI': 1, 'TPR': 9, 'CHK': 10, 'AJG': 1, 'COG': 10, 'ARE': 5, 'AWK': 4, 'DG': 9, 'LRCX': 0, 'HCN': 5, 'APH': 0, 'DPS': 8, 'EMN': 6, 'FFIV': 0, 'INTU': 0, 'LUV': 3, 'HOG': 9, 'AME': 3, 'FLR': 3, 'CSRA': 0, 'HES': 10, 'SJM': 8, 'ALLE': 3, 'UA': 9, 'PFG': 1, 'NI': 4, 'OKE': 10, 'NWL': 9, 'WHR': 9, 'NWSA': 9, 'AMP': 1, 'ANSS': 0, 'CME': 1, 'QRVO': 0, 'COF': 1, 'ULTA': 9, 'EXPD': 3, 'ALK': 3, 'NTAP': 0, 'C': 1, 'NEM': 6, 'MMC': 1, 'APA': 10, 'COP': 10, 'NBL': 10, 'FITB': 1, 'AVB': 5, 'BBY': 9, 'FCX': 6, 'VNO': 5, 'CSX': 3, 'FDX': 3, 'PSA': 5, 'PGR': 1, 'PG': 8, 'PNR': 3, 'HBI': 9, 'CVX': 10, 'MCHP': 0, 'CAT': 3, 'LNT': 4, 'RF': 1, 'L': 1, 'FOX': 9, 'PDCO': 2, 'AET': 2, 'BCR': 2, 'CELG': 2, 'MET': 1, 'DWDP': 6, 'BSX': 2, 'EMR': 3, 'LEN': 9, 'GPN': 0, 'DLTR': 9, 'RCL': 9, 'DRI': 9, 'TSS': 0, 'FLS': 3, 'GRMN': 9, 'BWA': 9, 'VMC': 6, 'TRIP': 9, 'ESRX': 2, 'MMM': 3, 'OXY': 10, 'BLK': 1, 'BXP': 5, 'SYK': 2, 'DAL': 3, 'BAC': 1, 'CNP': 4, 'FIS': 0, 'PFE': 2, 'ADP': 0, 'AOS': 3, 'DLR': 5, 'EQR': 5, 'LLY': 2, 'MRK': 2, 'AYI': 3, 'WYN': 9, 'CERN': 2, 'LB': 9, 'KIM': 5, 'PCAR': 3, 'KSU': 3, 'AON': 1, 'GM': 9, 'TJX': 9, 'GD': 3, 'MPC': 10, 'UTX': 3, 'CHTR': 9, 'APTV': 9, 'VZ': 7, 'HIG': 1, 'VRSK': 3, 'AVY': 6, 'LOW': 9, 'SNI': 9, 'HPE': 0, 'CVS': 8, 'EFX': 3, 'MGM': 9, 'HOLX': 2, 'CL': 8, 'SCHW': 1, 'KORS': 9, 'TXN': 0, 'BF.B': 8, 'HRB': 1, 'ADI': 0, 'HST': 5, 'PEG': 4, 'FBHS': 3, 'ED': 4, 'VRSN': 0, 'MAS': 3, 'EW': 2, 'M': 9, 'ADS': 0, 'JBHT': 3, 'CBOE': 1, 'NEE': 4, 'UNH': 2, 'RRC': 10, 'IVZ': 1, 'BEN': 1, 'HCA': 2, 'VIAB': 9, 'MA': 0, 'IPG': 9, 'VFC': 9, 'AKAM': 0, 'HSY': 8, 'CMCSA': 9, 'CBG': 5, 'ICE': 1, 'PWR': 3, 'DISH': 9, 'AFL': 1, 'COST': 8, 'WEC': 4, 'NKE': 9, 'AMZN': 9, 'PNC': 1, 'NDAQ': 1, 'REGN': 2, 'URI': 3, 'SNA': 9, 'CCL': 9, 'MTD': 2, 'SRCL': 3, 'AMT': 5, 'HLT': 9, 'FE': 4, 'A': 2, 'SEE': 6, 'CPB': 8, 'MSFT': 0, 'KLAC': 0, 'ANDV': 10, 'PCG': 4, 'AES': 4, 'EXPE': 9, 'WMB': 10, 'FTV': 3, 'PYPL': 0, 'ETFC': 1, 'WRK': 6, 'ZTS': 2, 'HPQ': 0, 'MO': 8, 'CRM': 0, 'FMC': 6, 'ESS': 5, 'BHGE': 10, 'STX': 0, 'RL': 9, 'NOV': 10, 'AIZ': 1, 'FAST': 3, 'AIG': 1, 'TXT': 3, 'VRTX': 2, 'AGN': 2, 'BA': 2, 'EQT': 10, 'CLX': 8, 'IQV': 2, 'EQIX': 5, 'RE': 1, 'JPM': 1, 'SRE': 4, 'GWW': 3, 'HBAN': 1, 'AEP': 4, 'UPS': 3, 'MSI': 0, 'MNST': 8}
    return ticker_sector


 # "Assets ['LUK-W', 'YUM-W', 'TAP/A', 'JCI-W', 'XRX-W', 'VAR-W', 'CAG-W', 'VNO-W', 'RMDCD', 'LEN/B', 'BA/', 'PG/WD', 'HCP-W', 'STZ/B', 'MKC/V', 'APD-W', 'LMT/WD', 'DHR-W', 'HPE-W', 'WRK-W', 'HON-W', 'MPC/C'] not in group mapping"

def select_industry_from_sp500_wiki_for_io(sector_str):
    sp500_df = pd.read_csv('sp500_csv_list.csv')
    sector_rows = sp500_df[sp500_df['GICS Sector'] == sector_str]
    sector_list = list(sector_rows['Ticker symbol'])
    sector_list_df = pd.DataFrame(sector_list, columns = ['Ticker'])
    return sector_list_df


def add_kalman_filter_value_on_df_na_dropped(df, target_col, kf_col_str):
    # Initialize a Kalman Filter.
    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)

    kf_mean, _ = kf.filter(df[target_col].dropna())
    df.dropna(inplace = True)
    df[kf_col_str] = kf_mean

    return df

# Get alphalens results:
def get_alphalens_results(factor_df, metric_str, price_df, periods):
# here an example of periods = (1, 5, 10)
    factor_data_metric = alphalens.utils.get_clean_factor_and_forward_returns(factor_df[metric_str], 
                                                                       price_df, 
                                                                       quantiles=5,
                                                                       bins = None,
                                                                       # groupby=ticker_sector,
                                                                       # groupby_labels=sector_names,
                                                                       periods=periods)
    return alphalens.tears.create_full_tear_sheet(factor_data_metric, by_group=False)