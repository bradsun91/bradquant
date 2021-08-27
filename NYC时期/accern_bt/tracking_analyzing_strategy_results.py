
#---------------------------------------------------------------Tracking and Analysizing Strategy Results---------------------------------------------------------------------------#

import pandas as pd
import glob
import csv

################ 1. Startegy Tracking Summary Codebase ################:

def combine_all_strat_results_into_one_csv(result_output_file):
    result_path = '/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/results/*.csv'
    result_output_file = result_output_file
    files = sorted(glob.glob(result_path)) 
    header_saved = False
    with open(result_output_file,'wb') as fout:
        for filename in files:
            print ('Processing', filename)
            with open(filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header.encode('utf-8'))
                    header_saved = True
                for line in fin:
                    fout.write(line.encode('utf-8'))
                    

def clean_combined_strat_csv(result_output_file):
    read_results = pd.read_csv(result_output_file, error_bad_lines=False)
    read_results = read_results[read_results.algo_id != 'algo_id']
    read_results = read_results[read_results.sharpe != '--']
    read_results.drop(['algo_id'], axis=1, inplace=True)
    read_results.dropna(inplace=True)
    title_to_convert_to_float = ['alpha', 'beta', 'sharpe', 'sortino', 'volatility']
    title_to_strip_pct = ['total_returns', 'benchmark_returns', 'max_drawdown']
    
    for col in title_to_convert_to_float:
        read_results[col] = read_results[col].apply(lambda x: float(x))
    for col in title_to_strip_pct:
        read_results[col] = read_results[col].apply(lambda x: x.strip('%'))
        read_results[col] = read_results[col].apply(lambda x: float(x))
        read_results[col] = read_results[col]/100
    
    pd.options.display.max_colwidth = 100
    read_results.columns = ['algo_name','backtest_id', 'alpha', 'beta', 'sharpe', 'sortino', 'total_returns',
       'benchmark_returns', 'volatility', 'max_drawdown', 'backtest_url']
    read_results.reset_index(inplace=True)
    del read_results['index']
    return read_results


def combine_all_configs(result_output_file):
    path = r'/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/configs/'              
    all_files = sorted(glob.glob(os.path.join(path, "*.csv")))            
    df_from_each_file = (pd.read_csv(f, error_bad_lines=False) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file)
    concatenated_df.to_csv(result_output_file)
    # '/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/configs/all_strat_configs_by_2_20.csv'



# all_strat_results_path = '/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/results/all_strat_results_by_3_6.csv'
# all_strat_configs_path = '/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/configs/all_strat_configs_by_3_6.csv'
# strat_summry_output_path = '/Users/Brad Sun/Dropbox/ACCERN/json_backtest_record/strat_summaries/strat_summary_by_3_6.csv'
# # Write into raw results file:
# combine_all_strat_results_into_one_csv(all_strat_results_path)

# clean_strat_file = clean_combined_strat_csv(all_strat_results_path)
# combine_all_configs(all_strat_configs_path)
# configs = pd.read_csv(all_strat_configs_path)
# merged = clean_strat_file.merge(configs, on='algo_name')
# merged.to_csv(strat_summry_output_path)
# merge_ = pd.read_csv(strat_summry_output_path)



################ 2. Use code above to generate heatmaps ################

# all_strats = pd.read_csv(strat_summry_output_path)
# print ('Number of total results in the file: ', len(all_strats))
# print ('Snapshot:')
# all_strats.head(3)
# research = all_strats[all_strats.algo_name.str.contains('3_6')]
# research['annualized_returns'] = (1+research['total_returns'])**(1/4.25) - 1

# # Construct a heatmap analysis graph for all strats:
# df = pd.DataFrame(index = research.index)
# df['sharpe'] = research.sharpe
# df['long_exposure'] = research.long_half
# df['num_stocks'] = research.num_stocks_to_trade
# df['capital_capacity'] = research.capital
# df['alpha'] = research.alpha
# df['beta'] = research.beta
# df['total_returns'] = research.total_returns
# df['volatility'] = research.volatility
# df['max_drawdown'] = research.max_drawdown
# df['annualized_returns'] = research.annualized_returns
# df_reset_index = df.reset_index()

# df_1_mil = df_reset_index[df_reset_index.capital_capacity == 1000000]
# df_100_mil = df_reset_index[df_reset_index.capital_capacity == 100000000]

# sharpe_1_mil = pd.pivot_table(df_1_mil, index = 'num_stocks', columns=['long_exposure'], values='sharpe')
# alpha_1_mil = pd.pivot_table(df_1_mil, index = 'num_stocks', columns=['long_exposure'], values='alpha')
# beta_1_mil = pd.pivot_table(df_1_mil, index = 'num_stocks', columns=['long_exposure'], values='beta')
# vol_1_mil = pd.pivot_table(df_1_mil, index = 'num_stocks', columns=['long_exposure'], values='volatility')
# total_r_1_mil = pd.pivot_table(df_1_mil, index = 'num_stocks', columns=['long_exposure'], values='total_returns')
# annualized_1_mil = pd.pivot_table(df_1_mil, index = 'num_stocks', columns=['long_exposure'], values='annualized_returns')

# sharpe_100_mil = pd.pivot_table(df_100_mil, index = 'num_stocks', columns=['long_exposure'], values='sharpe')
# alpha_100_mil = pd.pivot_table(df_100_mil, index = 'num_stocks', columns=['long_exposure'], values='alpha')
# beta_100_mil = pd.pivot_table(df_100_mil, index = 'num_stocks', columns=['long_exposure'], values='beta')
# vol_100_mil = pd.pivot_table(df_100_mil, index = 'num_stocks', columns=['long_exposure'], values='volatility')
# total_r_100_mil = pd.pivot_table(df_100_mil, index = 'num_stocks', columns=['long_exposure'], values='total_returns')
# annualized_100_mil = pd.pivot_table(df_100_mil, index = 'num_stocks', columns=['long_exposure'], values='annualized_returns')


# create four different templates of generating heatmaps

def get_heatmap(df, metric_str, cap, color_1, color_2, vmax, vmin):
    fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches
    cmap = sns.diverging_palette(color_1, color_2, sep=20, as_cmap=True)
    sns.heatmap(df, cmap = cmap, ax=ax, vmax=vmax, vmin=vmin, annot=True,linewidths=.5)
    plt.title('Parameter_Sensitivity_Analysis_on_{} (${}million)'.format(metric_str, cap), fontsize = 14)
    ax.invert_yaxis()
    
def get_heatmap_plain(df, metric_str, cap, vmax, vmin):
    fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches
#     cmap = sns.diverging_palette(color_1, color_2, sep=20, as_cmap=True)
    sns.heatmap(df, cmap = 'Blues', ax=ax, vmax=vmax, vmin=vmin, annot=True,linewidths=.5)
    plt.title('Parameter_Sensitivity_Analysis_on_{} (${}million)'.format(metric_str, cap), fontsize = 14)
    ax.invert_yaxis()
    
def get_heatmap_plain_no_min_max(df, metric_str, cap):
    fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches
#     cmap = sns.diverging_palette(color_1, color_2, sep=20, as_cmap=True)
    sns.heatmap(df, cmap = 'Blues', ax=ax, annot=True,linewidths=.5)
    plt.title('Parameter_Sensitivity_Analysis_on_{} (${}million)'.format(metric_str, cap), fontsize = 14)
    ax.invert_yaxis()
    
def get_heatmap_plain_no_min_max_vol(df, metric_str, cap):
    fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches
#     cmap = sns.diverging_palette(color_1, color_2, sep=20, as_cmap=True)
    sns.heatmap(df, cmap = 'Greys', ax=ax, annot=True,linewidths=.5)
    plt.title('Parameter_Sensitivity_Analysis_on_{} (${}million)'.format(metric_str, cap), fontsize = 14)
    ax.invert_yaxis()

# get_heatmap_plain_no_min_max(sharpe_1_mil, 'Sharpe', '1')
# get_heatmap_plain_no_min_max(sharpe_100_mil, 'Sharpe', '100')
# get_heatmap_plain_no_min_max(alpha_1_mil, 'Alpha', '1')
# get_heatmap_plain_no_min_max(alpha_100_mil, 'Alpha', '100')
# get_heatmap(beta_1_mil, 'Beta', '1', 10, 100, 1.1, -1.1)
# get_heatmap(beta_100_mil, 'Beta', '100', 10, 100, 1.1, -1.1)
# get_heatmap(total_r_1_mil, 'Total_Returns', '1', 10, 100, 2, -2)
# get_heatmap(total_r_100_mil, 'Total_Returns', '100', 10, 100, 2, -2)
# get_heatmap_plain_no_min_max_vol(vol_1_mil, 'Portfolio_Volatility', '1')
# get_heatmap_plain_no_min_max_vol(vol_100_mil, 'Portfolio_Volatility', '100')
# get_heatmap(annualized_1_mil, 'Annualized_Returns', '1', 10, 200, 0.3, -0.3)
# get_heatmap(annualized_100_mil, 'Annualized_Returns', '100', 10, 200, 0.3, -0.3)


################ 3. Get Three Columns for FF and HFRI graphs ################
# # Only get strategies with 1mil capacity:
# one_mil_df = research[research.algo_name.str.contains('3_2')]

# # Let's see if the number of these results are correct 
# print ('Number of results: ', len(one_mil_df)) 
# # Drop whatever are not the three columns
# one_mil_df.drop(['Unnamed: 0', 'algo_name', 'alpha', 'beta', 'sharpe',
#        'sortino', 'total_returns', 'benchmark_returns', 'volatility',
#        'max_drawdown', 'backtest_url', 'Unnamed: 0.1',
#        'MAX_LONG_POSITION_SIZE', 'MAX_SHORT_POSITION_SIZE', 'algo_file',
#        'bottom_pct_to_trade', 'capital', 'data_file', 'end_date',
#        'event_impact_pos', 'event_relevance_cutoff', 
#        'long_sentiment_value', 'num_stocks_long', 'num_stocks_short',
#         'short_half', 'start_date',
#        'top_bottom_pct_to_trade', 'top_pct_to_trade','annualized_returns'], axis = 1, inplace = True)
# # Check:
# print ('Checking snapshot: ', one_mil_df.head(2))
# # Export file to feed into Quantopian's research enviroment
# one_mil_df.to_csv('3_2_GMO_daily_sp500_sum_ev_sentiment_backtest_id_and_paramtrs.csv')
# # Next step is to open the exported csv file using sublime and make the three columns readable by quantopian's code built by Sam

################ 4. Getting anchor strategy's backtest_id to generate the heatmap table for ff and hrfi ################

#------------------------------------------------------------------------------------------------------------------------------------------#
