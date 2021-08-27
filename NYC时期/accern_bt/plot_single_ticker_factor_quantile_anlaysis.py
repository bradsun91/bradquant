# Refer to ipynb 1_11_Renaissance_new_data_generating_and_Alphalens_all_in_one_SPY_Conquest_Capital
"""
pricing_read = pd.read_csv('1_11_spy_pricing_for_alphalens_conquest_final.csv')
accern_factor = pd.read_csv('1_11_accern_factor_for_alphalens_reninv_final.csv')
"""
def single_ticker_quantile_return_analysis(factor_csv_file, price_csv_file, target_metric_str):
	pricing_read = pd.read_csv(price_csv_file)
	accern_factor = pd.read_csv(factor_csv_file)
	accern_factor_entity_sentiment_sort = accern_factor.sort_values(target_metric_str, ascending=False)
	accern_factor_entity_sentiment = pd.DataFrame(index = accern_factor_entity_sentiment_sort.index)
	accern_factor_entity_sentiment[target_metric_str] = accern_factor_entity_sentiment_sort[target_metric_str]
	accern_factor_entity_sentiment['datetime'] = accern_factor_entity_sentiment_sort['datetime']
	accern_factor_entity_sentiment.index = accern_factor_entity_sentiment.datetime

	# Cut this series into 5 different quantiles
	accern_factor_entity_sentiment['quantiles'] = pd.qcut(accern_factor_entity_sentiment[target_metric_str], 5, labels=np.arange(1, 6, 1))
	accern_factor_entity_sentiment['datetime'] = accern_factor_entity_sentiment.index
	merged_entity_sentiment = accern_factor_entity_sentiment.merge(pricing_read, on='datetime')
	merged_entity_sentiment_sort = merged_entity_sentiment.sort_values('datetime')

	merged_entity_sentiment_sort['1_period_return_forward'] = merged_entity_sentiment_sort['SPY'].pct_change(periods = 1).shift(-1)
	merged_entity_sentiment_sort['5_period_return_forward'] = merged_entity_sentiment_sort['SPY'].pct_change(periods = 5).shift(-5)
	merged_entity_sentiment_sort['10_period_return_forward'] = merged_entity_sentiment_sort['SPY'].pct_change(periods = 10).shift(-10)

	one_period_mean_return_forward = merged_entity_sentiment_sort.groupby(['quantiles'])['1_period_return_forward'].mean()
	five_period_mean_return_forward = merged_entity_sentiment_sort.groupby(['quantiles'])['5_period_return_forward'].mean()
	ten_period_mean_return_forward = merged_entity_sentiment_sort.groupby(['quantiles'])['10_period_return_forward'].mean()

	mean_return_quantile_analysis_df = pd.DataFrame()
	mean_return_quantile_analysis_df['1_period_mean_return_forward'] = one_period_mean_return_forward
	mean_return_quantile_analysis_df['5_period_mean_return_forward'] = five_period_mean_return_forward
	mean_return_quantile_analysis_df['10_period_mean_return_forward'] = ten_period_mean_return_forward

	mean_return_quantile_analysis_df.plot(kind='bar', figsize=(18, 6))