# spearman correlation_between_two_series_lag: from inear corrlation lecture in quantopian:
# np.random.seed(161)
# X = np.random.rand(10)
# Y = np.random.rand(10)
def spearman_corr(X, Y)
	plt.scatter(X, Y)
	
	r_s = stats.spearmanr(X, Y)
	print ('Spearman Rank Coefficient: ', r_s[0])
	print ('p-value', r_s[1])


# how to test if a ranking system work using spearman ranking system and forward-walking correlation 
# see quantopian's linear correlation lecture and 'the art of not following the market', 'The good and the bad, and the correlated...' in the forum

 