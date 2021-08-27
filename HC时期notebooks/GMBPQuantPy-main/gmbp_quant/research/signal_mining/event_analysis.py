import gmbp_quant.common.utils.miscs as mu
from scipy import stats
import pandas as pd


def summarize_event_details(event_details, return_cols,
                            win_rate_threshold=0.0,
                            one_sample_hypothesis_test_mean=0.0):
    return_cols = list(mu.iterable_to_tuple(return_cols, raw_type='str'))
    returns = event_details[return_cols]

    avg_returns = returns.mean().to_frame(name='Avg').T
    win_rates = ((returns > win_rate_threshold).sum()/len(returns)).to_frame(name='WinRate').T
    results = stats.ttest_1samp(returns.values, popmean=one_sample_hypothesis_test_mean,
                                nan_policy='omit')
    tstats = pd.DataFrame(results.statistic, index=return_cols, columns=['t-stats']).T
    pvalues = pd.DataFrame(results.pvalue, index=return_cols, columns=['p-value']).T

    summary = pd.concat([avg_returns, win_rates, tstats, pvalues])
    summary.index.name = 'Item'

    return summary
#
