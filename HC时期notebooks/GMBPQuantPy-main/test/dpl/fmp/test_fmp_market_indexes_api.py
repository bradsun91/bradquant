import pandas as pd
import gmbp_quant.common.utils.datetime_utils as dtu
import gmbp_quant.common.unittest as ut
from gmbp_quant.dpl.fmp.fmp_market_indexes_api import request_historical_daily_prices


class TestFMPMarketIndexesAPI(ut.TestCase):
    def test_request_historical_daily_prices(self):
        symbol = '^NDX'
        target = request_historical_daily_prices(index_symbol=symbol)
        start_dateid, end_dateid = 20210101, 20210312
        target = target.set_index('date').loc[dtu.dateid_to_datestr(dateid=start_dateid, sep='-'):dtu.dateid_to_datestr(dateid=end_dateid, sep='-')].reset_index()
        benchmark_file = self.get_benchmark_file(basename=f'index_historical_daily_price.{symbol}.{start_dateid}_{end_dateid}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #
#


if __name__ == '__main__':
    ut.main()
#
