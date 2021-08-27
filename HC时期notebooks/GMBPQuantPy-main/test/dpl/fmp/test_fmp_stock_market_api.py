import pandas as pd
import gmbp_quant.common.utils.datetime_utils as dtu
import gmbp_quant.common.unittest as ut
from gmbp_quant.dpl.fmp.fmp_stock_market_api import request_historical_sector_performance


class TestFMPStockMarketAPI(ut.TestCase):
    def test_request_historical_sector_performance(self):
        target = request_historical_sector_performance()
        benchmark_file = self.get_benchmark_file(basename=f'request_historical_sector_performance.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark.head(), target.head())
    #
#


if __name__ == '__main__':
    ut.main()
#