import pandas as pd
import gmbp_quant.common.utils.datetime_utils as dtu
import gmbp_quant.common.unittest as ut
from gmbp_quant.dpl.tiingo.tiingo_eod_api import request_historical_daily_prices


class TestTiingoEODAPI(ut.TestCase):
    def test_request_historical_daily_prices(self):
        symbol = 'SPY'
        start_dateid, end_dateid = 20210301, 20210312
        target = request_historical_daily_prices(symbol=symbol, start_dateid=start_dateid, end_dateid=end_dateid)
        benchmark_file = self.get_benchmark_file(basename=f'historical_daily_price.{symbol}.{start_dateid}_{end_dateid}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target[['date','open','high','low','close','volume','divCash','splitFactor']])
    #
#


if __name__ == '__main__':
    ut.main()
#