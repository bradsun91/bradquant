import pandas as pd
import gmbp_quant.common.unittest as ut
from gmbp_quant.dpl.polygon.polygon_stock_api import request_open_close


class TestPolygonStockAPI(ut.TestCase):
    def test_request_open_close(self):
        dateid = 20210302
        symbol = 'AAPL'
        target = request_open_close(symbol='AAPL', dateid=dateid, adjusted=False)
        benchmark_file = self.get_benchmark_file(basename=f'open_close.{symbol}.{dateid}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #
#


if __name__ == '__main__':
    ut.main()
#
