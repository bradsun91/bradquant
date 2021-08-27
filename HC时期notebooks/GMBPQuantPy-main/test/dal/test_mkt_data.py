import pandas as pd
import gmbp_quant.dal.mkt_data as dalmd
from gmbp_quant.common.utils.miscs import iterable_to_tuple
import gmbp_quant.common.unittest as ut


class TestMktData(ut.TestCase):
    def test_query_security_lookup(self):
        target = dalmd.query_security_lookup(tickers='AAPL,MSFT')
        benchmark_file = self.get_benchmark_file(basename=f'security_lookup.AAPL_MSFT.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)

        cols = 'ID,TICKER,REGION'
        target = dalmd.query_security_lookup(tickers='AAPL,MSFT', cols=cols)
        self.assertEqual(benchmark[list(iterable_to_tuple(cols, raw_type='str'))], target)
    #

    def test_query_security_day_price(self):
        # dateid = 20210105
        # target = dalsd.query_security_day_price(end_dateid=dateid)
        # benchmark_file = self.get_benchmark_file(basename=f'security_day_price.20210105.csv')
        # benchmark = pd.read_csv(benchmark_file, sep=',')
        # self.assertEqual(benchmark, target)

        tickers = 'TSLA'
        start_dateid, end_dateid = 20101001, 20101231
        target = dalmd.query_security_day_price(tickers=tickers, start_dateid=start_dateid, end_dateid=end_dateid)
        benchmark_file = self.get_benchmark_file(basename=f'security_day_price.{tickers}.{start_dateid}.{end_dateid}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)

        cols = 'SECURITY_LOOKUP_ID,TICKER,time_x,ADJ_CLOSE'
        target = dalmd.query_security_day_price(tickers=tickers, start_dateid=start_dateid, end_dateid=end_dateid,
                                                cols=cols)
        self.assertEqual(benchmark[list(iterable_to_tuple(cols, raw_type='str'))], target)
    #
#


if __name__ == '__main__':
    ut.main()
#
