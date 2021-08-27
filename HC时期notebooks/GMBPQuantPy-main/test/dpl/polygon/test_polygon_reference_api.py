import pandas as pd
import gmbp_quant.common.unittest as ut
from gmbp_quant.dpl.polygon.polygon_reference_api import *

class TestPolygonReferenceAPI(ut.TestCase):
    def test_request_tickers(self):
        target = request_tickers(sort_field="ticker", ticker_type="cs", market="stocks", locale="us", perpage=1)
        benchmark_file = self.get_benchmark_file(basename=f'tickers.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #

    def test_request_ticker_types(self):
        target = request_ticker_types()
        benchmark_file = self.get_benchmark_file(basename=f'sec_types.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #
    
    def test_request_ticker_details(self):
        symbol = 'AAPL'
        details, tags, similars = request_ticker_details(symbol=symbol)
        benchmark_file = self.get_benchmark_file(basename=f'ticker_details.{symbol}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',', parse_dates=['listdate','updated'])
        self.assertEqual(benchmark, details)

        benchmark_file = self.get_benchmark_file(basename=f'ticker_details_tags.{symbol}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, tags)

        benchmark_file = self.get_benchmark_file(basename=f'ticker_details_similars.{symbol}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, similars)
    #

    def test_request_markets(self):
        target = request_markets()
        benchmark_file = self.get_benchmark_file(basename=f'markets.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #

    def test_request_locales(self):
        target = request_locales()
        benchmark_file = self.get_benchmark_file(basename=f'locales.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #

    def test_request_stock_splits(self):
        symbol = 'AAPL'
        target = request_stock_splits(symbol=symbol)
        benchmark_file = self.get_benchmark_file(basename=f'stock_splits.AAPL.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',', parse_dates=['exDate','paymentDate','declaredDate'])
        self.assertEqual(benchmark, target)
    #

    def test_request_stock_dividends(self):
        symbol = 'AAPL'
        target = request_stock_dividends(symbol=symbol)
        benchmark_file = self.get_benchmark_file(basename=f'stock_dividends.AAPL.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',', parse_dates=['exDate','paymentDate','recordDate','declaredDate'])
        self.assertEqual(benchmark, target)
    #

    def test_request_stock_financials(self):
        symbol = 'AAPL'
        limit = 5
        report_type = 'Y'
        target = request_stock_financials(symbol=symbol, limit=limit, report_type=report_type)
        benchmark_file = self.get_benchmark_file(basename=f'stock_financials.AAPL.limit=5.report_type=Y.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',', parse_dates=['calendarDate','reportPeriod','updated','dateKey'])
        self.assertEqual(benchmark, target)
    #

    def test_request_stock_exchanges(self):
        target = request_stock_exchanges()
        benchmark_file = self.get_benchmark_file(basename=f'stock_exchanges.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #

    def test_request_crypto_exchanges(self):
        target = request_crypto_exchanges()
        benchmark_file = self.get_benchmark_file(basename=f'crypto_exchanges.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #

    def test_request_tick_conditions(self):
        target = request_tick_conditions()
        benchmark_file = self.get_benchmark_file(basename=f'tick_conditions.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #

    def test_request_tickers(self):
        target = request_tickers(num_pages=3)
        benchmark_file = self.get_benchmark_file(basename=f'tickers.num_pages=3.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #
#

if __name__ == '__main__':
    ut.main()
#
