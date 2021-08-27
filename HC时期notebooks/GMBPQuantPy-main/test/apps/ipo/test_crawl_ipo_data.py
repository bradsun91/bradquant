import pandas as pd
import gmbp_quant.common.unittest as ut
from gmbp_quant.apps.dpl.ipo.crawl_ipo_data import crawl_ipo_data_single_date


class TestCrawlIPOData(ut.TestCase):
    def test_crawl_ipo_data_single_date(self):
        dateid = 20210216
        target = crawl_ipo_data_single_date(dateid=dateid)
        benchmark_file = self.get_benchmark_file(basename=f'ipo.{dateid}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #
#


if __name__ == '__main__':
    ut.main()
#
