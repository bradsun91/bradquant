import pandas as pd
import gmbp_quant.dal.bpii as dalbpii
import gmbp_quant.common.unittest as ut


class TestBPII(ut.TestCase):
    def test_query_bpii(self):
        target = dalbpii.query_bpii(start_dt=20210226, end_dt=20210226)
        benchmark_file = self.get_benchmark_file(basename=f'bpii.20210226.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #
#


if __name__ == '__main__':
    ut.main()
#
