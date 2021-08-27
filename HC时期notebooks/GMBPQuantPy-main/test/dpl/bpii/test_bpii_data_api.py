import pandas as pd
import gmbp_quant.common.unittest as ut
from gmbp_quant.dpl.bpii.bpii_data_api import create_signature, request_bpii_ts, calc_bpii_signals


class TestBPIIDataAPI(ut.TestCase):
    # def test_create_signature(self):
    #     current_time = '1614301999592'
    #     signature = create_signature(current_time=current_time)
    #     self.assertEqual(signature, b'Al//+2Xe9t09VLGbZ0Mu/JWOCS7gp9yMK+GID4Wt3QI=')
    # #
    def test_calc_bpii_signals(self):
        data = pd.DataFrame({'large': [1.33, -1.1, -2.66], 'average': [1.24, -0.83, -2.13]})
        signals = calc_bpii_signals(data, large_col='large', average_col='average')
        benchmark_file = self.get_benchmark_file(basename=f'bpii_signals.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, signals)
    #

    def test_request_bpii_ts(self):
        dateid = 20210301
        target = request_bpii_ts(dateid=dateid)
        benchmark_file = self.get_benchmark_file(basename=f'bpii_ts.{dateid}.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #
#


if __name__ == '__main__':
    ut.main()
#
