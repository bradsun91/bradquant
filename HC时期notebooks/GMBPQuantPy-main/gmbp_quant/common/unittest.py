import unittest
import os
import inspect
import pandas as pd


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        def dataframe_equals(df1, df2, msg=None):
            return df1.equals(df2)
        #

        def index_equals(idx1, idx2, msg=None):
            return idx1.equals(idx2)
        #

        self.addTypeEqualityFunc(pd.DataFrame, dataframe_equals)
        self.addTypeEqualityFunc(pd.Index, index_equals)
    #

    def get_benchmark_file(self, basename):
        frame = inspect.stack()[1]
        caller_file = os.path.realpath(frame[0].f_code.co_filename)
        benchmark_file = os.path.join(os.path.dirname(caller_file), 'benchmark', basename)
        return benchmark_file
    #

    # def load_benchmark_dataframe(self, basename):
    #     benchmark_file = self.get_benchmark_file(basename=basename)
    #     benchmark = pd.read_csv(benchmark_file, sep=',')
    #     return benchmark
    # #
#


def main():
    unittest.main()
#
