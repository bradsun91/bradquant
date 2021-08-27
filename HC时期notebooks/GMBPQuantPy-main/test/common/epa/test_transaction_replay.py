import gmbp_quant.common.unittest as ut
import gmbp_quant.common.epa.transaction_replay as tr
import pandas as pd
from pandas._testing import assert_frame_equal


class TestTransactionReplay(ut.TestCase):
    # TODO(Weimin): add more test cases for options, futures, and more complex cases
    def test_transaction_replay1(self):
        transaction_df = pd.read_csv(self.get_benchmark_file("transaction_1.csv"))
        snapshot_df, portfolio_df = tr.transaction_replay(transaction_df, initial_portfolio={"TSLA": 1, "AAPL": 1})
        assert_frame_equal(snapshot_df, pd.read_csv(self.get_benchmark_file("portfolio_daily_snapshot_1.csv")))
        assert_frame_equal(portfolio_df, pd.read_csv(self.get_benchmark_file("portfolio_daily_summary_1.csv")))

    def test_transaction_replay2(self):
        # same condition with test case 1, but with different initial_portfolio
        transaction_df = pd.read_csv(self.get_benchmark_file("transaction_1.csv"))
        snapshot_df, portfolio_df = tr.transaction_replay(transaction_df, initial_portfolio={})
        assert_frame_equal(snapshot_df, pd.read_csv(self.get_benchmark_file("portfolio_daily_snapshot_2.csv")))
        assert_frame_equal(portfolio_df, pd.read_csv(self.get_benchmark_file("portfolio_daily_summary_2.csv")))


if __name__ == "__main__":
    ut.main()
