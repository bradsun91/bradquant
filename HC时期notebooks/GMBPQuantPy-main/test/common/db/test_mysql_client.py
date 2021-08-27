import pandas as pd
import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.db.mysql_client import get_mysql_client
import gmbp_quant.common.unittest as ut


class TestMySQLClient(ut.TestCase):
    def test_mysql_client_select(self):
        db_config = ecfg.get_env_config().get_db_config(ecfg.DBProp.DB)
        db_client = get_mysql_client(db_config=db_config)
        target = db_client.execute_query(query='SELECT * FROM world.city LIMIT 5', log_exception=True)
        benchmark_file = self.get_benchmark_file(basename=f'world_city.top5.csv')
        benchmark = pd.read_csv(benchmark_file, sep=',')
        self.assertEqual(benchmark, target)
    #
#


if __name__ == '__main__':
    ut.main()
#
