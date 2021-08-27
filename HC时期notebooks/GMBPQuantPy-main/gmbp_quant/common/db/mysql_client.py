from gmbp_quant.common.db.db_client import DBClient


class MySQLClient(DBClient):
    _instances = dict()

    def __init__(self, db_config):
        super().__init__(db_config=db_config, client_type='mysql')
    #

    @staticmethod
    def get_instance(db_config):
        if db_config not in MySQLClient._instances:
            MySQLClient._instances[db_config] = MySQLClient(db_config=db_config)
        #
        return MySQLClient._instances[db_config]
    #
#


def get_mysql_client(db_config):
    return MySQLClient.get_instance(db_config=db_config)
#
