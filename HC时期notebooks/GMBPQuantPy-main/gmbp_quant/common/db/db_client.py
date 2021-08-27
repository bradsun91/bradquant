import traceback
from sqlalchemy import create_engine, schema
from sqlalchemy.sql import text as SQLStmt
from sqlalchemy.orm import close_all_sessions
from sqlalchemy.exc import ProgrammingError
import pandas as pd

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


class DBConfig:
    @property
    def host(self):
        return self._host
    #

    @host.setter
    def host(self, host):
        self._host = host
    #

    @property
    def port(self):
        return self._port

    #
    @port.setter
    def port(self, port):
        self._port = int(port)
    #

    @property
    def username(self):
        return self._username
    #

    @username.setter
    def username(self, username):
        self._username = username
    #

    @property
    def password(self):
        return self._password
    #

    @password.setter
    def password(self, password):
        self._password = password
    #

    def __init__(self, host, port, username, password):
        # self._host = host
        # self._port = port
        # self._username = username
        # self._password = password
        self.host = host
        self.port = port
        self.username = username
        self.password = password
    #

    def __repr__(self):
        return f'{self.username}:{self.password}@{self.host}:{self.port}'
    #
#


class DBClient:
    @property
    def engine(self):
        return self._engine
    #

    @engine.setter
    def engine(self, engine):
        self._engine = engine
    #

    DIALECT_MAP = {'MYSQL': 'mysql+pymysql'}

    def __init__(self, db_config, client_type, schema=None):
        # engine_str = f'{client_type}://{db_config.username}:{db_config.password}@{db_config.host}:{db_config.port}'

        engine_str = f"{DBClient.DIALECT_MAP.get(client_type.upper(), 'mysql+pymysql')}://{db_config}"
        if schema is not None:
            engine_str = f'{engine_str}/{schema}'
        #
        try:
            self.engine = create_engine(engine_str)
        except Exception as e:
            logger.error(f'Failed to create engine for {engine_str} !')
            self.engine = None
            traceback.print_exc()
        #
    #

    # def __del__(self):
    #     close_all_sessions()
    #     if self.engine is not None:
    #         self.engine.dispose()
    #     #
    # #

    def execute_sql_stmt(self, sql_stmt, log_exception=False):
        try:
            with self.engine.connect() as connection:
                cursor = connection.execute(SQLStmt(sql_stmt).execution_options(autocommit=True))
            #
        except Exception as e:
            if log_exception:
                logger.error(f'Exception caught ！')
                traceback.print_exc()
            #
            cursor = None
        #
        return cursor
    #

    def execute_query(self, query, ret_as_df=True, log_exception=False):
        try:
            with self.engine.connect() as connection:
                cursor = connection.execute(SQLStmt(query).execution_options(autocommit=True))
                ret = cursor.fetchall()
                if ret is not None and ret_as_df:
                    cols = ret[0].keys()
                    ret = pd.DataFrame(ret)
                    ret.columns = cols
                #
            #
        except Exception as e:
            if log_exception:
                logger.error(f'Exception caught ！')
                traceback.print_exc()
            #
            ret = None
        #
        return ret
    #

    def create_schema_if_not_exists(self, schema_name):
        try:
            with self.engine.connect() as connection:
                if schema_name not in connection.dialect.get_schema_names(connection):
                    connection.execute(schema.CreateSchema(schema_name))
                    logger.info(f'Schema "{schema_name}" created')
                #
        except ProgrammingError:
            pass
        #
    #
#
