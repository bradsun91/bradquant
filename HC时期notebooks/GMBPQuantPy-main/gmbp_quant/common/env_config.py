import os, enum
import getpass
from configparser import ConfigParser
from gmbp_quant.common.db.db_client import DBConfig

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


@enum.unique
class Prop(enum.Enum):
    IB_TWS_HOST = 'ib_tws.host'
    IB_TWS_PORT = 'ib_tws.port'
    IB_TWS_CLIENT_ID = 'ib_tws.client_id'

    AV_API_KEY = 'av.api_key'
    TIINGO_TOKEN = 'tiingo.token'

    BPII_URL = 'bpii.url'
    BPII_KEY = 'bpii.key'

    POLYGON_KEY = 'polygon.key'
    FMP_KEY = 'fmp.key'

    SEC_MASTER_V0_SCHEMA = 'sec_master_v0.schema'
#


@enum.unique
class DBProp(enum.Enum):
    DB = 'db'
#

# @enum.unique
# class CredentialProp(enum.Enum):
#     IB_TWS = 'ib_tws'
# #
#
# @enum.unique
# class ConnectionProp(enum.Enum):
#     IB_TWS = 'ib_tws'
#


class EnvConfig:
    _instance = None

    def __init__(self, env, overrides=None, env_config_file=None):
        self._env = env
        self._overrides = overrides
        self._parser = ConfigParser()

        if env_config_file is None:
            env_config_file = os.path.join(os.path.dirname(__file__), 'env_config.cfg')
        #

        self._parser.read(env_config_file)

        if env not in self._parser.sections():
            envs = ','.join(self._parser.sections())
            raise Exception(f'env={env} is unknown in sections: {envs} !')
        #

        if env == 'PROD' and overrides:
            raise Exception('overrides are not supported in env=PROD !')
        #

        # user = getpass.getuser()
        #
        # logger.info(75 * '#')
        # logger.info(f'### Initializing EnvConfig with env={env}, user={user}')
        # if overrides is not None:
        #     for key, val in overrides.items():
        #         logger.info(f'###   [override] {key} => {val}')
        #     #
        # #
        # logger.info(75 * '#')
    #

    @property
    def env(self):
        return self._env
    #

    def _get(self, key, default_val=None):
        if key in self._overrides:
            return self._overrides[key]
        #
        try:
            return self._parser.get(self.env, key)
        except:
            if default_val is not None:
                return default_val
            #
            raise
        #
    #

    def get(self, prop, default_val=None):
        return self._get(key=prop.value, default_val=default_val)
    #

    def get_credential(self, credential_prop):
        user = self._get(f'{credential_prop.value}.user')
        password = self._get(f'{credential_prop.value}.password')
        return (user, password)
    #

    def get_db_config(self, db_prop):
        if isinstance(db_prop, DBProp):
            db_prop = db_prop.value
        #
        return DBConfig(host=self._get(f'{db_prop}.host'),
                        port=self._get(f'{db_prop}.port'),
                        username=self._get(f'{db_prop}.username'),
                        password=self._get(f'{db_prop}.password'))
    #

    @staticmethod
    def get_instance(env=None, overrides=None):
        if EnvConfig._instance is not None:
            if env is not None or overrides:
                raise Exception('Only the first call to EnvConfig.get_instance() can use arguments !')
            #
            return EnvConfig._instance
        #

        if env is None:
            env = 'RESEARCH'
        #
        env = env.upper()

        EnvConfig._instance = EnvConfig(env=env, overrides=overrides)

        return EnvConfig._instance
    #
#


def get_env_config(env=None, **overrides):
    return EnvConfig.get_instance(env=env, overrides=overrides)
#
