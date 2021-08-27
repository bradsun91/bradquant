import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
import pandas as pd


from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)

def request_historical_sector_performance(limit: int = 5):
    key = ecfg.get_env_config().get(ecfg.Prop.FMP_KEY)
    data = request_get_as_json(
        url=f'https://financialmodelingprep.com/api/v3/historical-sectors-performance?limit={limit}&apikey={key}')
    print(data)
    if data:
        return pd.DataFrame(data)
    return None