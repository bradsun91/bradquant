import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
import pandas as pd

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def request_historical_daily_prices(index_symbol):
    key = ecfg.get_env_config().get(ecfg.Prop.FMP_KEY)
    data = request_get_as_json(url=f'https://financialmodelingprep.com/api/v3/historical-price-full/{index_symbol}?apikey={key}')
    if data is None:
        return None
    #
    daily = pd.DataFrame(data['historical'])
    daily.drop(columns=['label', 'changeOverTime'], inplace=True)
    for col in ['date']:
        daily[col] = pd.to_datetime(daily[col])
    #

    return daily
#
