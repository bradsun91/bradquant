from datetime import datetime
import requests

import gmbp_quant.common.env_config as ecfg
import gmbp_quant.common.utils.datetime_utils as dtu
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json

import pandas as pd
from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def request_open_close(symbol: str, dateid: int, adjusted=True):
    key = ecfg.get_env_config().get(ecfg.Prop.POLYGON_KEY)
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{dtu.dateid_to_datestr(dateid=dateid, sep='-')}" \
          f"?unadjusted={not adjusted}&apiKey={key}"
    data = requests.get(url).json()

    status = data['status']
    if status != 'OK':
        logger.error(f'Failed with status={status} on url={url}')
        return None
    #

    data = pd.Series(data).to_frame().T
    return data
#


def request_grouped_daily(dateid, adjusted=True):
    key = ecfg.get_env_config().get(ecfg.Prop.POLYGON_KEY)
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{dtu.dateid_to_datestr(dateid=dateid, sep='-')}" \
          f"?unadjusted={not adjusted}&apiKey={key}"
    data = request_get_as_json(url=url)
    if data is None:
        return None
    #
    daily = pd.DataFrame(data['results'])
    daily.rename(columns={'T': 'Ticker',
                          'o': 'Open',
                          'h': 'High',
                          'l': 'Low',
                          'c': 'Close',
                          'v': 'Volumn',
                          'vw': 'VWAP',
                          't': 'DateTime',
                          'n': 'NumberTransactions'}, inplace=True)
    local_timezone = datetime.now().astimezone().tzinfo
    for col in ['DateTime']:
        daily[col] = pd.to_datetime(daily[col].astype(int), unit='ms', utc=True).dt.tz_convert(local_timezone)
    #
    return daily
#
