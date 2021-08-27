import gmbp_quant.common.env_config as ecfg
import gmbp_quant.common.utils.datetime_utils as dtu
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
import pandas as pd

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def request_historical_daily_prices(symbol, start_dateid=None, end_dateid=None):
    start_dateid, end_dateid = dtu.infer_start_dateid_end_dateid(start_date=start_dateid, end_date=end_dateid)
    token = ecfg.get_env_config().get(ecfg.Prop.TIINGO_TOKEN)
    data = request_get_as_json(url=f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?token={token}&"
                                   f"startDate={dtu.dateid_to_datestr(start_dateid, sep='-')}&endDate={dtu.dateid_to_datestr(end_dateid, sep='-')}")
    if data is None:
        return None
    #
    daily = pd.DataFrame(data)
    for col in ['date']:
        daily[col] = pd.to_datetime(daily[col])
    #

    return daily
#