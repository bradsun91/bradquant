import pytz
import gmbp_quant.common.env_config as ecfg
import gmbp_quant.common.utils.datetime_utils as dtu
from gmbp_quant.common.db.mysql_client import get_mysql_client
import pandas as pd


def query_bpii(start_dt, end_dt, timezone='America/New_York'):
    start_dt = dtu.parse_datetime(dt=start_dt)
    if start_dt.hour == 0 and start_dt.minute == 0 and start_dt.second == 0:
        start_dt = start_dt.replace(second=1)
    #
    end_dt = dtu.parse_datetime(dt=end_dt)
    if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
    #

    query = f'''
        SELECT * FROM quant.bpii
        WHERE Signal_Time BETWEEN '{start_dt.strftime('%Y-%m-%d %H:%M:%S')}' AND '{end_dt.strftime('%Y-%m-%d %H:%M:%S')}'
    '''

    db_config = ecfg.get_env_config().get_db_config(ecfg.DBProp.DB)
    db_client = get_mysql_client(db_config=db_config)
    bpii = pd.read_sql(sql=query, con=db_client.engine, parse_dates=['Signal_Time'])
    bpii['Signal_Time'] = bpii['Signal_Time'].dt.tz_localize(pytz.timezone(timezone))
    return bpii
#
