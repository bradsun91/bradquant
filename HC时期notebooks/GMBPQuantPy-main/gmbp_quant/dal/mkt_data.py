import traceback
import pandas as pd
import gmbp_quant.common.env_config as ecfg
import gmbp_quant.common.utils.datetime_utils as dtu
from gmbp_quant.common.utils.miscs import iterable_to_db_str, iterable_to_tuple
from gmbp_quant.common.db.mysql_client import get_mysql_client

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def query_security_lookup(tickers=None, cols=None):
    schema = ecfg.get_env_config().get(ecfg.Prop.SEC_MASTER_V0_SCHEMA, None)
    query = f"""
        SELECT * FROM {schema}.security_lookup
    """
    if tickers is not None:
        tickers_db_str = iterable_to_db_str(tickers, raw_type='str')
        query = f'{query} WHERE Ticker IN {tickers_db_str}'
    #

    try:
        db_config = ecfg.get_env_config().get_db_config(ecfg.DBProp.DB)
        db_client = get_mysql_client(db_config=db_config)
        if cols is not None:
            cols = list(iterable_to_tuple(cols, raw_type='str'))
        #
        security_lookup = pd.read_sql(sql=query, con=db_client.engine, columns=cols)
    except Exception as e:
        traceback.print_exc()
        security_lookup = None
    #
    return security_lookup
#


def query_security_day_price(tickers=None, start_dateid=None, end_dateid=None, cols=None):
    where_clauses = list()
    if tickers is None:
        if end_dateid is None:
            end_dateid = dtu.today()
            logger.warn(f'"tickers" and "end_dateid" found both None! Set "end_dateid" to be CTD={end_dateid}')
        #
    else:
        sec_id_ticker = query_security_lookup(tickers=tickers, cols='ID,TICKER')
        where_clauses.append(f"SECURITY_LOOKUP_ID IN {iterable_to_db_str(sec_id_ticker['ID'], raw_type='int')}")
    #

    if start_dateid is not None:
        if end_dateid is None:
            end_dateid = start_dateid
            logger.warn(f'"start_dateid"={start_dateid} but "end_dateid" is None! Set "end_dateid" to be "start_dateid"')
        #
    else:
        if end_dateid is not None:
            start_dateid, end_dateid = dtu.infer_start_dateid_end_dateid(start_date=start_dateid, end_date=end_dateid,
                                                                         date_range_mode='SINGLE_DATE')
        #
    #

    if start_dateid is not None and end_dateid is not None:
        where_clauses.append(f"time_X BETWEEN '{dtu.dateid_to_datestr(start_dateid)}' AND '{dtu.dateid_to_datestr(end_dateid)}'")
    #

    schema = ecfg.get_env_config().get(ecfg.Prop.SEC_MASTER_V0_SCHEMA, None)
    query = f"""
            SELECT * FROM {schema}.security_day_price
            WHERE {' AND '.join(where_clauses)}
        """
    logger.debug(query)

    try:
        db_config = ecfg.get_env_config().get_db_config(ecfg.DBProp.DB)
        db_client = get_mysql_client(db_config=db_config)
        if cols is not None:
            cols = list(iterable_to_tuple(cols, raw_type='str'))
        #
        security_day_price = pd.read_sql(sql=query, con=db_client.engine, columns=cols)
    except Exception as e:
        traceback.print_exc()
        security_day_price = None
    #
    return security_day_price
#


# def query_security_day_price(tickers=None, start_dateid=None, end_dateid=None, cols=None,
#                              data_source='V0'):
#     if data_source == 'V0':
#         return query_security_day_price(tickers=tickers, start_dateid=start_dateid, end_dateid=end_dateid, cols=cols)
#     #
#
#     raise ValueError(f'"data_source"={data_source} is not supported in [V0] !')
# #
