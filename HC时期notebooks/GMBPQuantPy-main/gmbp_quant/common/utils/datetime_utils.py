import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz, re
from gmbp_quant.common.utils.miscs import iterable_to_tuple

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def datetime_to_dateid(date):
    """
    >>> import gmbp_quant.common.utils.datetime_utils as dtu
    >>> dtu.datetime_to_dateid(datetime(2019, 9, 25))
    20190925
    >>> adt_utc = datetime(2020, 8, 7, 2, 0, 0, tzinfo=pytz.UTC)
    >>> adt_ny = adt_utc.astimezone(pytz.timezone('America/New_York'))
    >>> dtu.datetime_to_dateid(adt_utc)
    20200807
    >>> dtu.datetime_to_dateid(adt_ny)
    20200806
    """
    return int(date.strftime('%Y%m%d %Z')[:8])
#


def dateid_to_datetime(dateid, timezone=None):
    """
    >>> import gmbp_quant.common.utils.datetime_utils as dtu
    >>> dtu.dateid_to_datetime(dateid=20190925)
    datetime.datetime(2019, 9, 25, 0, 0)
    """
    dt = datetime.strptime(str(dateid), '%Y%m%d')
    if timezone is not None:
        dt = dt.replace(tzinfo=pytz.timezone(timezone))
    #

    return dt
#


def dateid_to_datestr(dateid, sep='-'):
    """
    >>> import gmbp_quant.common.utils.datetime_utils as dtu
    >>> dtu.dateid_to_datestr(dateid=20190925)
    '2019-09-25'
    >>> dtu.dateid_to_datestr(dateid=20191013, sep='/')
    '2019/10/13'
    """
    date = str(dateid)
    return f'{date[:4]}{sep}{date[4:6]}{sep}{date[6:]}'
#


def today(timezone=None):
    today = datetime.now() if timezone is None else datetime.now(pytz.timezone(timezone))
    return datetime_to_dateid(date=today)
#


def is_weekday(date):
    """
    >>> import gmbp_quant.common.utils.datetime_utils as dtu
    >>> dtu.is_weekday(20200511)
    True
    """
    if isinstance(date, int):
        date = dateid_to_datetime(dateid=date)
    elif not isinstance(date, datetime):
        raise Exception(f'type(date)={type(date)} is not supported [int|datetime] !')
    #
    return date.weekday() not in [5, 6]
#


def get_biz_dateids(start_dateid, end_dateid):
    """
    >>> import gmbp_quant.common.utils.datetime_utils as dtu
    >>> dtu.get_biz_dateids(start_dateid=20200910, end_dateid=20200917)
    [20200910, 20200911, 20200914, 20200915, 20200916, 20200917]
    """
    bdates = pd.bdate_range(dateid_to_datestr(dateid=start_dateid, sep='-'),
                            dateid_to_datestr(dateid=end_dateid, sep='-')).strftime('%Y%m%d')
    return [int(date) for date in bdates]
#


def prev_biz_dateid(dateid):
    one_day = timedelta(days=1)
    next_day = dateid_to_datetime(dateid) - one_day
    while not is_weekday(date=next_day):
        next_day -= one_day
    #
    return datetime_to_dateid(next_day)
#


def next_biz_dateid(dateid):
    one_day = timedelta(days=1)
    next_day = dateid_to_datetime(dateid) + one_day
    while not is_weekday(date=next_day):
        next_day += one_day
    #
    return datetime_to_dateid(next_day)
#


def shift_biz_dates(dateid, offset_days):
    if offset_days is None or not isinstance(offset_days, int):
        logger.warn(f'Skipping since "offset_days" is None or not type(int) !')
        return dateid
    #

    shift_func = prev_biz_dateid if offset_days<=0 else next_biz_dateid
    offset_days = abs(offset_days)

    for i in range(offset_days):
        dateid = shift_func(dateid=dateid)
    #

    return dateid
#


def infer_biz_dateids(start_dateid=None, end_dateid=None, dateids=None):
    if dateids is not None:
        dateids = iterable_to_tuple(dateids, raw_type='int')
        dateids = [dateid for dateid in dateids if not is_weekday(dateid)]
    else:
        if start_dateid is None or end_dateid is None:
            raise ValueError(f'Please provide either "dateids" or ("start_dateid", "end_dateid") !')
        #

        if isinstance(start_dateid, str):
            start_dateid = int(start_dateid)
        elif isinstance(start_dateid, datetime):
            start_dateid = datetime_to_dateid(start_dateid)
        #

        if isinstance(end_dateid, str):
            end_dateid = int(end_dateid)
        elif isinstance(end_dateid, datetime):
            end_dateid = datetime_to_dateid(end_dateid)
        #

        dateids = get_biz_dateids(start_dateid=start_dateid, end_dateid=end_dateid)
    #

    return dateids
#


def infer_start_dateid_end_dateid(start_date=None, end_date=None, date_range_mode='SINGLE_DATE'):
    # if start_date is None and end_date is None:
    #     raise ValueError(f'"start_date" and "end_date" cannot be both None !')
    # #

    ctd = today()
    if end_date in ['CTD', 'TODAY', None]:
        end_date = ctd
    elif end_date in ['PTD', 'YEST']:
        end_date = prev_biz_dateid(dateid=ctd)
    #

    end_date = int(end_date)
    if not is_weekday(end_date):
        end_date = prev_biz_dateid(dateid=end_date)
    #

    if start_date is not None:
        start_date = int(start_date)
        if not is_weekday(start_date):
            start_date = next_biz_dateid(dateid=start_date)
        #
        if start_date <= end_date:
            return start_date, end_date
        else:
            logger.error(f'Inconsistency found: start_date={start_date} > end_date={end_date} ! '
                         f'start_date will be re-inferred by date_range_mode={date_range_mode} instead.')
        #
    #

    ndays_matched = re.match(r'(\d+)D$', date_range_mode, re.IGNORECASE)

    if date_range_mode == 'SINGLE_DATE':
        start_date = end_date
    elif date_range_mode == 'ROLLING_WEEK':
        start_date = shift_biz_dates(dateid=end_date, offset_days=-4)
    elif date_range_mode == 'MTD':
        start_date = int(end_date/100.0) + 1
        if not is_weekday(date=start_date):
            start_date = next_biz_dateid(dateid=start_date)
        #
    elif date_range_mode == 'ROLLING_MONTH':
        start_date = dateid_to_datetime(dateid=end_date) - relativedelta(months=1)
        start_date = datetime_to_dateid(date=start_date)
        if not is_weekday(date=start_date):
            start_date = next_biz_dateid(dateid=start_date)
        #
    elif ndays_matched is not None:
        ndays = int(ndays_matched.group()[0])
        start_date = shift_biz_dates(dateid=end_date, offset_days=-(ndays-1))
    else:
        raise ValueError(f'date_range_mode={date_range_mode} is not supported in [SINGLE_DATE|ROLLING_WEEK|MTD|ROLLING_MONTH|*D] !')
    #

    return start_date, end_date
#


def parse_datetime(dt, format=None, timezone=None, ret_as_timestamp=False):
    if isinstance(dt, int):
        dt = str(dt)
    #
    if (isinstance(dt, datetime) and not ret_as_timestamp) or \
            (isinstance(dt, pd.Timestamp) and ret_as_timestamp):
        return dt
    #

    try:
        dt = pd.Timestamp(dt)
        if timezone is not None:
            dt = dt.tz_localize(timezone)
        #
    except Exception:
        pass
    #

    if isinstance(dt, pd.Timestamp):
        return (dt if ret_as_timestamp else dt.to_pydatetime())
    #
    if isinstance(dt, str):
        dt = pd.to_datetime(dt, format=format)
        if timezone is not None:
            dt = dt.tz_localize(timezone)
        #
        return (dt if ret_as_timestamp else dt.to_pydatetime())
    #
    raise Exception(f'type(dt)={type(dt)} is not supported [dateid|str|datetime|pd.Timestamp|np.datetime64] !')
#
