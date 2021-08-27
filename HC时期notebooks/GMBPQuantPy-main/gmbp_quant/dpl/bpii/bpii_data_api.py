import time, hashlib, base64, requests, pytz
from datetime import datetime
import gmbp_quant.common.env_config as ecfg
import gmbp_quant.common.utils.datetime_utils as dtu
import numpy as np
import pandas as pd

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def create_signature(bpii_key=None, current_time=None):
    if bpii_key is None:
        bpii_key = ecfg.get_env_config().get(ecfg.Prop.BPII_KEY)
    #
    if current_time is None:
        current_time = str(int(time.time() * 1000))
    #
    sha256 = hashlib.sha256()
    sha256.update((current_time + bpii_key).encode('utf-8'))
    sha = sha256.digest()
    signature = base64.b64encode(sha)
    return signature
#


def calc_bpii_signals(data, large_col='large', average_col='average'):
    if large_col not in data.columns:
        logger.error(f'Skipping since no large_col={large_col} in data.columns={data.columns} !')
        return data
    if average_col not in data.columns:
        logger.error(f'Skipping since no average_col={average_col} in data.columns={data.columns} !')
        return data
    #
    data['LmA'] = data[large_col] - data[average_col]
    data['PHA'] = data['LmA'] / data[average_col].abs()
    data.replace(-np.inf, -1e5, inplace=True)
    data.replace(np.inf, 1e5, inplace=True)
    data['BPIISignal'] = pd.cut(data['PHA'],
                                bins=[-np.inf, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, np.inf],
                                labels=[-4, -3, -2, -1, 1, 2, 3, 4])
    return data
#


def _process_request_return(data, append_bpii_signals=False):
    is_success = data['success']
    ret_code = data['code']
    if not is_success:
        logger.error(f'Failed with ret_code={ret_code}')
        return None
    #

    # print(f"timestamp={dpl['timestamp']} -> {datetime.fromtimestamp(int(dpl['timestamp']) / 1000.0, tz=datetime.now().astimezone().tzinfo)}")
    # print(f"timeX={dpl['result']['timeX']} -> {datetime.fromtimestamp(int(dpl['result']['timeX']) / 1000.0, tz=datetime.now().astimezone().tzinfo)}")

    result = data['result']
    if isinstance(result, dict):
        result = [result]
    #
    df = pd.DataFrame(result)

    cols = [col for col in ['average', 'small', 'middle', 'large', 'sp500'] if col in df.columns]
    for col in cols:
        df[col] = df[col].astype(float)
    #
    df['RespondDateTime'] = data['timestamp']
    df['SignalDateTime'] = df['timeX']
    local_timezone = datetime.now().astimezone().tzinfo
    for col in ['SignalDateTime', 'RespondDateTime']:
        df[col] = pd.to_datetime(df[col].astype(int), unit='ms', utc=True).dt.tz_convert(local_timezone)
    #

    if append_bpii_signals:
        df = calc_bpii_signals(df, large_col='large', average_col='average')
    #

    cols = ['SignalDateTime', 'average', 'small', 'middle', 'large', 'sp500', 'LmA', 'PHA', 'BPIISignal',
            'timeX', 'RespondDateTime']
    cols = [col for col in cols if col in df.columns]
    return df[cols]
#


def request_bpii_realtime(append_bpii_signals=False):
    bpii_url = ecfg.get_env_config().get(ecfg.Prop.BPII_URL)
    url = f'{bpii_url}/bpiiRealtime'

    current_time = str(int(time.time() * 1000))
    signature = create_signature(current_time=current_time)

    headers = {
        'Authorization': signature,
        'timeStamp': current_time
    }

    logger.info(f'Requesting {url} with headers={headers}')
    data = requests.get(url, headers=headers).json()
    bpii_realtime = _process_request_return(data=data, append_bpii_signals=append_bpii_signals)
    return bpii_realtime
#


def request_bpii_ts(dateid=None, start_dt=None, end_dt=None, append_bpii_signals=False):
    if start_dt is None or end_dt is None:
        if dateid is None:
            raise Exception(f'Please provide either "dateid" or ("start_dt", "end_dt") !')
        #
        start_dt = f'{str(dateid)}-000001'
        end_dt = f'{str(dateid)}-235959'
    #

    bpii_url = ecfg.get_env_config().get(ecfg.Prop.BPII_URL)
    url = f'{bpii_url}/bpiiTimeSeries'

    current_time = str(int(time.time() * 1000))
    signature = create_signature(current_time=current_time)

    headers = {
        'Authorization': signature,
        'timeStamp': current_time
    }
    params = {
        'beginTime': dtu.parse_datetime(dt=start_dt).strftime('%Y-%m-%d %H:%M:%S'),
        'endTime': dtu.parse_datetime(dt=end_dt).strftime('%Y-%m-%d %H:%M:%S'),
    }

    logger.info(f'Requesting {url} with headers={headers}, params={params}')
    data = requests.get(url, headers=headers, params=params).json()
    bpii_ts = _process_request_return(data=data, append_bpii_signals=append_bpii_signals)
    return bpii_ts
#
