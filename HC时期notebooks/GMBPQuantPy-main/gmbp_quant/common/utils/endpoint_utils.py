import pandas as pd
import json
import pickle
import requests
from requests_ntlm import HttpNtlmAuth
import gmbp_quant.common.env_config as ecfg

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def get_request(url, is_post, body=None, ret_as_df=True, protocol='json',
                credentail_prop=None):
    with requests.Session() as session:
        (username, password) = ecfg.get_env_config().get_credential(credential_prop=credentail_prop)
        session.auth = HttpNtlmAuth(username=username, password=password, session=session)

        if is_post:
            response = session.post(url, json=body, timeout=300)
        else:
            response = session.get(url, timeout=300)
        #

        if response.ok:
            if protocol == 'json':
                data = json.loads(response.content.decode('utf-8'))
            elif protocol == 'pkl':
                data = pickle.loads(response.content)
            #
        else:
            logger.error(f'Error in response {response.status_code} from the request: {url} with {body}')
            response.raise_for_status()
            data = None
        #

        if protocol == 'json' and ret_as_df:
            try:
                data = pd.DataFrame(data['Data'])
            except Exception:
                data = pd.DataFrame(data)
            #
        #
    #

    return data
#


def request_get_as_json(url, params=None, session=None):
    response = requests.get(url, params=params) if session is None else session.get(url, params=params)
    if response.status_code != 200:
        logger.error(f'Response status_code={response.status_code} on {url}')
        return None
    #
    return response.json()
#
