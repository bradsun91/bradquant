import sys
import argparse
import pandas as pd

from fmp_data_loader import FmpDataLoader
import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./fmp_data', help='FMP form data folder')
parser.add_argument('--form', type=str, help='FMP form type')
parser.add_argument('--report_date', type=str, help='FMP data report_date')
opt = parser.parse_args()


if __name__ == '__main__':
    report_date = opt.report_date
    data_dir = Path(opt.data)
    form_type = opt.form

    key = ecfg.get_env_config().get(ecfg.Prop.FMP_KEY)

    if form_type == 'inst_holder':
        data = request_get_as_json(
            url=f'https://financialmodelingprep.com/api/v3/stock/list?apikey={key}')

        table = pd.DataFrame(data)

        fmp_loader = FmpDataLoader(data_dir, form_type)
        for ticker in table.loc[:, 'symbol']:
            print(f'Downloading FMP data {ticker} to json file ...')
            fmp_loader.request_inst_holder_data(ticker)
        #
    elif form_type == 'insider_trading':
        data = request_get_as_json(
            url=f'https://financialmodelingprep.com/api/v3/stock/list?apikey={key}')

        table = pd.DataFrame(data)

        fmp_loader = FmpDataLoader(data_dir, form_type)
        for ticker in table.loc[:, 'symbol']:
            print(f'Downloading insider_trading {ticker} to json file ...')
            fmp_loader.request_insider_trading_data(ticker)
        #
    elif form_type == 'etf_holder':
        data = request_get_as_json(
            url=f'https://financialmodelingprep.com/api/v3/etf/list?apikey={key}')

        table = pd.DataFrame(data)

        fmp_loader = FmpDataLoader(data_dir, form_type)
        for holder in table.loc[:, 'symbol']:
            print(f'Downloading FMP data {holder} to json file ...')
            fmp_loader.request_etf_holder_data(holder)
        #
    else:
        data = request_get_as_json(
            url=f'https://financialmodelingprep.com/api/v3/cik_list?apikey={key}')

        table = pd.DataFrame(data)

        fmp_loader = FmpDataLoader(data_dir, form_type, report_date)
        for cik in table.loc[:, 'cik']:
            print(f'Downloading FMP data {cik} to json file ...')
            fmp_loader.request_form_13_data(cik, report_date)
        #
    #
#