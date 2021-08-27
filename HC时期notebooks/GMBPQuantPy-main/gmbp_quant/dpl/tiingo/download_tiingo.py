import sys
import argparse
import pandas as pd

from tiingo_data_loader  import TiingoDataLoader
import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./tiingo_data', help='Tiingo form data folder')
parser.add_argument('--form', type=str, help='Tiingo form type')
parser.add_argument('--start_date', type=str, help='Start date')
parser.add_argument('--end_date', type=str, help='End date')
opt = parser.parse_args()


if __name__ == '__main__':
    start_date = opt.start_date
    end_date = opt.end_date
    data_dir = Path(opt.data)
    form_type = opt.form

    # We use FMP stocklist, however, other source can be used, too.
    fmp_key = ecfg.get_env_config().get(ecfg.Prop.FMP_KEY)
    tiingo_key = ecfg.get_env_config().get(ecfg.Prop.TIINGO_TOKEN)

    if form_type == 'daily_price':
        fmp_stock_list = request_get_as_json(
            url=f'https://financialmodelingprep.com/api/v3/stock/list?apikey={fmp_key}')

        fmp_stock_table = pd.DataFrame(fmp_stock_list)

        tiingo_loader = TiingoDataLoader(data_dir, form_type)
        for symbol in fmp_stock_table.loc[:, 'symbol']:
            print(f'Downloading Tiingo data {symbol} to json file ...')
            tiingo_loader.request_daily_price_data(symbol, start_date, end_date)
        #
    #
#