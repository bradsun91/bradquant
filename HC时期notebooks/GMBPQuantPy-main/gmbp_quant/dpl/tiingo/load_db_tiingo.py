import sys
import argparse
import pandas as pd

from tiingo_data_loader import TiingoDataLoader
import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./tingo_data', help='Tiingo data folder')
parser.add_argument('--form', type=str, help='Tiingo form type')
opt = parser.parse_args()


if __name__ == '__main__':
    data_dir = Path(opt.data)
    form_type = opt.form

    if form_type == 'daily_price':
        tiingo_loader = TiingoDataLoader(data_dir, form_type)

        print(f'Parsing Tiingo data to tsv files ...')
        tiingo_loader.parse_daily_price_to_tsv()

    print('Inserting parsed Tiingo data to DB ...')
    tiingo_loader.insert_db()