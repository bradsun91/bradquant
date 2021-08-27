import sys
import argparse
import pandas as pd

from yahoo_data_loader import YahooDataLoader
import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./yahoo_data', help='Yahoo data folder')
parser.add_argument('--form', type=str, help='Yahoo form type')
opt = parser.parse_args()


if __name__ == '__main__':
    data_dir = Path(opt.data)
    form_type = opt.form

    if form_type == 'ipo':
        yahoo_loader = YahooDataLoader(data_dir, form_type)

        print(f'Parsing Yahoo IPO data to tsv files ...')
        yahoo_loader.parse_ipo_to_tsv()

    print('Inserting parsed Yahoo IPO data to DB ...')
    yahoo_loader.insert_db()