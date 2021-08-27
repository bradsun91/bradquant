import sys
import argparse
import pandas as pd

from yahoo_data_loader  import YahooDataLoader
import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./yahoo_data', help='Yahoo data folder')
parser.add_argument('--form', type=str, help='Tiingo form type')
parser.add_argument('--ipo_date', type=str, help='IPO date')
opt = parser.parse_args()


if __name__ == '__main__':
    ipo_date = opt.ipo_date
    data_dir = Path(opt.data)
    form_type = opt.form

    yahoo_loader = YahooDataLoader(data_dir, form_type)
    if form_type == 'ipo':
        print(f'Downloading yahoo data {ipo_date} to raw file ...')
        yahoo_loader.request_ipo_data(ipo_date)
    #
#