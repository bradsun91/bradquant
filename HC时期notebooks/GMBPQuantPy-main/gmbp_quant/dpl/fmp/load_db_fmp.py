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

    if form_type == 'inst_holder':
        fmp_loader = FmpDataLoader(data_dir, form_type)

        print(f'Parsing FMP data to tsv files ...')
        fmp_loader.parse_inst_holder_to_tsv()

    elif form_type == 'insider_trading':
        fmp_loader = FmpDataLoader(data_dir, form_type)

        print(f'Parsing FMP data to tsv files ...')
        fmp_loader.parse_insider_trading_to_tsv()

    elif form_type == 'etf_holder':
        fmp_loader = FmpDataLoader(data_dir, form_type)

        print(f'Parsing FMP data to tsv files ...')
        fmp_loader.parse_etf_holder_to_tsv()

    else:
        fmp_loader = FmpDataLoader(data_dir, form_type, report_date)

        print(f'Parsing FMP data to tsv files ...')
        fmp_loader.parse_form_13_to_tsv()

    print('Inserting parsed FMP data to DB ...')
    fmp_loader.insert_db()