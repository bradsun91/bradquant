import re
import os
import io
import sys
import json as json
import pandas as pd
import lxml.etree as et
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from sqlalchemy.orm import sessionmaker

sys.path.append('..')
from gmbp_quant.common.logger import LOG
import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.db.db_client import DBClient
from gmbp_quant.common.utils.datetime_utils import dateid_to_datestr
from gmbp_quant.common.utils.datetime_utils import infer_start_dateid_end_dateid
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
from gmbp_quant.dpl.base.data_loader import DataLoader
from gmbp_quant.dpl.yahoo.yahoo_schemas import YahooIpoSchema


logger = LOG.get_logger(__name__)


class YahooDataLoader(DataLoader):
    """A :class:`YahooDataLoader` class

    Args:
        data_dir: folder storing Yahoo data
        form_type: Yahoo data type, currently support
                    'ipo'
    """

    def __init__(self, data_dir, form_type):
        self.form_dir = data_dir / form_type
        self.tsv_dir = data_dir / 'tsv_files' / form_type
        self.form_list = list(self.form_dir.glob('*.txt'))
        schemas = {
            'ipo': YahooIpoSchema
        }
        super().__init__(data_dir, form_type, self.tsv_dir, 'yahoodata', schemas[form_type])

        self.form_dir.mkdir(parents=True, exist_ok=True)

    def request_ipo_data(self, dateid=None):
        datestr = dateid_to_datestr(dateid=dateid)
        url = f'https://finance.yahoo.com/calendar/ipo?day={datestr}'

        print (url)
        ipo_data_raw = pd.read_html(url)

        if len(ipo_data_raw) == 0:
            logger.info(f'No IPO data found for date={dateid}')
            return None
        if len(ipo_data_raw) > 2:
            logger.warn(f'len(ipo_data)={len(ipo_data_raw)} is not expected !')
        #

        data_file = self.form_dir / f'{datestr}-tsv.txt'
        pd.concat(ipo_data_raw).to_csv(data_file, sep='\t', index=False)

        return ipo_data_raw
    #

    def parse_ipo_to_tsv(self):
        """Parse Yahoo IPO data and save to tsv files
        """

        print(f'File count {len(self.form_list)}')
        for form in tqdm(self.form_list):
            file_id = form.stem.split('-')[0]
            print(f'Processing {form}...')
            table = pd.read_csv(form, sep='\t')

            table_sorted = table
            table_sorted['PriceRange'] = table_sorted['Price Range']
            table_sorted['IpoDate'] = pd.to_datetime(table_sorted['Date']).dt.tz_localize(None)
            table_sorted = table_sorted.drop(columns=['Price Range'])
            table_sorted = table_sorted.drop(columns=['Date'])

            output_path = self.tsv_dir / f'{file_id}.tsv'
            table_sorted.to_csv(output_path, sep='\t', index=False)
            #
        #
    #

