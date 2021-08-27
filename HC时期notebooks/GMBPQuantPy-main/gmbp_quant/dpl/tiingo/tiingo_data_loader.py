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
from gmbp_quant.dpl.tiingo.tiingo_schemas import TiingoDailyPriceSchema


logger = LOG.get_logger(__name__)


class TiingoDataLoader(DataLoader):
    """A :class:`TiingoDataLoader` class

    Args:
        data_dir: folder storing Tiingo data
        form_type: Tiingo data type, currently support
                    'daily_price'
    """

    def __init__(self, data_dir, form_type):
        self.form_dir = data_dir / form_type
        self.tsv_dir = data_dir / 'tsv_files' / form_type
        self.form_list = list(self.form_dir.glob('*.txt'))
        schemas = {
            'daily_price': TiingoDailyPriceSchema
        }
        super().__init__(data_dir, form_type, self.tsv_dir, 'tiingodata', schemas[form_type])

        self.form_dir.mkdir(parents=True, exist_ok=True)

    def request_daily_price_data(self, symbol, start_dateid=None, end_dateid=None):
        start_dateid, end_dateid = infer_start_dateid_end_dateid(start_date=start_dateid, end_date=end_dateid)
        token = ecfg.get_env_config().get(ecfg.Prop.TIINGO_TOKEN)
        data = request_get_as_json(
            url=f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?token={token}&"
                f"startDate={dateid_to_datestr(start_dateid, sep='-')}&"
                f"endDate={dateid_to_datestr(end_dateid, sep='-')}")

        if data is None:
            return None
        #

        data_file = self.form_dir / f'{symbol}-json.txt'
        with open(data_file, 'w') as data_file_output:
            json.dump(data, data_file_output)

        return data
    #

    def parse_daily_price_to_tsv(self):
        """Parse Tiingo Daily Price and save to tsv files
        """

        for form in tqdm(self.form_list):
            file_id = form.stem.split('-')[0]
            with open(form, 'r') as json_file:
                data = json.load(json_file)
                table = pd.DataFrame(data)

                if not table.empty:
                    table_sorted = table
                    table_sorted['symbol'] = file_id
                    table_sorted['report_date'] = pd.to_datetime(table_sorted['date']).dt.tz_localize(None)
                    table_sorted = table_sorted.drop(columns=['date'])

                    output_path = self.tsv_dir / f'{file_id}.tsv'
                    table_sorted.to_csv(output_path, sep='\t', index=False)
            #
        #
    #

