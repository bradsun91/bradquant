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
from gmbp_quant.common.utils.endpoint_utils import request_get_as_json
from gmbp_quant.dpl.base.data_loader import DataLoader
from gmbp_quant.dpl.fmp.fmp_schemas import Fmp13FSchema, Fmp4Schema
from gmbp_quant.dpl.fmp.fmp_schemas import FmpInstHolderSchema, FmpEtfHolderSchema, FmpInsiderTradingSchema


logger = LOG.get_logger(__name__)


class FmpDataLoader(DataLoader):
    """A :class:`FMPDataLoader` class

    Args:
        data_dir: folder storing FMP data
        form_type: FMP form type, currently support
                    '13F'
                    '4'
                    'inst_holder'
                    'eft_holder'
        report_date: optional
    """

    def __init__(self, data_dir, form_type, report_date=None):
        if report_date is None:
            self.form_dir = data_dir / form_type
            self.tsv_dir = data_dir / 'tsv_files' / form_type
        else:
            self.form_dir = data_dir / form_type / report_date
            self.tsv_dir = data_dir / 'tsv_files' / form_type / report_date
        self.form_list = list(self.form_dir.glob('*.txt'))
        self.report_date = report_date
        schemas = {
            '13F': Fmp13FSchema,
            '4': Fmp4Schema,
            'inst_holder': FmpInstHolderSchema,
            'insider_trading': FmpInsiderTradingSchema,
            'etf_holder': FmpEtfHolderSchema
        }
        super().__init__(data_dir, form_type, self.tsv_dir, 'fmpdata', schemas[form_type])

        self.form_dir.mkdir(parents=True, exist_ok=True)

    def request_form_13_data(self, cik, report_date):
        key = ecfg.get_env_config().get(ecfg.Prop.FMP_KEY)
        data = request_get_as_json(
            url=f'https://financialmodelingprep.com/api/v3/form-thirteen/{cik}?date={report_date}&apikey={key}')
        if data is None:
            return None
        #

        data_file = self.form_dir / f'{cik}-json.txt'
        with open(data_file, 'w') as data_file_output:
            json.dump(data, data_file_output)

        return data
    #

    def request_inst_holder_data(self, ticker):
        key = ecfg.get_env_config().get(ecfg.Prop.FMP_KEY)

        # Free version only allows ticker=AAPL
        url = f'https://financialmodelingprep.com/api/v3/institutional-holder/{ticker}?apikey={key}'
        data = request_get_as_json(url)
        if data is None:
            return None
        #

        data_file = self.form_dir / f'{ticker}-json.txt'
        with open(data_file, 'w') as data_file_output:
            json.dump(data, data_file_output)

        return data
    #

    def request_etf_holder_data(self, ticker):
        key = ecfg.get_env_config().get(ecfg.Prop.FMP_KEY)
        url =f'https://financialmodelingprep.com/api/v3/etf-holder/{ticker}?apikey={key}'
        data = request_get_as_json(url)
        if data is None:
            return None
        #

        data_file = self.form_dir / f'{ticker}-json.txt'
        with open(data_file, 'w') as data_file_output:
            json.dump(data, data_file_output)

        return data
    #

    def request_insider_trading_data(self, ticker):
        key = ecfg.get_env_config().get(ecfg.Prop.FMP_KEY)

        # Free version only allows ticker=AAPL
        url =f'https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&apikey={key}'
        data = request_get_as_json(url)
        if data is None:
            return None
        #

        data_file = self.form_dir / f'{ticker}-json.txt'
        with open(data_file, 'w') as data_file_output:
            json.dump(data, data_file_output)

        return data
    #

    def parse_form_13_to_tsv(self):
        """Parse FMP form 13 and save to tsv files
        """

        for form in tqdm(self.form_list):
            cik = form.stem.split('-')[0]
            with open(form, 'r') as json_file:
                data = json.load(json_file)
                table = pd.DataFrame(data)

                if not table.empty:
                    table_sorted = table.sort_values(by=['nameOfIssuer'])
                    table_sorted['report_date'] = table_sorted['date']
                    table_sorted = table_sorted.drop(columns=['date'])

                    output_path = self.tsv_dir / f'{cik}.tsv'
                    table_sorted.to_csv(output_path, sep='\t', index=False)
            #
        #
    #

    def parse_inst_holder_to_tsv(self):
        """Parse FMP forms and save to tsv files
        """

        for form in tqdm(self.form_list):
            ticker = form.stem.split('-')[0]
            with open(form, 'r') as json_file:
                data = json.load(json_file)
                table = pd.DataFrame(data)

                if not table.empty:
                    table_sorted = table
                    table_sorted['ticker'] = ticker

                    output_path = self.tsv_dir / f'{ticker}.tsv'
                    table_sorted.to_csv(output_path, sep='\t', index=False)
            #
        #
    #

    def parse_etf_holder_to_tsv(self):
        """Parse FMP forms and save to tsv files
        """

        for form in tqdm(self.form_list):
            holder = form.stem.split('-')[0]
            with open(form, 'r') as json_file:
                data = json.load(json_file)
                table = pd.DataFrame(data)

                if not table.empty:
                    table_sorted = table
                    table_sorted['holder'] = holder

                    output_path = self.tsv_dir / f'{holder}.tsv'
                    table_sorted.to_csv(output_path, sep='\t', index=False)
            #
        #
    #

    def parse_insider_trading_to_tsv(self):
        """Parse FMP forms and save to tsv files
        """

        for form in tqdm(self.form_list):
            holder = form.stem.split('-')[0]
            with open(form, 'r') as json_file:
                data = json.load(json_file)
                table = pd.DataFrame(data)

                if not table.empty:
                    table_sorted = table
                    table_sorted['holder'] = holder

                    output_path = self.tsv_dir / f'{holder}.tsv'
                    table_sorted.to_csv(output_path, sep='\t', index=False)
            #
        #
    #

