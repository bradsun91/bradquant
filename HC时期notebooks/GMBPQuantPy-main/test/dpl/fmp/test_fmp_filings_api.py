import json as json
import os as os
import pandas as pd
from pathlib import Path
import gmbp_quant.common.utils.datetime_utils as dtu
import gmbp_quant.common.unittest as ut
from gmbp_quant.dpl.fmp.fmp_data_loader import FmpDataLoader


class TestFMPFilingsAPI(ut.TestCase):
    def test_request_form_13_data(self):
        data_dir = Path(__file__).parent / 'fmp_data'
        cik = '0000007195'
        form_type = '13F'
        report_date = '2020-09-30'
        fmp_loader = FmpDataLoader(data_dir, form_type, report_date)
        target = fmp_loader.request_form_13_data(cik, report_date)
    #

    def test_parse_form_13_to_tsv(self):
        # data_file and benchmark_file are manually created by above test: test_request_form_13_data
        data_dir = Path(__file__).parent / 'fmp_data'
        form_type = '13F'
        report_date = '2020-09-30'
        fmp_loader = FmpDataLoader(data_dir, form_type, report_date)

        fmp_loader.parse_form_13_to_tsv()
    #

    def test_request_inst_holder_data(self):
        data_dir = Path(__file__).parent / 'fmp_data'
        ticker = 'AAPL'
        form_type = 'inst_holder'
        fmp_loader = FmpDataLoader(data_dir, form_type)
        target = fmp_loader.request_inst_holder_data(ticker)
    #

    def test_parse_inst_holder_to_tsv(self):
        data_dir = Path(__file__).parent / 'fmp_data'
        form_type = 'inst_holder'
        fmp_loader = FmpDataLoader(data_dir, form_type)

        fmp_loader.parse_inst_holder_to_tsv()
    #

    def test_request_etf_holder_data(self):
        data_dir = Path(__file__).parent / 'fmp_data'
        holder = 'SPY'
        form_type = 'etf_holder'
        fmp_loader = FmpDataLoader(data_dir, form_type)
        target = fmp_loader.request_etf_holder_data(holder)
    #

    def test_parse_etf_holder_to_tsv(self):
        data_dir = Path(__file__).parent / 'fmp_data'
        form_type = 'etf_holder'
        fmp_loader = FmpDataLoader(data_dir, form_type)

        fmp_loader.parse_etf_holder_to_tsv()
    #

    def test_request_insider_trading_data(self):
        data_dir = Path(__file__).parent / 'fmp_data'
        ticker = 'AAPL'
        form_type = 'insider_trading'
        fmp_loader = FmpDataLoader(data_dir, form_type)
        target = fmp_loader.request_insider_trading_data(ticker)
    #

    def test_parse_insider_trading_to_tsv(self):
        data_dir = Path(__file__).parent / 'fmp_data'
        form_type = 'insider_trading'
        fmp_loader = FmpDataLoader(data_dir, form_type)

        fmp_loader.parse_insider_trading_to_tsv()
    #
#

if __name__ == '__main__':
    ut.main()
#
