import re
import os
import io
import sys
import yaml
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
from gmbp_quant.dpl.base.data_loader import DataLoader
from gmbp_quant.dpl.sec_edgar.sec_schemas import (
        Sec13FHRSchema, Sec13FHRASchema,
        Sec4NonDerivativeSchema, Sec4DerivativeSchema)


schemas = {
    'Sec13FHRSchema': Sec13FHRSchema,
    'Sec13FHRASchema': Sec13FHRASchema,
    'Sec4NonDerivativeSchema': Sec4NonDerivativeSchema,
    'Sec4DerivativeSchema': Sec4DerivativeSchema
}

logger = LOG.get_logger(__name__)


class SecEdgarDataLoader(DataLoader):
    """A :class:`SecEdgarDataLoader` class

    Args:
        data_dir: folder storing SEC EDGAR data
        form_type: SEC form type, currently support
                    '13F-HR'
                    '13F-HRA'
                    '4-NonDerivative'
                    '4-Derivative'
        year: year of SEC reporting
        quarter: quarter of SEC reporting
    """

    def __init__(self, data_dir, form_type, year, quarter):

        self.data_dir = Path(data_dir)
        self.form_type = form_type
        self.xsl_file = Path(__file__).parent / 'xsl_files' / f'{form_type}.xsl'

        sec_yml_path = Path(__file__).parent / 'sec_forms.yml'
        with open(sec_yml_path, 'r') as yamlfile:
            self.sec_yml = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
        self.form_dir = self.data_dir/ self.sec_yml[form_type]['form_folder'] / year / quarter
        self.form_list = list(self.form_dir.glob('*.txt'))
        self.tsv_dir = self.data_dir / 'tsv_files' / form_type / year / quarter
        self.tsv_dir.mkdir(parents=True, exist_ok=True)
        self.year, self.quarter = year, quarter

        self._db_name = 'secdata'
        self._schema = schemas[self.sec_yml[form_type]['schema']]
        self.table_keyword = self.sec_yml[form_type]['table_keyword']

        super().__init__(data_dir, form_type, self.tsv_dir, self._db_name, self._schema)


    def get_essentials_from_txt(self, xml_txt):
        """Get essential fields from xml text.
        """

        rows = xml_txt.split('\n')
        cik_regex = re.compile(self.sec_yml['cik'], re.I)
        essentials_regex = {}
        for k, v in self.sec_yml[self.form_type]['essentials'].items():
            essentials_regex[k] = re.compile(v, re.I)
        
        # get regex dict by form type
        essentials = {}
        for row in rows:
            for k, v in essentials_regex.items():
                if v.search(row):
                    essentials[k] = row.replace('\t', '').split(':')[1]
            if cik_regex.search(row):
                essentials['cik'] = row.replace('\t', '').split(':')[1]
                return essentials


    def parse_forms_to_tsv(self):
        """Parse SEC forms and save to tsv files
        """

        for form in tqdm(self.form_list):
            with open(form, 'r') as f:
                xml_data = f.read()
            # get CIK and dates
            essentials = self.get_essentials_from_txt(xml_data)

            soup = BeautifulSoup(xml_data, 'lxml')
            table_re = re.compile(f'^{self.table_keyword}', re.I)
            if soup.find(table_re):
                soup.find(table_re).attrs = {}
            else:
                continue
            
            namespace_re = re.compile('^.+\:.+$', re.I)
            tag_name_re = re.compile('^.+\:(.+)$', re.I)
            for tag in soup.find_all(namespace_re):
                tag.name = tag_name_re.match(tag.name).group(1)

            data_table = soup.find(table_re)
            # check if table is empty
            if data_table is None:
                continue
            
            with open(self.xsl_file, 'r') as f:
                xslt = et.parse(f)

            dom = et.parse(io.StringIO(data_table.prettify()))
            transform = et.XSLT(xslt)
            tsv_data = transform(dom)

            output_path = self.tsv_dir / f'{form.stem}.tsv'
            with open(output_path, 'w') as f:
                f.write(str(tsv_data))

            tsv_data = pd.read_csv(output_path, sep='\t')
            # fill in essential fields
            tsv_data['document_id'] = form.stem
            for k, v in essentials.items():
                tsv_data[k] = v

            tsv_data.to_csv(output_path, sep='\t', index=False)
