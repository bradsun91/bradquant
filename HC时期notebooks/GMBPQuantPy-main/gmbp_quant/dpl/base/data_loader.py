import re
import os
import io
import sys
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

logger = LOG.get_logger(__name__)


class DataLoader:
    """A :class:`DataLoader` class, which is base class for other data loaders

    Args:
        data_dir: folder storing downloaded and processed data
        file_id: File Id
        form_type: Filing form type, for example
                    '13F-HR'
                    '13F-HRA'
                    '4'
                    'inst_holder"
    """

    def __init__(self, data_dir, form_type, tsv_dir, db_name, schema):
        self.data_dir = Path(data_dir)
        self.form_type = form_type
        self.tsv_dir = tsv_dir
        self.tsv_dir.mkdir(parents=True, exist_ok=True)
        self._db_name = db_name
        self._schema = schema

    def insert_db(self):
        """Insert parsed tsv data to MySQL DB
        """
        
        db_config = ecfg.get_env_config().get_db_config(ecfg.DBProp.DB)
        db_client = DBClient(db_config=db_config, client_type='mysql',
                                schema=self._db_name)
        self._schema().__table__.create(bind=db_client.engine, checkfirst=True)
        
        session = sessionmaker()
        session.configure(bind=db_client.engine)
        sess = session()

        tsv_list = list(self.tsv_dir.glob('*.tsv'))
        for tsv_path in tqdm(tsv_list):
            tsv_data = pd.read_csv(tsv_path, sep='\t', dtype={'cik':'str'})
            file_id = tsv_path.stem
            tsv_cols = tsv_data.columns
            try:
                for idx, row in tsv_data.iterrows():
                    record_dict = {}
                    logger.info(idx, row)
                    # replace nan with None
                    for col in tsv_cols:
                        record_dict[col] = None if pd.isna(row[col]) else row[col]
                    record_dict['row_num'] = idx
                    record = self._schema(**record_dict)
                    sess.add(record)
                sess.commit()
            except Exception as e:
                # rollback the changes on error
                logger.info(
                    f'Failed. Insert parsed data for file_id {file_id} form {self.form_type} '
                    f'from {tsv_path.name} into '
                    f'{self._db_name}.{self._schema().__tablename__} '
                    f'encountered failure: {str(e)}')
                sess.rollback()
            finally:
                # close session
                sess.close()