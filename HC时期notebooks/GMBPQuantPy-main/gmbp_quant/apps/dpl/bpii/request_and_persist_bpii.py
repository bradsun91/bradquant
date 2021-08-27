import os, sys, traceback
import gmbp_quant.common.utils.datetime_utils as dtu
from gmbp_quant.common.utils.miscs import iterable_to_tuple
import gmbp_quant.common.env_config as ecfg
from gmbp_quant.common.db.mysql_client import get_mysql_client
from gmbp_quant.dpl.bpii.bpii_data_api import request_bpii_ts
from sqlalchemy import TIMESTAMP, FLOAT, INTEGER

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def persist_bpii_ts_to_db(bpii_ts):
    db_config = ecfg.get_env_config().get_db_config(ecfg.DBProp.DB)
    db_client = get_mysql_client(db_config=db_config)
    col_map = {'average': 'Average', 'small': 'SmallC', 'middle': 'MiddleC', 'large': 'LargeC', 'sp500': 'SPX_INDEX',
               'SignalDateTime': 'Signal_Time'}
    cols = ['Signal_Time', 'Average', 'SmallC', 'MiddleC', 'LargeC', 'SPX_INDEX', 'LmA', 'PHA', 'BPIISignal']
    dtype_map = {'Signal_Time': TIMESTAMP, 'Average': FLOAT, 'SmallC': FLOAT, 'MiddleC': FLOAT, 'LargeC': FLOAT,
                 'SPX_INDEX': FLOAT, 'LmA': FLOAT, 'PHA': FLOAT, 'BPIISignal': INTEGER}
    try:
        bpii_ts = bpii_ts.copy()
        bpii_ts['BPIISignal'] = bpii_ts['BPIISignal'].astype(int)
        # bpii_ts['timeX'] /= 1e3
        schema = 'quant'
        table = 'bpii'
        bpii_ts.rename(columns=col_map)[cols].to_sql(name=table, con=db_client.engine, schema=schema,
                                                     if_exists='append', index=False, method='multi', dtype=dtype_map)
        logger.info(f'bpii_ts on date={dateid} -> [{schema}].[{table}]')
    except Exception as e:
        logger.error(f'Exception caught !')
        traceback.print_exc()
    #
#


def setup_cli_options(parser=None):
    if parser is None:
        from optparse import OptionParser, IndentedHelpFormatter
        parser = OptionParser(formatter=IndentedHelpFormatter(width=200), epilog='\n')
    #

    today = dtu.today()
    parser.add_option('-E', '--end_date',
                      dest='end_date', default=today,
                      help=f'Default: Current Trading Date {today} .')
    parser.add_option('-S', '--start_date',
                      dest='start_date', default=None,
                      help=f'Default: If not provided, it will get inferred according to "date_range_mode" .')
    parser.add_option('-R', '--date_range_mode',
                      dest='date_range_mode', default='SINGLE_DATE',
                      help=f'Default: %default. Supported values are [SINGLE_DATE|ROLLING_WEEK|ROLLING_MONTH] .')
    parser.add_option('-T', '--tasks',
                      dest='tasks', default='PERSIST_TO_DB',
                      help='Comma separated. Default: %default. Supported are [PERSIST_TO_DB|DUMP_TO_FILES] .')
    parser.add_option('-f', '--output_file',
                      dest='output_file', default=None,
                      help='Output BPII data file. If not provided, then the output file will be inferred from "output_dir" .')
    parser.add_option('-o', '--output_dir',
                      dest='output_dir', default=None,
                      help='Output BPII data file directory. If provided, the output file(s) will be inferred as "bpii.YYYYMMDD.csv" .')
    parser.add_option('-C', '--connection_env',
                      dest='connection_env', default='RESEARCH',
                      help=f'Default: %default .')

    return parser
#


if __name__ == '__main__':
    logger.info(' '.join(sys.argv))

    options, args = setup_cli_options().parse_args()

    connection_env = options.connection_env
    ecfg.get_env_config(env=connection_env)
    logger.info(f'connection environment is {ecfg.get_env_config().env}')

    tasks = iterable_to_tuple(options.tasks, raw_type='str')

    output_file = options.output_file
    if 'DUMP_TO_FILES' in tasks:
        if output_file is None:
            if options.output_dir is None:
                raise Exception(f'With "DUMP_TO_FILES" in "tasks", '
                                f'please provide either "output_file" or "output_dir" !')
            #
        #
    #

    start_dateid, end_dateid = dtu.infer_start_dateid_end_dateid(start_date=options.start_date, end_date=options.end_date,
                                                                 date_range_mode=options.date_range_mode)
    dateids = dtu.infer_biz_dateids(start_dateid=start_dateid, end_dateid=end_dateid)

    for dateid in dateids:
        bpii_ts = request_bpii_ts(dateid=dateid, append_bpii_signals=True)
        if bpii_ts is None:
            logger.warn(f'No bpii_ts on dateid={dateid}')
            continue
        #
        if 'PERSIST_TO_DB' in tasks:
            persist_bpii_ts_to_db(bpii_ts=bpii_ts)
        #

        if 'DUMP_TO_FILES' in tasks:
            if output_file is None:
                output_file = os.path.join(options.output_dir, f'bpii.{dateid}.csv')
            #
            bpii_ts.to_csv(output_file, sep=',', index=False)
            logger.info(f'bpii_ts on date={dateid} -> {output_file}')
        #
    #
#
