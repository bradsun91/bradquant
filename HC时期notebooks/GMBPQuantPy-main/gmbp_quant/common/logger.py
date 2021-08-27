import logging
import yaml


def config_logging(logging_config=None, config_file_type='YAML'):
    if logging_config is None:
        return
    #
    logging_config_file = None
    if isinstance(logging_config, str):
        if config_file_type != 'YAML':
            raise Exception(f'config_file_type={config_file_type} is not supported in [YAML] ！')
        #
        logging_config_file = logging_config
        with open(logging_config_file, 'r') as fh:
            logging_config = yaml.safe_load(fh.read())
        #
    #

    if not isinstance(logging_config, dict):
        raise Exception('logging_config has to be of type [dict | str(for config file name)] !')
    #

    logging.config.dictConfig(logging_config)

    if logging_config_file is not None:
        logging.getLogger().info(f'logging configured from {logging_config_file}.')
    else:
        logging.getLogger().info(f'logging configured according to dict param logging_config.')
    #
#


DEFAULT_FORMAT = '[%(levelname).1s %(asctime)s %(module)s:%(funcName)s] - %(message)s'
LEVEL_2_LOGGING_LEVEL = {'DEBUG'    : logging.DEBUG,
                         'INFO'     : logging.INFO,
                         'WARN'     : logging.WARN,
                         'ERROR'    : logging.ERROR,
                         'CRITICAL' : logging.CRITICAL
                         }


class LoggerManager:
    def __init__(self):
        self.loggers = dict()
        self.file_handlers = dict()
    #

    def get_logger(self, logger_name, logger_format=DEFAULT_FORMAT):
        if logger_name in self.loggers:
            return self.loggers[logger_name]
        #

        logger = logging.getLogger(logger_name)
        logger.propagate = False
        formatter = logging.Formatter(logger_format)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)

        self.loggers[logger_name] = logger
        return logger
    #

    def set_level(self, level, logger_name=None):
        loggers = list()
        if logger_name is not None:
            logger = self.get_logger(logger_name)
            loggers.append(logger)
        else:
            loggers = self.loggers.values()
        #
        for logger in loggers:
            logger.setLevel(LEVEL_2_LOGGING_LEVEL[level])
        #
    #

    def set_logfile(self, logfile, logger_name=None, logger_format=DEFAULT_FORMAT):
        loggers = list()
        if logger_name is not None:
            logger = self.get_logger(logger_name)
            loggers.append(logger)
        else:
            loggers = self.loggers.values()
        #

        formatter = logging.Formatter(logger_format)

        for logger in loggers:
            fh = logging.FileHandler(logfile, 'a')
            self.file_handlers[logfile] = fh
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        #
    #

    def disable_logfile(self, logfile):
        fh = self.file_handlers.get(logfile, None)
        if fh is None:
            return
        #
        for logger in self.loggers.values():
            logger.removeHandler(fh)
        #
        fh.close()
    #
#


LOG = LoggerManager()
