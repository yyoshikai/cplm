import sys, os, logging
from logging import Logger, Formatter, Handler, StreamHandler, FileHandler, Filter, LogRecord, getLogger
from collections.abc import Callable
from tqdm import tqdm as _tqdm
from .utils import get_git_hash

DEFAULT_FMT = "[{asctime}][{name}][{levelname}]{message}"
DEFAULT_DATEFMT = "%y%m%d %H:%M:%S"

INFO_WORKER = 25

log_name2level = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'warn': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

class TqdmHandler(logging.Handler):
    def __init__(self,level=logging.NOTSET):
        super().__init__(level)

    def emit(self,record):
        try:
            msg = self.format(record)
            _tqdm.write(msg, file=sys.stdout)
            sys.stdout.flush()
            self.flush()
        except Exception:
            self.handleError(record)

class LambdaFilter(Filter):
    def __init__(self, fn: Callable[[LogRecord], bool]):
        self.fn = fn
    
    def filter(self, record: LogRecord):
        return self.fn(record)

def get_logger(name=None, level=logging.DEBUG, stream=False) -> logging.Logger:
    logger = getLogger(name)
    logger.setLevel(level)
    if stream:
        add_stream_handler(logger)
    return logger

def add_file_handler(logger: Logger, path: str, level=logging.DEBUG, mode: str='w', 
        fmt: str=DEFAULT_FMT, datefmt=DEFAULT_DATEFMT, keep_level: bool=False) -> FileHandler:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    handler = FileHandler(path, mode)
    handler.setFormatter(Formatter(fmt, datefmt, style='{'))
    handler.setLevel(level)
    if logger.level > level and not keep_level:
        logger.setLevel(level)
    logger.addHandler(handler)
    return handler

def add_stream_handler(logger: Logger, level=logging.INFO, tqdm: bool=True, 
        fmt: str=DEFAULT_FMT, datefmt: str=DEFAULT_DATEFMT, add=False, 
        keep_level: bool=False) -> Handler:
    if not add:
        for handler in logger.handlers:
            if isinstance(handler, (TqdmHandler, StreamHandler)):
                return handler
    if logger.level > level and not keep_level:
        logger.setLevel(level)
    if tqdm:
        handler = TqdmHandler(level)
    else:
        handler = StreamHandler()
        handler.setLevel(level)
    handler.setFormatter(Formatter(fmt, datefmt, style='{'))
    logger.addHandler(handler)

    return handler

def log_git_hash(logger):
    logger.info(f"git hash={get_git_hash()}")

def disable_openbabel_log():
    from openbabel import openbabel
    handler = openbabel.OBMessageHandler()
    handler.SetOutputLevel(-1)