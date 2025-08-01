import sys, os, logging
from logging import Logger, Formatter, StreamHandler, FileHandler, Filter, LogRecord, getLogger
from collections.abc import Callable
from tqdm import tqdm as _tqdm

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
        fmt: str=DEFAULT_FMT, datefmt=DEFAULT_DATEFMT):
    handler = FileHandler(path, mode)
    handler.setFormatter(Formatter(fmt, datefmt, style='{'))
    handler.setLevel(level)
    logger.addHandler(handler)

def add_stream_handler(logger: Logger, level=logging.INFO, tqdm: bool=True, 
        fmt: str=DEFAULT_FMT, datefmt: str=DEFAULT_DATEFMT, add=False):
    if not add:
        for handler in logger.handlers:
            if isinstance(handler, (TqdmHandler, StreamHandler)):
                return
    if tqdm:
        handler = TqdmHandler(level)
    else:
        handler = StreamHandler()
        handler.setLevel(level)
    handler.setFormatter(Formatter(fmt, datefmt, style='{'))
    logger.addHandler(handler)
