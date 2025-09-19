import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import TypeVar, Optional
from logging import getLogger
from time import time

import pandas as pd
import torch.distributed as dist
from tqdm import tqdm

from .utils import IterateRecorder

T_co = TypeVar('T_co', covariant=True)

class _Watch:
    def hold(self, name: str):
        return _WatchHolder(name, self)
    def add_time(self, name: str, start: float, end: float):
        raise NotImplementedError

class Watch(_Watch):
    def __init__(self):
        self.name2time = {}
    def add_time(self, name: str, start: float, end: float):
        if name not in self.name2time:
            self.name2time[name] = 0.0
        self.name2time[name] += end - start

class _WatchHolder:
    def __init__(self, name: str, watch: _Watch):
        self.name = name
        self.watch = watch
        self.start = None

    def __enter__(self):
        self.start = time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.watch.add_time(self.name, self.start, time())

class AddWatch(_Watch):
    def __init__(self):
        self.name2time = defaultdict(float)
    def add_time(self, name: str, start: float, end: float):
        self.name2time[name] += end - start

class FileWatch(_Watch):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, file_path: str, freq: int=10000):
        self.path = file_path
        self.freq = freq

        self.names = []
        self.starts = []
        self.ends = []
        with open(self.path, 'w') as f:
            f.write("name,start,time\n")

    def hold(self, name: str):
        return _WatchHolder(name, self)

    def add_time(self, name: str, start: float, end: float):
        self.names.append(name)
        self.starts.append(start)
        self.ends.append(end)
        if len(self.names) >= self.freq:
            self.flush()

    def flush(self):
        self.logger.info(f"Writing times...")
        with open(self.path, 'a') as f:
            for name, start, end in zip(self.names, self.starts, self.ends):
                f.write(f"{name},{start},{end-start}\n")
        self.names = []
        self.starts = []
        self.ends = []
        self.logger.info(f"Wrote.")

class WriteHolder:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.start = None

    def __enter__(self):
        self.start = time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        end = time()
        with open(self.path, 'a') as f:
            f.write(f"{self.name},{self.start},{end}\n")

class PrintHolder:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        end = time()
        print(f"{self.name}: {end-self.start:.05f}s")

class wtqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name2time = defaultdict(float)
        self.cur_job = 'after_init'
        self.start_ = time()

    def __iter__(self):
        if self.disable:
            yield from self.iterable
            return

        self.start('iter_data')
        for item in super().__iter__():
            self.start('after_iter')
            yield item
            self.start('iter_data')

    def start(self, name):
        t = time()
        self.name2time[self.cur_job] += t - self.start_
        self.cur_job = name
        self.start_ = t

    def update(self, n=1):
        self.set_postfix_str(', '.join([f"{key}={value:.03f}" for key, value in sorted(self.name2time.items(), key=lambda x: x[1], reverse=True)]), refresh=False)
        super().update(n)

class TimerTqdm(tqdm):
    logger = getLogger(f'{__module__}.{__qualname__}')

    def __init__(self, iterable: Optional[Iterable]=None,
            time_path: Optional[str]=None, log_interval: int=None, file_interval: int=None, disable_bar: bool=False, sync_ddp: bool=False, *args, **kwargs):
        self.disable_bar = disable_bar
        super().__init__(iterable, *args, **kwargs)

        # Recording
        self.name2time = defaultdict(float)

        if time_path is not None:
            assert file_interval is not None
            self.recorder = IterateRecorder(time_path, [], file_interval)
        else:
            self.recorder = None
        self.log_interval = log_interval

        self.sync_ddp = sync_ddp

        # current state
        self.cur_job = 'after_init'
        self.start_ = time()

    def __iter__(self):
        self.start('iter_data')
        for item in super().__iter__():
            self.start('after_iter')
            yield item
            self.start('iter_data')
        self.recorder.flush()

    def start(self, name, never_sync: bool=False):
        if self.sync_ddp and not never_sync:
            dist.barrier()
        t = time()
        self.name2time[self.cur_job] += t - self.start_
        self.cur_job = name
        self.start_ = t
        self.prev_name2time = {}

    def sorted_items(self) -> list[tuple[str, float]]:
        return sorted(self.name2time.items(), key=lambda x: x[1], reverse=True)

    def update(self, n=1):

        # set postfix
        if not self.disable:
            self.set_postfix_str(', '.join([f"{key}={value:.03f}" for key, value in self.sorted_items()]), refresh=False)

        super().update(n)

        # Add cur_time
        if self.recorder is not None:
            print(f"{self.prev_name2time=}", flush=True)
            self.recorder.record(n=self.n, **{name: t - self.prev_name2time.get(name, 0) for name, t in self.name2time.items()})
        self.prev_name2time = self.name2time.copy()
        print(f"{self.prev_name2time=}", flush=True)

        # log
        if self.log_interval is not None and \
                ((self.n-n) // self.log_interval) != (self.n // self.log_interval):
            name_times = self.sorted_items()
            max_name_len = max([len(name) for name, _ in name_times])
            for name, time in name_times:
                self.logger.debug(f"Timer[{self.desc}][{self.n}]{name.ljust(max_name_len)}:{time:5.2f}")

    def display(self, msg=None, pos=None):
        if not self.disable_bar:
            super().display(msg, pos)

LOGTIME = False
def set_logtime(logtime: bool):
    global LOGTIME
    LOGTIME = logtime

class logtime:
    def __init__(self, logger: logging.Logger, prefix: str='', level=logging.DEBUG, thres: float=0):
        self.logger = logger
        self.prefix = prefix
        self.level = level
        self.thres = thres
    def __enter__(self):
        if LOGTIME:
            self.start = time()
    def __exit__(self, exc_type, exc_value, traceback):
        if LOGTIME:
            elapse = time() - self.start
            if elapse >= self.thres:
                self.logger.log(self.level, f"{self.prefix} {elapse:.4f}") 
class logend:
    def __init__(self, logger: logging.Logger, process_name: str, level=logging.INFO, thres: float=0.0):
        self.logger = logger
        self.process_name = process_name
        self.level = level
        self.thres = thres
    def __enter__(self):
        self.start = time()
        self.logger.log(self.level, f"{self.process_name}...")
    def __exit__(self, exc_type, exc_value, traceback):
        t = time() - self.start
        if t >= self.thres:
            self.logger.log(self.level, f"{self.process_name} ended ({t:.03}s).")

class rectime: 
    def __init__(self):
        pass
    def __enter__(self):
        self.start = time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.time = time() - self.start