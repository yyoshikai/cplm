from collections import defaultdict
from collections.abc import Iterable
from typing import TypeVar, Optional
from logging import getLogger
from time import time

from tqdm import tqdm

from .utils import IterateRecorder
from .notice import notice, SLACK_URL

T_co = TypeVar('T_co', covariant=True)

class _Watch:
    def hold(self, name: str):
        return _WatchHolder(name, self)
    def add_time(self, name: str, start: float, end: float):
        raise NotImplementedError

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

class EndEstimator:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, duration_h, total_step, name):
        self.duration_h = duration_h
        self.total_step = total_step
        self.name = name
        self.start_time = None
        self.notified = False
        self.is_started = False
    def start(self):
        self.start_time = time()
        self.is_started = True
    def check(self, cur_step: int):
        assert cur_step > 0
        if self.start_time is None:
            raise ValueError("start() should be called before check().")
        est_time = (time() - self.start_time) * self.total_step / cur_step
        m, s = divmod(int(est_time), 60)
        h, m = divmod(m, 60)
        msg = f"Estimated end time={h}:{m:02}:{s:02} at step {cur_step}"
        self.logger.info(msg)
        if est_time > self.duration_h * 3600 * 0.95:
            if not self.notified and SLACK_URL is not None:
                notice(f"[WARNING][{self.name}] {msg}", )
                self.notified = True

