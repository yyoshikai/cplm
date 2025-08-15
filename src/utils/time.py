from collections import defaultdict
from collections.abc import Iterable
from typing import TypeVar
from logging import getLogger
from time import time
from tqdm import tqdm
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

        self.start('first_iter_data')
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
