from collections import defaultdict
from time import time

class Watch:
    def __init__(self):
        self.name2time = {}
    def hold(self, name: str):
        return _WatchHolder(name, self)
    def add_time(self, name: str, t: float):
        if name not in self.name2time:
            self.name2time[name] = 0.0
        self.name2time[name] += t

class _WatchHolder:
    def __init__(self, name: str, watch: Watch):
        self.name = name
        self.watch = watch
        self.start = None

    def __enter__(self):
        self.start = time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.watch.add_time(self.name, time() - self.start)
