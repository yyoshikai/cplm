import math
from collections import defaultdict
from collections.abc import Iterable
from graphlib import TopologicalSorter
from typing import TypeVar
from logging import Logger
from time import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ..utils import should_show
from ..utils.logger import NO_DUP


"""
Basic usage:

loop_streamer = LoopStremer()

loop_streamer.start_loops() (required for some)
for i in range(10):
    loop_streamer.put('first process')
    a = 1

    loop_streamer.put('second process')
    b = a+1

    loop_streamer.end_loop()
"""
T = TypeVar('T')

class Looper:
    def start_loops(self):
        pass
    def put(self, process: str):
        pass
    def end_loop(self):
        pass
    def end_loops(self):
        pass

class Loopers(list[Looper], Looper):
    def start_loops(self):
        for streamer in self:
            streamer.start_loops()
    def put(self, process):
        for streamer in self:
            streamer.put(process)
    def end_loop(self):
        for streamer in self:
            streamer.end_loop()
    def end_loops(self):
        for streamer in self:
            streamer.end_loops()

class LogLooper(Looper):
    def __init__(self, logger: Logger, interval: int, init: int, loop_name: str, loops_name: str|None=None):
        self.logger = logger
        self.i_loop = 0
        self.interval = interval
        self.init = init
        self.loop_name = loop_name
        self.loops_name = loops_name
    def start_loops(self):
        if self.loops_name is not None:
            self.logger.info(f"{self.loops_name} started.", **NO_DUP)
    def put(self, process):
        if self.i_loop < self.init:
            self.logger.debug(f"{self.loop_name}[{self.i_loop}] {process} started.")
    def end_loop(self):
        self.i_loop += 1
        if self.i_loop <= self.init or self.i_loop % self.interval == 0:
            self.logger.info(f"{self.i_loop} {self.loop_name} finished.", **NO_DUP)
    def end_loops(self):
        if self.loops_name is not None:
            self.logger.info(f"{self.loops_name} finished.", **NO_DUP)

class TimeLooper(Looper):
    def __init__(self):
        self.start_loops_called = False
    
        self.cur_process = None
        self.cur_start_t = None
    
        self.process_graph = defaultdict(set) # {later process: {former processes}}
        self.process_graph['init'] = set()
        self.process2t = {}

    def processes(self):
        return list(TopologicalSorter(self.process_graph).static_order())

    def start_loops(self):
        self.cur_start_t = time()
        self.cur_process = 'init'
        self.start_loops_called = True

    def put(self, process):
        cur_end_t = time()
        if not self.start_loops_called:
            raise ValueError(f"start_loops() must be called before put({process}).")
        if self.cur_process in self.process2t:
            self.process2t[self.cur_process] += cur_end_t - self.cur_start_t
        else:
            if process != 'init':
                self.process_graph[process].add(self.cur_process)
            self.process2t[self.cur_process] = cur_end_t - self.cur_start_t
            self.cur_process = process
            self.cur_start_t = cur_end_t

    def end_loop(self):
        self.put('init')
        self.process2t = {}

class TimeWriteLooper(TimeLooper):
    def __init__(self, path: str, write_max_interval: int):
        super().__init__()
        self.write_max_interval = write_max_interval
        self.path = path
        
        self.i_loop = 0
        self.n_written = 0
        self.process2ts = {}
        self.df_processes = None

    def end_loop(self):
        self.put('init')
        for process, ts in self.process2ts.items():
            self.process2ts[process].append(self.process2t.pop(process, math.nan))
        for new_process, t in self.process2t.items():
            self.process2ts[new_process] = [math.nan]*(self.i_loop-self.n_written)+[t]
        self.i_loop += 1
        if should_show(self.i_loop, self.write_max_interval):
            self.write()
        self.process2t = {}
    
    def end_loops(self):
        if self.n_written != self.i_loop:
            self.write()
    
    def write(self):
        processes = self.processes()
        if self.df_processes is None:
            pd.DataFrame(self.process2ts)[processes].to_csv(self.path, index=False)
        elif self.df_processes != processes:
            df_written = pd.read_csv(self.path)
            data = { process: np.concatenate([
                    df_written[process] if process in df_written.columns else np.full(self.n_written, np.nan), 
                    np.array(ts)
            ]) for process, ts in self.process2ts.items() }
            pd.DataFrame(data)[processes].to_csv(self.path, index=False)
        else:
            pd.DataFrame(self.process2ts)[processes].to_csv(self.path, mode='a', header=False, index=False)
        self.df_processes = processes
        self.process2ts = {process: [] for process in processes}
        self.n_written = self.i_loop

class TimeLogLooper(TimeLooper):
    def __init__(self, logger: Logger, loop_name: str, max_interval: int):
        super().__init__()
        self.max_interval = max_interval
        self.loop_name = loop_name

        self.process2t_total = defaultdict(float)
        self.logger = logger
        self.i_loop = 0
        self.last_log_loop = None

    def end_loop(self):
        self.put('init')
        for process, t in self.process2t.items():
            self.process2t_total[process] += t
        self.i_loop += 1
        if should_show(self.i_loop, self.max_interval):
            self.log()
        self.process2t = defaultdict(float)
    
    def end_loops(self):
        if self.last_log_loop != self.i_loop:
            self.log()
    
    def log(self):
        processes = self.processes()
        self.logger.debug(f"Times in {self.i_loop} {self.loop_name}:")
        max_process_width = max(len(process) for process in processes)
        max_t_width = max(len(str(int(t))) for t in self.process2t_total.values())
        for process in processes:
            self.logger.debug(f"    {process:>{max_process_width}}: "
                    f"{self.process2t_total[process]:>{max_t_width}.03f}s")
        self.last_log_loop = self.i_loop


def iter_loop(it: Iterable[T], loop_streamer: Looper) -> Iterable[T]:
    loop_streamer.start_loops()
    for item in it:
        yield item
        loop_streamer.end_loop()
    loop_streamer.end_loops()

class TqdmLooper(Looper):
    def __init__(self, total: int|None, desc: str|None):
        self.total = total
        self.desc = desc
        self.pbar = None
    def start_loops(self):
        self.pbar = tqdm(total=self.total, desc=self.desc)
    def end_loop(self):
        assert self.pbar is not None, "start_loops() is not called before end_loop()"
        self.pbar.update()

class GPUUseLooper(Looper):
    def __init__(self, logger: Logger, device: torch.device, loop_name: str, init: int):
        self.logger = logger
        self.device = device
        self.loop_name = loop_name
        self.init = init
        self.i_loop = 0
    def start_loops(self):
        torch.cuda.reset_peak_memory_stats()
    def end_loop(self):
        if self.i_loop < self.init:
            gpu_use = torch.cuda.max_memory_allocated(self.device)/2**30
            self.logger.debug(f"{self.loop_name}[{self.i_loop}] max GPU use={gpu_use:.03f}")
        self.i_loop += 1
        if self.i_loop < self.init:
            torch.cuda.reset_peak_memory_stats()
