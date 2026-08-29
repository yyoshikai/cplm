import logging, multiprocessing as mp
import concurrent.futures as cf
from logging import getLogger
from src.utils.logger import get_logger, add_file_handler

def worker_fn(idx, log_queue):
    print(f"Print from worker_fn({idx})")
    logger = getLogger()
    logger.addhandler(logging.handlers.QueueHandler(log_queue))
    logger.info(f"Logging from worker_fn({idx})")


logger = get_logger(stream=True)

logger.info("Logging from __main__")
log_queue = mp.Manager().Queue(-1)
with cf.ProcessPoolExecutor(max_workers=28) as e:
    futures = []
    for idx in range(100):
        futures.append(e.submit(worker_fn, idx, log_queue))
    for f in futures:
        f.result()