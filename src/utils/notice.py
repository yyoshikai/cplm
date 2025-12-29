import datetime, json
from urllib import request
from logging import getLogger
from time import time
try:
    from .._api_key import SLACK_URL
except ImportError:
    SLACK_URL = None
logger = getLogger(__name__)

def notice(message, icon=':sunglasses:'):
    if SLACK_URL is None:
        logger.warning("SLACK_URL is not defined.")
        return
    try:
        data = {
            'text': message,
            "icon_emoji": icon,
            "link_names":1
        }
        headers = {
            'Content-Type': 'application/json',
        }

        req = request.Request(SLACK_URL, json.dumps(data).encode(), headers)
        with request.urlopen(req) as res:
            body = res.read()
    except (ConnectionError):
        logger.error("Connection error at notice()")

class EndEstimator:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, duration_h, total_step, name):
        self.duration_h = duration_h
        self.total_step = total_step
        self.name = name
        self.start_time = None
        self.notified = False
    def start(self):
        self.start_time = time()
    def check(self, cur_step: int):
        if self.start_time is None:
            raise ValueError("start() should be called before check().")
        est_time = (time() - self.start_time()) * self.total_step / cur_step
        m, s = divmod(int(est_time), 60)
        h, m = divmod(m, 60)
        msg = f"Estimated end time={h}:{m}:{s} at step {cur_step}"
        self.logger.info(msg)
        if est_time > self.duration_h * 3600 * 0.95:
            if not self.notified and SLACK_URL is not None:
                notice(f"[WARNING][{self.name}] {msg}", )
                self.notified = True




