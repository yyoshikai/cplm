import datetime, json
from urllib import request
from logging import getLogger
from time import time
try:
    from ._api_key import SLACK_URL
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
