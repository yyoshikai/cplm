import sys, os
import re
import shutil
from glob import glob
import subprocess
import pickle
from datetime import datetime
import pandas as pd

def cleardir(path, exist_ok=None):
    _cleardir(path)
    os.makedirs(path)

def _cleardir(path):
    for cpath in glob(os.path.join(path, '*')):
        if os.path.isdir(cpath):
            _cleardir(cpath)
        else:
            os.remove(cpath)
    if os.path.exists(path):
        os.rmdir(path)


def make_dir(path=None, duplicate='ask'):
    if os.path.exists(path):
        if duplicate == 'error':
            raise FileExistsError(f"'{path}' already exists.")
        elif duplicate == 'ask':
            answer = None
            while answer not in ['y', 'n']:
                answer = input(f"'{path}' already exists. Will you overwrite? (y/n)")
            if answer == 'n':
                return
        elif duplicate in {'overwrite', 'merge'}:
            pass
        else:
            raise ValueError(f"Unsupported duplicate: {duplicate}")
    if duplicate == 'merge':
        os.makedirs(path, exist_ok=True)
    else:
        cleardir(path)
    return path

def timestamp():
    dt_now = datetime.now()
    return f"{dt_now.year%100:02}{dt_now.month:02}{dt_now.day:02}"
