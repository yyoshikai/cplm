import sys, os, shutil, stat
from glob import glob
from pathlib import Path
from datetime import datetime
from addict import Dict

WORKDIR = os.environ.get('WORKDIR', "/workspace")

def cleardir(path):
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
        elif duplicate == 'exit':
            print(f"{path} already exists. process finished.")
            sys.exit()
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

def timestamp2():
    dt_now = datetime.now()
    return f"{dt_now.year%100:02}{dt_now.month:02}{dt_now.day:02}_{dt_now.hour:02}{dt_now.minute:02}{dt_now.second:02}"


def subs_vars(config, vars):
    if isinstance(config, str):
        if config in vars:
            return vars[config]
        for key, value in vars.items():
            config = config.replace(key, str(value))
        return config
    elif isinstance(config, dict):
        return Dict({label: subs_vars(child, vars) for label, child in config.items()})
    elif isinstance(config, list):
        return [subs_vars(child, vars) for child in config]
    else:
        return config

def make_pardir(path: str):
    return os.makedirs(os.path.dirname(path), exist_ok=True)

def mwrite(path: str, string: str):
    make_pardir(path)
    with open(path, 'w') as f:
        f.write(string)