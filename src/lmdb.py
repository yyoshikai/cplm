import os
import lmdb

def load_lmdb(path: str) ->tuple[lmdb.Environment, lmdb.Transaction]:
    env = lmdb.open(path, subdir=False, readonly=True,
        lock=False, readahead=False, meminit=False, max_readers=256)
    txn = env.begin()
    return env, txn

def new_lmdb(path: str, keep_exists: bool=False) -> tuple[lmdb.Environment, lmdb.Transaction]:
    if os.path.exists(path) and not keep_exists:
        os.remove(path)
    env = lmdb.open(path, subdir=False, readonly=False,
        lock=False, readahead=False, meminit=False, max_readers=1,
        map_size=int(100e9))
    txn = env.begin(write=True)
    return env, txn