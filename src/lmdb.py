import lmdb

def load_lmdb(path: str) ->tuple[lmdb.Environment, lmdb.Transaction]:
    env = lmdb.open(path, subdir=False, readonly=True,
        lock=False, readahead=False, meminit=False, max_readers=256)
    txn = env.begin()
    return env, txn