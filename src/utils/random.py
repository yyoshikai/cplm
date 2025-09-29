import random
import numpy as np
import torch
import torch.distributed as dist

# random
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

def random_state_dict():
    state_dict = {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all()
    }
    return state_dict

def load_random_state_dict(state_dict: dict):
    random.setstate(state_dict['random'])
    np.random.set_state(state_dict['numpy'])
    torch.set_rng_state(state_dict['torch'])
    torch.cuda.set_rng_state_all(state_dict['cuda'])


def ddp_set_random_seed(seed: int):
    """
    DDPでの挙動について
    1(採用). 各プロセスでmanual_seed(): 
        set_device()前だと0しか初期化されない
        → current_deviceでdeviceを確認する。
    2(x). masterのみでmanual_seed_all():
        node間並列ではmaster nodeしか初期化されない
    3(x). 各プロセスでmanual_seed_all()
        あるプロセスが初期化後処理を行った後別のプロセスが再度初期化しないよう, 処理をブロックする。
        (masterのみでの初期化を防止することを兼ねる。)
        init_process_group()前(is_initialized()=False)だと同期できないのでエラー
        → プロセスごとに異なるseedを指定するかもしれないのでやめる
    """
    # check init_process_group() and set_device() was called.
    assert dist.is_initialized()
    estimated_device = dist.get_rank() % torch.cuda.device_count()
    assert estimated_device == torch.cuda.current_device()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)