import itertools as itr
import torch
from torch.utils.data import DataLoader
from src.data.datasets.crossdocked import CDDataset

# indexが重複していないか？
datas = {split: CDDataset(split) for split in ['train', 'valid', 'test']}
idxs = {split: list(data.indices) for split, data in datas.items()}
for split0, split1 in itr.combinations(['train', 'valid', 'test'], 2):
    idx0 = idxs[split0]
    idx1 = idxs[split1]
    print(f"{split0=}, {split1=}, {len(idx0)=}, {len(idx1)=}, {len(idx0&idx1)=}, {len(idx0)|len(idx1)=}")

# test setのdnameがsplit_by_nameと一致しているか？
loader = DataLoader(datas['test'], batch_size=None, shuffle=False, num_workers=16)
test_dnames = {item[3].split('/')[-2] for item in loader}
split_by_name = torch.load("/workspace/cheminfodata/crossdocked/targetdiff/split_by_name.pt", weights_only=True)
test_dnames0 = {item[0].split('/')[0] for item in split_by_name['test']}
print(f"{len(test_dnames)=}, {len(test_dnames0)=}, {len(test_dnames0&test_dnames)=}")

# (scaffoldの調査は面倒なので省略)


