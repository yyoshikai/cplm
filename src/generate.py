import sys, os, itertools, pickle, yaml, math
from collections.abc import Sized
from typing import Optional, Literal
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, StackDataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem, RDLogger

sys.path += ["/workspace/cplm"]
from src.data.tokenizer import FloatTokenizer, \
    ProteinAtomTokenizer, VocEncoder, TokenEncodeDataset
from src.data import untuple_dataset, index_dataset
from src.data.finetune import  CDDataset, RandomScoreDataset
from src.data.tokenizer import TokenizeDataset, ArrayTokenizeDataset, SentenceDataset
from src.model import Model
from src.model.mamba import MambaModel2
from src.utils import logend, set_random_seed
from src.evaluate import parse_mol_tokens, parse_mol
from src.utils.logger import add_stream_handler, add_file_handler, get_logger
from src.utils.path import cleardir
from src.utils.time import wtqdm
PROJ_DIR = "/workspace/cplm"
WORKDIR = "/workspace"

def generate(model: Model | MambaModel2, rdir: str, n_trial: int, token_per_batch: int, 
        seed: int, max_len: int, index: str, pocket_coord_heavy: bool, 
        coord_range: float, prompt_score: Literal['data', 'low', 'no_score'], 
        state_vocs: list, gtype: int=2):
    
    assert prompt_score in ['data', 'low', 'no_score']

    if os.path.exists(f"{rdir}/info.csv"):
        print(f"{rdir} already finished.")
        return
    
    # Environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(seed)
    ## Result dir
    cleardir(rdir)

    ## Logger
    logger = get_logger()
    add_stream_handler(logger)
    add_file_handler(logger, f"{rdir}/debug.log")
    RDLogger.DisableLog("rdApp.*")

    ## Save args
    with open(f"{rdir}/args.yaml", 'w') as f:
        yaml.dump(dict(rdir=rdir, token_per_batch=token_per_batch, seed=seed, max_len=max_len, index=index, pocket_coord_heavy=pocket_coord_heavy, coord_range=coord_range, prompt_score=prompt_score, gtype=gtype), f)

    # Data
    with logend(logger, 'Prepare data'):

        ## CrossDocked
        data = CDDataset(f"{WORKDIR}/cplm/preprocess/results/finetune/r4_all", seed, random_rotate=False, mol_atom_h=True, mol_coord_h=True, 
            pocket_coord_heavy=pocket_coord_heavy)
        pocket_atom, pocket_coord, _, _, score, center  = untuple_dataset(data, 6)

        ## Sentence
        float_tokenizer = FloatTokenizer(-coord_range, coord_range)
        pocket_atom = TokenizeDataset(pocket_atom, ProteinAtomTokenizer())
        pocket_coord = ArrayTokenizeDataset(pocket_coord, float_tokenizer)
        sentence = ['[POCKET]']

        sentence += [pocket_atom, '[XYZ]', pocket_coord]

        if prompt_score != 'no_score':
            if prompt_score == 'low':
                score = RandomScoreDataset(-12.0, -10.0, len(pocket_atom), seed)
            score = TokenizeDataset(score, float_tokenizer)
            sentence += ['[SCORE]', score]
        sentence += ['[LIGAND]']
        data = SentenceDataset(*sentence)
        
        data_vocs = data.vocs()
        assert data_vocs <= set(state_vocs)
        assert state_vocs[0] == '[PAD]'
        voc_encoder = VocEncoder(state_vocs[1:])
        data = TokenEncodeDataset(data, voc_encoder)

        ## Stack 
        idx_data, data = index_dataset(data)
        data = StackDataset(idx_data, data, center)

        ## Generation data index
        indices = np.load(f"../index/results/{index}.npy")
        data = Subset(data, indices)
    def collate_fn(batch):
        indices, batch, centers = list(zip(*batch))
        batch = pad_sequence(batch, padding_value=voc_encoder.pad_token)
        return indices, batch, centers
    batch_size = token_per_batch // max_len
    num_workers = min(28, batch_size)
    model.to(device)

    # 生成
    sampler = UnfinishedSampler(data, n_trial)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    smiless = [None] * len(data)
    errors = [None] * len(data)
    indices = [None] * len(data)
    os.makedirs(f"{rdir}/sdf", exist_ok=True)

    with torch.inference_mode():
        for batch_idxs in (pbar:=wtqdm(batch_sampler)):
            logger.info(f"{batch_idxs=}")
            pbar.start('data')
            train_loader = DataLoader(Subset(data, batch_idxs), shuffle=False, 
                    num_workers=num_workers, batch_size=len(batch_idxs), 
                    collate_fn=collate_fn, )
            batch_indices, batch, centers = next(iter(train_loader))

            pbar.start("generation")
            batch = batch.to(device)

            match gtype:
                case 1:
                    outputs = model.generate(batch, '[END]', max_len, voc_encoder.pad_token)
                case 2:
                    outputs = model.generate2(batch, '[END]', max_len, voc_encoder.pad_token, 10)
            outputs = [out.cpu().numpy() for out in outputs]
            # detokenize
            pbar.start("detokenize")
            wordss = []
            end_token = voc_encoder.voc2i['[END]']
            for tokens in outputs:
                words = voc_encoder.decode(itertools.takewhile(lambda x: x != end_token, tokens))
                wordss.append(words)

            # parse SMILES and coordinates
            pbar.start("parsing")
            for i in range(len(wordss)):

                words = wordss[i]
                center = centers[i]
                idx = batch_idxs[i]

                if not sampler.is_remain[idx]: continue

                indices[idx] = batch_indices[i]

                error, smiles, coords = parse_mol_tokens(words)
                smiless[idx] = smiles
                if error != "":
                    errors[idx] = error
                    continue

                coords += center
                error, mol = parse_mol(smiles, coords)
                errors[idx] = error
                if error != "":
                    continue
                with open(f"{rdir}/sdf/{idx}.sdf", 'w') as f:
                    f.write(Chem.MolToMolBlock(mol))
                sampler.is_remain[idx] = False
                logger.info(f"{idx}: finished")

    with open(f"{rdir}/tokens.txt", 'w') as f:
        for words in wordss:
            f.write(','.join(words)+'\n')

    with open(f"{rdir}/tokens.pkl", 'wb') as f:
        pickle.dump(outputs, f)

    df = pd.DataFrame({'idx': indices, 'smiles': smiless, 'error': errors})
    df.to_csv(f"{rdir}/info.csv")

class UnfinishedSampler:
    def __init__(self, dataset: Sized, max_cycle: int=math.inf):
        
        self.iter_idxs = list(range(len(dataset)))
        self.is_remain = np.full(len(dataset), True)
        self.max_cycle = max_cycle

    def __iter__(self):

        i_cycle = 0
        while True:
            if np.all(~self.is_remain):
                return
            for i in np.where(self.is_remain)[0]:
                if self.is_remain[i]:
                    yield i
            i_cycle += 1
            if i_cycle >= self.max_cycle:
                return