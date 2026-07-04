import re
from time import time
from collections.abc import Generator, Callable
from logging import getLogger
from typing import Literal
import numpy as np, pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from openbabel import openbabel as ob
from .utils import should_show
from .utils.path import make_pardir, mwrite
from .model import Streamer, WrapperStreamer
from .data.tokenizer import VocEncoder
from .data.mol_tokenizer import MolTokenizer, encode_token_stream, pos_offset_stream
from .chem.convert import obmol2rdmol

class NoTokenRangeStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, voc_size):
        super().__init__(streamer)
        self.token_range = list(range(voc_size))
    def put(self, tokens: list[int]) -> tuple[bool, int, list[int]]:
        is_remain, position, token_range = self.streamer.put(tokens)
        return is_remain, position, self.token_range

# Verbosity
class TokenWriteStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, voc_encoder: VocEncoder, prompt_position: list[int], prompt_csv_path: str|None, new_csv_path: str):
        super().__init__(streamer)
        self.voc_encoder = voc_encoder
        self.prompt_position = prompt_position
        self.prompt_csv_path = prompt_csv_path
        self.new_csv_path = new_csv_path
        self.is_prompt = True
    def put(self, tokens: list[int]):
        token_ids = tokens
        tokens = self.voc_encoder.decode(tokens)
        if self.is_prompt:
            assert len(tokens) == len(self.prompt_position), f"{len(tokens)=} != {len(self.prompt_position)=}"
            if self.prompt_csv_path is not None:
                make_pardir(self.prompt_csv_path)
                pd.DataFrame({'position': self.prompt_position, 'token': tokens}).to_csv(self.prompt_csv_path, index=False)
            self.prompt_position = None
            mwrite(self.new_csv_path, "position,token\n")
        else:
            if len(tokens) == 0: # generation is ended
                return False, 0, [self.voc_encoder.pad_token]
            assert len(tokens) == 1, str(tokens)
            with open(self.new_csv_path, 'a') as f:
                f.write(f"{tokens[0]}\n")
        
        is_remain, position, token_range = self.streamer.put(token_ids)
        with open(self.new_csv_path, 'a') as f:
            f.write(f"{position},")
        self.is_prompt = False
        return is_remain, position, token_range

class RangeWriteStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, voc_encoder: VocEncoder, range_path: str):
        super().__init__(streamer)
        self.voc_encoder = voc_encoder
        self.range_path = range_path
        self.is_init = True
    def put(self, tokens: list[int]):
        is_remain, position, token_range = self.streamer.put(tokens)
        if self.is_init:
            mwrite(self.range_path, "token_range\n")
            self.is_init = False
        with open(self.range_path, 'a') as f:
            f.write(','.join(self.voc_encoder.i2voc[t] for t in token_range)+'\n')
        return is_remain, position, token_range

class TokenSaveStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer):
        super().__init__(streamer)
        self.prompt_tokens = None
        self.new_tokens = []
    def put(self, tokens: list[int]):
        if self.prompt_tokens is None:
            self.prompt_tokens = tokens
        else:
            self.new_tokens += tokens
        return self.streamer.put(tokens)

class PositionSaveStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer):
        super().__init__(streamer)
        self.new_positions = []
    def put(self, tokens: list[int]):
        is_remain, position, token_range = self.streamer.put(tokens)
        if is_remain:
            self.new_positions.append(position)
        return is_remain, position, token_range

class TimeLogStreamer(WrapperStreamer):
    logger = getLogger(f"{__qualname__}")
    def __init__(self, streamer: Streamer, name: str, wait_time: float=1.0):
        super().__init__(streamer)
        self.name = name
        self.is_prompt = True
        self.mean_dt = None
        self.n = 0
        self.wait_time = wait_time
    def put(self, tokens: list[int]):
        if self.is_prompt:
            self.start = self.init = time()
            self.is_prompt = False
        else:
            dt = (end:=time()) - self.start
            self.mean_dt = dt if self.mean_dt is None else self.mean_dt*0.9+dt*0.1
            self.start = end
            self.n += 1
            if should_show(self.n) and (t:=self.start-self.init) >= self.wait_time:
                est_n = self.estimated_n_token()
                if est_n is None:
                    self.logger.info(f"[{self.name}]generated {self.n}/? token in {t:.02f}s")
                else:
                    est_t = self.mean_dt*(est_n-self.n)
                    self.logger.info(f"[{self.name}]generated {self.n}/{est_n} token in {t:02f}s (estimated to end in {est_t:.02f}s)")
        return self.streamer.put(tokens)

class TqdmStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, total: int|None=None, desc: str|None=None):
        super().__init__(streamer)
        self.pbar = tqdm(total=total, desc=desc)
    
    def put(self, tokens: list[int]):
        self.pbar.update()
        return self.streamer.put(tokens)

class GPUUseStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, device: torch.device, outer: Callable[[int], None]):
        super().__init__(streamer)
        self.device = device
        self.outer = outer
        self.outer(torch.cuda.max_memory_allocated(self.device))
    def put(self, tokens: list[int]):
        self.outer(torch.cuda.max_memory_allocated(self.device))
        return self.streamer.put(tokens)

class DummyStreamer(Streamer):
    def __init__(self, voc_size: int, token_range_size: int, n_gen: int):
        self.position = 0
        self.voc_size = voc_size
        self.token_range = list(range(voc_size))
        self.n_gen = n_gen
        self.i_gen = 0

    def put(self, tokens: list[int]) -> tuple[bool, int, list[int]]:
        self.position += len(tokens)
        self.i_gen += 1
        return self.i_gen <= self.n_gen, self.position, self.token_range


# Ligands
MOL_ERRORS = {
    (ValueError, r"could not convert string to float: '.+'"): 'COORD_NOT_FLOAT', 
    (ValueError, r"SMILES is invalid\."): 'SMILES', 
    (ValueError, r"SMILES is invalid: .+"): 'SMILES', 
    (ValueError, r"SMILES mismatch\."): 'SMILES_MISMATCH',

}

class LigandStreamer(Streamer):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, mol_tokenizer: MolTokenizer, voc_encoder: VocEncoder, end_token: str, cls: Literal['rdkit', 'ob']):
        self.mol_tokenizer = mol_tokenizer
        self.voc_encoder = voc_encoder
        self.is_init = True
        self.end_token = end_token
        self.cls = cls

        self._ligand = None
        self._error = 'PARSE_NOT_ENDED'

    def put(self, tokens: list[int]) -> tuple[bool, int, list[int]]:
        if self.is_init:
            pos_offset = len(tokens)
            stream = self.mol_tokenizer.decode_stream(self.end_token, 'rdkit')
            stream = pos_offset_stream(stream, pos_offset)
            self.stream = encode_token_stream(stream, self.voc_encoder)
            token_range, position = next(self.stream)
            self.is_init = False
            return True, position, token_range
        else:
            assert len(tokens) == 1
            try:
                token_range, position = self.stream.send(tokens[0])
                return True, position, token_range
            except StopIteration as e:
                ligand, self._atom_poss, self._coord_posss = e.value
                self._ligand = ligand
                
                self._error = None
                return False, None, None
            except Exception as e:
                for (etype, epattern), ename in MOL_ERRORS.items():
                    if isinstance(e, etype) and re.match(epattern, e.args[0]):
                        self._error = ename
                        break
                else:
                    self.logger.warning(f"Unknown error: {type(e).__name__}{e.args}")
                    self._error = 'UNK_ERROR'
                return False, None, None

    def ligand(self) -> str|None:
        return self._ligand
    def error(self) -> str|None:
        return self._error
    def atom_poss(self) -> tuple[list[int], list[int]]:
        """
        Returns
        -------
        atom_poss: 
            self.ligand() の i 番目の原子が, atom_poss[i] 番目のトークンで表されている
        coord_poss:
            同じく。座標について
        """
        return self._atom_poss, self._coord_posss

class SaveLigandStreamer(WrapperStreamer):
    def __init__(self, streamer: LigandStreamer, new_sdf_path: str):
        super().__init__(streamer)
        self.streamer = streamer
        self.new_sdf_path = new_sdf_path
    
    def put(self, tokens: list[int]):
        output = self.streamer.put(tokens)
        mol = self.streamer.ligand()
        if mol is not None:
            with open(self.new_sdf_path, 'w') as f:
                f.write(Chem.MolToMolBlock(mol))

        return output
