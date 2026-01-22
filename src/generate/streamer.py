import os
from time import time
from collections.abc import Generator
from logging import getLogger
import numpy as np
from src.utils import should_show
from src.utils.path import make_pardir
from src.model import Streamer
from src.data.tokenizer import FloatTokenizer, VocEncoder

def coord_streamer(n_atom: int, start_position: int, new_coord_path: str|None, voc_encoder: VocEncoder, coord_range: float, no_token_range: bool, atom_order: bool, center: np.ndarray|None) -> Generator[tuple[bool, list[int], list[int]]]:
    coord_tokenizer = FloatTokenizer('', -coord_range, coord_range)
    if no_token_range:
        int_token_range = frac_token_range = list(range(voc_encoder.voc_size))
    else:
        int_token_range = sorted(voc_encoder.encode(coord_tokenizer.int_vocs()))
        frac_token_range = sorted(voc_encoder.encode(coord_tokenizer.frac_vocs()))
    pos = start_position
    if new_coord_path is not None:
        make_pardir(new_coord_path)
        with open(new_coord_path, 'w') as f:
            f.write("idx,x,y,z\n")
    coordss = []
    for i_atom in range(n_atom):
        coords = []
        for dim in range(3):
            int_token = yield True, [pos], int_token_range
            frac_token = yield True, [pos+1], frac_token_range
            pos += 2
            coord_str = ''.join(voc_encoder.decode(int_token+frac_token))
            try:
                coord = float(coord_str)
            except Exception:
                return None, pos
            if center is not None:
                coord += center[dim]
            coords.append(coord)
        if atom_order:
            pos += 1
        if new_coord_path is not None:
            with open(new_coord_path, 'a') as f:
                f.write(f"{i_atom},{coords[0]},{coords[1]},{coords[2]}\n")
        coordss.append(coords)
    return np.array(coordss), pos

class GeneratorStreamer(Streamer):
    logger = getLogger(f"{__qualname__}")
    def __init__(self, name: str, prompt_token_path: str, new_token_path: str, voc_encoder: VocEncoder):
        self.name = name
        self.voc_encoder = voc_encoder
        self.prompt_token_path = prompt_token_path
        self.new_token_path = new_token_path
        self.mean_dt = None
        
        self.put_gen = self.put_generator()
        self.is_prompt = True
        self.new_token_dir_made = False
        self.n = 0
        next(self.put_gen)
    def estimated_n_token(self):
        return None
    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        raise NotImplementedError
    def put(self, token: list[int]) -> tuple[bool, list[int], list[int]]:
        if self.is_prompt:
            os.makedirs(os.path.dirname(self.prompt_token_path), exist_ok=True)
            with open(self.prompt_token_path, 'w') as f:
                f.write(' '.join(self.voc_encoder.decode(token))+'\n')
            self.start = self.init = time()
            self.is_prompt = False
        else:
            if not self.new_token_dir_made:
                os.makedirs(os.path.dirname(self.new_token_path), exist_ok=True)
                self.new_token_dir_made = True
            with open(self.new_token_path, 'a') as f:
                f.write(self.voc_encoder.i2voc[token[0]]+' ')
            dt = (end:=time()) - self.start
            self.mean_dt = dt if self.mean_dt is None else self.mean_dt*0.9+dt*0.1
            self.start = end
            self.n += 1
            if should_show(self.n) and (t:=self.start-self.init) >= 1.0:
                est_n = self.estimated_n_token()
                if est_n is None:
                    self.logger.info(f"[{self.name}]generated {self.n}/? token in {t:.02f}s")
                else:
                    est_t = self.mean_dt*(est_n-self.n)
                    self.logger.info(f"[{self.name}]generated {self.n}/{est_n} token in {t:02f}s (estimated to end in {est_t:.02f}s)")
        return self.put_gen.send(token)
