import os
from logging import getLogger
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Geometry import Point3D
from openbabel.openbabel import OBMol, OBMolAtomIter
from .data import WrapDataset, WrapTupleDataset, get_rng
from .protein import Pocket, get_coord_from_mol

class Scaler:
    def __init__(self, from_a: float, from_b: float, to_a: float, to_b: float):
        self.from_a = from_a
        self.from_b = from_b
        self.to_a = to_a
        self.to_b = to_b
        assert self.from_b - self.from_a > 0
        assert self.to_b > self.to_a
    def scale(self, from_v: float|np.ndarray|torch.Tensor):
        return (from_v-self.from_a) / (self.from_b-self.from_a) \
            * (self.to_b - self.to_a) + self.to_a
    
    def rescale(self, to_v: float|np.ndarray|torch.Tensor):
        return (to_v-self.to_a) / (self.to_b-self.to_a) \
            * (self.from_b - self.from_a) + self.from_a
    def __str__(self):
        return f"Scaler([{self.from_a}, {self.from_b}]->[{self.to_a}, {self.to_b}])"


class RescaleDataset(WrapDataset[float]):
    unk_logger = getLogger(f'unk.{__module__}.{__qualname__}')
    def __init__(self, dataset: Dataset[float], from_a: float, from_b: float, to_a: float, to_b: float):
        super().__init__(dataset)
        self.scaler = Scaler(from_a, from_b, to_a, to_b)
    def __getitem__(self, idx: int):
        return self.scaler.scale(self.dataset[idx])

class CoordTransformDataset(WrapTupleDataset[np.ndarray]):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, base_data: Dataset[Chem.Mol|Pocket|OBMol], *datas: 
                tuple[Dataset[Chem.Mol|Pocket|OBMol]], base_seed: 
                int=0, normalize_coord=False, random_rotate=False, 
                coord_noise_std=0.0):
        self.normalize_coord = normalize_coord
        self.random_rotate = random_rotate
        self.datas = (base_data,)+(datas)
        self.base_seed = base_seed
        self.coord_noise_std = coord_noise_std

        # Initialize TupleDataset
        tuple_size = 1+len(datas) \
            + (1 if self.normalize_coord else 0) \
            + (1 if self.random_rotate else 0)
        super().__init__(base_data, tuple_size)
    
    def __getitem__(self, idx: int) -> tuple[Chem.Mol|Pocket|OBMol]:
        items = [data[idx] for data in self.datas]
        coords = []
        for item in items:
            if isinstance(item, Chem.Mol):
                coord = item.GetConformer().GetPositions()
            elif isinstance(item, Pocket):
                coord = item.coord
            elif isinstance(item, OBMol):
                coord = item.GetCoordinates()
                coord = get_coord_from_mol(item)
            else:
                raise ValueError
            coords.append(coord)
        rng = get_rng(self.base_seed, idx)

        # normalize
        if self.normalize_coord:
            if coords[0].size > 0:
                center = np.mean(coords[0], axis=0)
            else:
                if len(coords) > 1:
                    raise ValueError("base_coord.size=0 and cannot normalized")
            coords = [coord - center for coord in coords]

        # random rotate
        if self.random_rotate:
            matrix = get_random_rotation_matrix(rng)
            coords = [np.matmul(coord, matrix) for coord in coords]

        # add noise
        if self.coord_noise_std > 0:
            noise = rng.normal(size=3, scale=self.coord_noise_std)   
            coords = [coord+noise  for coord in coords]
        
        # set coord
        for item, coord in zip(items, coords):
            if isinstance(item, Chem.Mol):
                # confへの代入のみで元の分子も変更されることを確認 @tests/test.ipynb
                conf = item.GetConformer()
                for i in range(len(coord)):
                    conf.SetAtomPosition(i, Point3D(*coord[i]))
            elif isinstance(item, Pocket):
                item.coord = coord
            else:
                for i, atom in enumerate(OBMolAtomIter(item)):
                    atom.SetVector(coord[i,0], coord[i,1], coord[i,2])
        if self.normalize_coord:
            items.append(center)
        if self.random_rotate:
            items.append(matrix)
        return tuple(items)

def get_random_rotation_matrix(rng: np.random.Generator):
    # get axes
    axes = []
    while(len(axes) < 2):
        new_axis = rng.random(3)
        
        new_norm = np.sqrt(np.sum(new_axis**2))
        if (new_norm < 0.1 or 1 <= new_norm): continue
        new_axis = new_axis / new_norm
        if np.any([np.abs(np.sum(axis*new_axis)) >= 0.9 for axis in axes]):
            continue
        axes.append(new_axis)

    # get rotation matrix
    axis0, axis1b = axes
    axis1 = np.cross(axis0, axis1b)
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = np.cross(axis0, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    return np.array([axis0, axis1, axis2])
