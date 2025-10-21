import os
from logging import getLogger
import numpy as np
from torch.utils.data import Dataset
from .data import WrapDataset, WrapTupleDataset, get_rng, WorkerAggregator

class RescaleDataset(WrapDataset[float]):
    unk_logger = getLogger(f'unk.{__module__}.{__qualname__}')
    def __init__(self, dataset: Dataset[float], from_a: float, from_b: float, to_a: float, to_b: float):
        super().__init__(dataset)
        self.from_a = from_a
        self.from_b = from_b
        self.to_a = to_a
        self.to_b = to_b
        assert self.from_b - self.from_a > 0
        assert self.to_b > self.to_a

    def __getitem__(self, idx: int):
        from_v = self.dataset[idx]
        to_v = (from_v-self.from_a) / (self.from_b-self.from_a) \
            * (self.to_b - self.to_a) + self.to_a
        return to_v

class CoordTransformDataset(WrapTupleDataset[np.ndarray]):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, base_coord_data: Dataset[np.ndarray], *coord_datas: tuple[Dataset[np.ndarray]], base_seed: int=0, normalize_coord=False, random_rotate=False, coord_noise_std=0.0):
        self.normalize_coord = normalize_coord
        self.random_rotate = random_rotate
        self.coord_datas = (base_coord_data,)+(coord_datas)
        self.base_seed = base_seed
        self.coord_noise_std = coord_noise_std

        # Initialize TupleDataset
        tuple_size = 1+len(coord_datas) \
            + (1 if self.normalize_coord else 0) \
            + (1 if self.random_rotate else 0)
        super().__init__(base_coord_data, tuple_size)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray,...]:
        coords = [coord_data[idx] for coord_data in self.coord_datas]
        rng = get_rng(self.base_seed, idx)
        
        # check data_epoch
        # self.logger.info(f"EPOCH={os.environ.get('EPOCH')}")

        # normalize
        if self.normalize_coord:
            if coords[0].size > 0:
                center = np.mean(coords[0], axis=0)
                coords = [coord - center for coord in coords]
            else:
                if len(coords) > 1:
                    raise ValueError("base_coord.size=0 and cannot normalized")

        # random rotate
        if self.random_rotate:
            matrix = get_random_rotation_matrix(rng)
            coords = [np.matmul(coord, matrix) for coord in coords]

        # add noise
        if self.coord_noise_std > 0:
            noise = rng.normal(size=3, scale=self.coord_noise_std)   
            coords = [coord+noise for coord in coords]
        
        if self.normalize_coord:
            coords.append(center)
        if self.random_rotate:
            coords.append(matrix)
        
        return tuple(coords)

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
