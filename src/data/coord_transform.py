import numpy as np

class CoordTransform:
    def __init__(self, seed:int=0, normalize_coord=False, random_rotate=False, coord_noise_std=0.0):
        self.rng = np.random.default_rng(seed)
        self.normalize_coord = normalize_coord
        self.random_rotate = random_rotate
        self.coord_noise_std = coord_noise_std
    
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        if coords.size == 0:
            return coords
        if self.normalize_coord:
            coords = coords - np.mean(coords, axis=0, keepdims=True)
        if self.random_rotate:
            matrix = get_random_rotation_matrix(self.rng)
            coords = np.matmul(coords, matrix)
        if self.coord_noise_std > 0:
            noise = self.rng.normal(size=3, scale=self.coord_noise_std)   
            coords += noise
        return coords

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
