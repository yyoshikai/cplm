import numpy as np


def is_collided(ax, ay, bx, by, cx, cy, dx, dy, eps):
    """
    線分abと線分cdの衝突を判定する
    """
    return (((bx-ax)*(cy-ay)-(by-ay)*(cx-ax))*((bx-ax)*(dy-ay)-(by-ay)*(dx-ax)) < -eps) \
        & (((dx-cx)*(ay-cy)-(dy-cy)*(ax-cx))*((dx-cx)*(by-cy)-(dy-cy)*(bx-cx)) < -eps)

def get_area(points: np.ndarray, eps: float):
    """
    points: np.ndarray[N, 2]
    """
    total_area = 0.0
    while True:
        n = len(points)
        c = np.arange(n, dtype=int)
        d = (c+1)%n
        for i in range(n):
            
            h, j = (i-1)%n, (i+1)%n
            # hit judge
            a,b = j, h
            if np.any(is_collided(
                points[a,0], points[a,1], 
                points[b,0], points[b,1], 
                points[c,0], points[c,1],
                points[d,0], points[d,1], 
                eps
            )): 
                continue

            total_area += ((points[j,0]-points[i,0])*(points[h,1]-points[i,1]) \
                - (points[j,1]-points[i,1])*(points[h,0]-points[i,0]))/2
            points = np.delete(points, i, axis=0)
            break
        else:
            raise ValueError(f"All edges are hit: {repr(points)}")
        if len(points) <= 2:
            break
    return total_area