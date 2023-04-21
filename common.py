import numpy as np

def get_color_map(alpha=255):
    return np.array([
        (  0,   0,   0, 0),
        (255, 165,   0, alpha),
        (  0, 255,   0, alpha),
        (255,   0,   0, alpha),
        (  0, 255, 255, alpha),
        (  0,   0, 255, alpha),
        (255, 255,   0, alpha),
    ], dtype=np.uint8)
