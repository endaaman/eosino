import numpy as np

def get_class_color_map(alpha=255):
    return np.array([
        (  0,   0,   0, 0),
        (255, 165,   0, alpha),
        (  0, 255,   0, alpha),
        (255,   0,   0, alpha),
        (  0, 255, 255, alpha),
        (  0,   0, 255, alpha),
        (255, 255,   0, alpha),
    ], dtype=np.uint8)


def get_cell_color_map(size=1000, alpha=255, seed=1):
    np.random.seed(seed)
    rgb_frame = np.random.randint(low=0, high=256, size=(size, 4))
    rgb_frame[0, :] = 0
    rgb_frame[:, 3] = alpha
    return rgb_frame
