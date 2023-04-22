import os
import os.path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imread
from pydantic import Field
from tqdm import tqdm

# from stardist.models import StarDist2D
# from stardist.plot import random_label_cmap, render_label
from endaaman import load_images_from_dir_or_file
from endaaman.ml import BaseCLI, define_ml_args

# from conic import predict, CLASS_NAMES
from common import get_cell_color_map, array_twice_size

J = os.path.join


def array_to_color_hex(a):
    return '#' + ''.join([f'{v:02X}'[:2] for v in a])

class CLI(BaseCLI):
    class CommonArgs(define_ml_args(seed=42)):

    def pre_common(self, a:CommonArgs):
        pass

    class ImageArgs(CommonArgs):
        dest:str = 'out'

    def run_image(self, a:ImageArgs):
        mask = np.load('data/3/P21-3612_1_vs_1_mask.npy')
        class_mask = mask[:, :, 1]
        for i in range(1, 7):
            cell_mask = mask[:, :, 0].copy()
            cell_mask[class_mask != i] = 0
            M = get_cell_color_map()
            im = M[cell_mask]
            Image.fromarray(im.astype(np.uint8)).save(J(a.dest, f'cell_{i}.png'))

    class M2lArgs(CommonArgs):
        src:str
        dest:str = 'out'

    def run_m2l(self, a:M2lArgs):
        # example = {
        #     'interval': {'min': [0, 0], 'max': [1919, 1079], 'n': 2},
        #     'pixelSizes': [
        #         {'size': 1.00000, 'unit': 'pixel'},
        #         {'size': 1.00000, 'unit': 'pixel'}],
        #     'labels': {'Label 1': [[476, 436], ],
        #                'Label 2': [[514, 356], ] },
        #     'colors': {'Label 1': '#00FF4A', 'Label 2': '#9500FF'}
        # }
        name = os.path.splitext(os.path.basename(a.src))[0]
        base = {
            'interval': {'min': [0, 0], 'max': [1919, 1079], 'n': 2},
            'pixelSizes': [
                {'size': 1.00000, 'unit': 'pixel'},
                {'size': 1.00000, 'unit': 'pixel'}],
            'labels': None,
            'colors': None,
        }

        mask = np.load(a.src)
        mask = array_twice_size(mask)
        class_mask = mask[:, :, 1]
        M = get_cell_color_map(np.max(mask)+10)
        for i in tqdm(range(1, 7)):
            labeling = base.copy()
            # gen labeling by class
            cell_mask = mask[:, :, 0].copy()
            cell_mask[class_mask != i] = 0
            labels = {}
            for y in range(cell_mask.shape[0]):
                for x in range(cell_mask.shape[1]):
                    cell = cell_mask[y, x]
                    if cell == 0:
                        continue
                    label = f'cell{cell}'
                    if label in labels:
                        labels[label].append([x, y])
                    else:
                        labels[label] = [[x, y]]
            colors = {k: array_to_color_hex(M[i+1][:3]) for i, k in enumerate(labels.keys())}
            labeling['labels'] = labels
            labeling['colors'] = colors
            with open(J(a.dest, f'{name}_{i}.labeling'), 'w', encoding='utf-8') as f:
                json.dump(labeling, f)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
