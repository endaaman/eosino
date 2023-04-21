import os
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import matplotlib
from imageio import imread
from stardist.models import StarDist2D
from stardist.plot import random_label_cmap, render_label

from pydantic import Field
from endaaman import load_images_from_dir_or_file
from endaaman.ml import BaseCLI, define_ml_args

from conic import predict, CLASS_NAMES
from common import get_color_map


cmap_random = random_label_cmap()
J = os.path.join


class CLI(BaseCLI):
    class CommonArgs(define_ml_args(seed=42)):
        dest:str = 'out/sd'

    def pre_common(self, a:CommonArgs):
        pass

    def predict_image(self, model, image, path):
        x = np.array(image)
        # x = x[:256, :256, :]
        u, count = predict(model, x,
            normalize         = True,
            test_time_augment = True,
            tta_merge         = dict(prob=np.median, dist=np.mean, prob_class=np.mean),
            refine_shapes     = {},
            crop_counts       = False,
        )
        return count, u

    class PredArgs(CommonArgs):
        src:str
        name:str = '{}'

    def run_pred(self, a:PredArgs):
        model = StarDist2D(None, name='conic', basedir='out/sd')
        images, paths = load_images_from_dir_or_file(a.src, with_path=True)
        MAP = get_color_map(alpha=150)

        for image, path in zip(images, paths):
            image = image.resize((image.width//2, image.height//2))
            count, mask = self.predict_image(model, image, path)
            cell_mask = mask[:, :, 0]
            class_mask = mask[:, :, 1]
            name = a.name.format(os.path.splitext(os.path.basename(path))[0])
            class_mask_image = Image.fromarray(MAP[class_mask])
            overlay_image = image.convert('RGBA').copy()
            overlay_image.paste(class_mask_image, (0, 0), class_mask_image)
            overlay_image = overlay_image.convert('RGB')

            image.save(J(a.dest, f'{name}.jpg'))
            class_mask_image.save(J(a.dest, f'{name}_mask.png'))
            overlay_image.save(J(a.dest, f'{name}_overlay.jpg'))

            np.save(J(a.dest, f'{name}_mask'), mask)

            c = {name: i for (name, i) in zip(list(CLASS_NAMES.values())[1:], count)}
            with open(J(a.dest, f'{name}.json'), 'w') as f:
                json.dump(c, f)



if __name__ == '__main__':
    cli = CLI()
    cli.run()
