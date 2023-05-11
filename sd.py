import os
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from imageio import imread
from stardist.models import StarDist2D
from stardist.plot import random_label_cmap, render_label

from pydantic import Field
from endaaman import load_images_from_dir_or_file
from endaaman.ml import BaseMLCLI

from conic import predict, CLASS_NAMES
from common import get_class_color_map, array_twice_size


cmap_random = random_label_cmap()
J = os.path.join


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        seed:int = 42
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
        MAP = get_class_color_map(alpha=150)

        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', 32)
        names = list(CLASS_NAMES.values())[1:]
        for image, path in zip(images, paths):
            image_to_pred = image.resize((image.width//2, image.height//2))
            counts, mask = self.predict_image(model, image_to_pred, path)
            # twice size
            mask = array_twice_size(mask)
            cell_mask = mask[:, :, 0]
            class_mask = mask[:, :, 1]
            name = a.name.format(os.path.splitext(os.path.basename(path))[0])
            class_mask_image = Image.fromarray(MAP[class_mask])
            overlay_image = image.convert('RGBA').copy()
            overlay_image.paste(class_mask_image, (0, 0), class_mask_image)
            overlay_image = overlay_image.convert('RGB')
            draw = ImageDraw.Draw(overlay_image)
            count_text = ' '.join([f'{n}:{v}' for n,v in zip(names, counts)])
            draw.rectangle([(0, 0), (overlay_image.width, 40)], fill=(0, 0, 0))
            draw.text((0, 0), count_text, fill=(255, 255, 255), font=font)

            image.save(J(a.dest, f'{name}.jpg'))
            # class_mask_image.save(J(a.dest, f'{name}_mask.png'))
            overlay_image.save(J(a.dest, f'{name}_overlay.jpg'))

            np.save(J(a.dest, f'{name}_mask'), mask)

            c = dict(zip(names, counts))
            with open(J(a.dest, f'{name}.json'), 'w', encoding='utf-8') as f:
                json.dump(c, f)



if __name__ == '__main__':
    cli = CLI()
    cli.run()
