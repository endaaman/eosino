import os
import os.path

import torch
import numpy as np
from PIL import Image

from endaaman import load_images_from_dir_or_file
from endaaman.cli import BaseCLI
from endaaman.ml import define_ml_args

from hover_net.models.hovernet.net_desc import HoVerNetExt

J = os.path.join


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

class CLI(BaseCLI):
    class CommonArgs(define_ml_args(seed=42)):
        hoge: int = 32

    def pre_common(self, a:CommonArgs):
        self.model = HoVerNetExt(num_types=7, pretrained_backbone='weights/resnet50-0676ba61.pth')
        # weight = torch.load('weights/hovernet-conic.pth', map_location=lambda storage, loc: storage)
        weight = torch.load('weights/hovernet-conic.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(weight)
        self.model.eval()

    class PredArgs(define_ml_args(seed=42)):
        src: str
        dest: str = 'out'

    def run_pred(self, a):
        ii, pp = load_images_from_dir_or_file(a.src, with_path=True)
        original_img, original_path = ii[0], pp[0]
        original_name = os.path.splitext(os.path.basename(original_path))[0]

        t = torch.from_numpy(np.array(original_img.convert('RGB')))[None, ...].permute(0, 3, 1, 2)
        o = self.model(t)
        m = o['tp'][0].detach().numpy()
        m = np.argmax(m, axis=0)

        MAP = get_color_map(200)
        mask_img = Image.fromarray(MAP[m]).convert('RGBA')

        overlay_img = original_img.convert('RGBA').copy()
        overlay_img.paste(mask_img, (0, 0), mask_img)


        original_img.save(J(a.dest, f'{original_name}_original.png'))
        mask_img.save(J(a.dest, f'{original_name}_mask.png'))
        overlay_img.save(J(a.dest, f'{original_name}_overlay.png'))






if __name__ == '__main__':
    cli = CLI()
    cli.run()
