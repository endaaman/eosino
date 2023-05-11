import os
import os.path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from imageio import imread
from pydantic import Field
from csbdeep.utils.tf import limit_gpu_memory
from stardist import gputools_available
from stardist.models import Config2D, StarDist2D
from stardist.plot import random_label_cmap, render_label
from augmend import (
    Augmend,
    AdditiveNoise,
    Elastic,
    FlipRot90,
    GaussianBlur,
    Identity,
)

from endaaman import load_images_from_dir_or_file
from endaaman.ml import BaseMLCLI, BaseMLArgs

from conic import predict, CLASS_NAMES
from conic import get_data, oversample_classes, CLASS_NAMES
from conic import HEStaining, HueBrightnessSaturation
from common import get_class_color_map, array_twice_size


cmap_random = random_label_cmap()
J = os.path.join


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        seed:int = 42

def get_class_count(Y0):
    class_count = np.bincount(Y0[:,::4,::4,1].ravel())
    df = pd.DataFrame(class_count, index=CLASS_NAMES.values(), columns=["counts"])
    df = df.drop("BACKGROUND")
    df["%"] = (100 * (df["counts"] / df["counts"].sum())).round(2)
    print(df)
    return class_count



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        dest:str = 'out/sd'

    def pre_common(self, a:CommonArgs):
        pass

    def run_train(self, a:CommonArgs):
        # you may need to adjust this to your GPU needs and memory capacity
        # os.environ['CUDA_VISIBLE_DEVICES'] = ...
        limit_gpu_memory(0.8, total_memory=24000)
        # limit_gpu_memory(None, allow_growth=True)
        args = SimpleNamespace()
        # data in
        args.datadir     = "./datasets/CoNIC" # path to 'Patch-level Lizard Dataset' as provided by CoNIC organizers
        args.oversample  = True     # oversample training patches with rare classes
        args.frac_val    = 0.1      # fraction of data used for validation during training
        args.seed        = None     # for reproducible train/val data sets
        # model out (parameters as used for our challenge submissions)
        args.modeldir    = "./out/sd"
        args.epochs      = 1000
        args.batchsize   = 4
        args.n_depth     = 4
        args.lr          = 3e-4
        args.patch       = 256
        args.n_rays      = 64
        args.grid        = (1,1)
        args.head_blocks = 2
        args.augment     = True
        args.cls_weights = False
        args.workers     = 1
        args.gpu_datagen = False and args.workers==1 and gputools_available() # note: ignore potential scikit-tensor error

        X, Y, D, Y0, idx = get_data(args.datadir, seed=args.seed)
        X, Xv, Y, Yv, D, Dv, Y0, Y0v, idx, idxv = train_test_split(X, Y, D, Y0, idx, test_size=args.frac_val, random_state=args.seed)
        class_count = get_class_count(Y0)

        if args.oversample:
            X, Y, D, Y0, idx = oversample_classes(X, Y, D, Y0, idx, seed=args.seed)
            class_count = get_class_count(Y0)

        if args.cls_weights:
            inv_freq = np.median(class_count) / class_count
            inv_freq = inv_freq ** 0.5
            class_weights = inv_freq.round(4)
        else:
            class_weights = np.ones(len(CLASS_NAMES))

        if args.augment:
            aug = Augmend()
            aug.add([HEStaining(amount_matrix=0.15, amount_stains=0.4), Identity()], probability=0.9)
            aug.add([FlipRot90(axis=(0,1)), FlipRot90(axis=(0,1))])
            aug.add([Elastic(grid=5, amount=10, order=1, axis=(0,1), use_gpu=False),
                     Elastic(grid=5, amount=10, order=0, axis=(0,1), use_gpu=False)], probability=0.8)
            aug.add([GaussianBlur(amount=(0,2), axis=(0,1), use_gpu=False), Identity()], probability=0.1)
            aug.add([AdditiveNoise(0.01), Identity()], probability=0.8)
            aug.add([HueBrightnessSaturation(hue=0, brightness=0.1, saturation=(1,1)), Identity()], probability=0.9)

            def augmenter(x,y):
                return aug([x,y])
        else:
            augmenter = None

        conf = Config2D(
            n_rays                = args.n_rays,
            grid                  = args.grid,
            n_channel_in          = X.shape[-1],
            n_classes             = len(CLASS_NAMES)-1,
            use_gpu               = args.gpu_datagen,
            backbone              = 'unet',
            unet_n_filter_base    = 64,
            unet_n_depth          = args.n_depth,
            head_blocks           = args.head_blocks,
            net_conv_after_unet   = 256,
            train_batch_size      = args.batchsize,
            train_patch_size      = (args.patch, args.patch),
            train_epochs          = args.epochs,
            train_steps_per_epoch = 1024 // args.batchsize,
            train_learning_rate   = args.lr,
            train_loss_weights    = (1.0, 0.2, 1.0),
            train_class_weights   = class_weights.tolist(),
            train_background_reg  = 0.01,
            train_reduce_lr       = {'factor': 0.5, 'patience': 80, 'min_delta': 0},
        )
        vars(conf)
        model = StarDist2D(conf, name='conic', basedir=args.modeldir)
        model.train(X, Y, classes=D, validation_data=(Xv, Yv, Dv), augmenter=augmenter, workers=args.workers)
        model.optimize_thresholds(Xv, Yv, nms_threshs=[0.1, 0.2, 0.3])


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
