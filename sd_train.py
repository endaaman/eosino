import os
from types import SimpleNamespace

os.environ['AUTOGRAPH_VERBOSITY'] = '1'
import tensorflow as tf
from csbdeep.utils.tf import limit_gpu_memory
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from stardist import gputools_available
from stardist.models import Config2D, StarDist2D

from conic import get_data, oversample_classes, CLASS_NAMES
from conic import HEStaining, HueBrightnessSaturation

from augmend import (
    Augmend,
    AdditiveNoise,
    Elastic,
    FlipRot90,
    GaussianBlur,
    Identity,
)


# you may need to adjust this to your GPU needs and memory capacity
# os.environ['CUDA_VISIBLE_DEVICES'] = ...
limit_gpu_memory(0.8, total_memory=24000)
# limit_gpu_memory(None, allow_growth=True)

def get_class_count(Y0):
    class_count = np.bincount(Y0[:,::4,::4,1].ravel())
    df = pd.DataFrame(class_count, index=CLASS_NAMES.values(), columns=["counts"])
    df = df.drop("BACKGROUND")
    df["%"] = (100 * (df["counts"] / df["counts"].sum())).round(2)
    print(df)
    return class_count

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

print(vars(args))


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
