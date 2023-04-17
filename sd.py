import os
from types import SimpleNamespace

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
args.modeldir    = "./models"
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
