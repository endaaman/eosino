import os
from types import SimpleNamespace

import numpy as np
from csbdeep.utils.tf import limit_gpu_memory
import pandas as pd
from sklearn.model_selection import train_test_split

from stardist import gputools_available
from stardist.models import Config2D, StarDist2D

from conic import get_data, oversample_classes, CLASS_NAMES
from conic import HEStaining, HueBrightnessSaturation
from augmend import (
    Augmend,
    AdditiveNoise,
    Augmend,
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
    display(df)
    return class_count
