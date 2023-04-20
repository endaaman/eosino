import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from imageio import imread
from conic import predict
from stardist.models import StarDist2D
from stardist.plot import random_label_cmap, render_label

np.random.seed(42)
cmap_random = random_label_cmap()

model = StarDist2D(None, name='conic', basedir='out/sd')

X = np.load('./data/images.npy')
x = X[10]

u, count = predict(model, x,
    normalize            = True,
    test_time_augment    = True,
    tta_merge            = dict(prob=np.median, dist=np.mean, prob_class=np.mean),
    refine_shapes        = {},
)

fig, axs = plt.subplots(1, 3, figsize=(14,5))

for ax, title, img in zip(
    axs.ravel(),
    ('input', 'prediction (instances)', 'prediction (class)'),
    (x,
     render_label(u[...,0], img=tf.image.adjust_brightness(x,-0.7)/255, normalize_img=False),
     render_label(u[...,1], img=tf.image.adjust_brightness(x,-0.7)/255, normalize_img=False, cmap=cmap_random),
    )
):
    ax.imshow(img, interpolation=None)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
