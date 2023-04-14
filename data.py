import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import cv2

from utils import remap_label

mpl.rcParams['figure.dpi'] = 120 # for high resolution figure in notebook
# Data Reading
# First, let's provide the path to all data provided as part of the challenge.

data_root = "./data" #! Change this according to the root path where the data is located

images_path = "%s/images.npy" % data_root # images array Nx256x256x3
labels_path = "%s/labels.npy" % data_root # labels array Nx256x256x3
counts_path = "%s/counts.csv" % data_root # csv of counts per nuclear type for each patch
info_path = "%s/patch_info.csv" % data_root # csv indicating which image from Lizard each patch comes from
# Now let's load in each of the files!

images = np.load(images_path)
labels = np.load(labels_path)
counts = pd.read_csv(counts_path)
patch_info = pd.read_csv(info_path)
# Let's take a look at the dimensions of the path data

print("Images Shape:", images.shape)
print("Labels Shape:", labels.shape)

# Images Shape: (4981, 256, 256, 3)
# Labels Shape: (4981, 256, 256, 2)
# As can be seen, both arrays have 4 dimensions. Specifically, the dimensions are NxHxWxC. Here, N is the number of patches,
# H and W are the patch dimensions and C is the number of channels. Therefore, in CoNIC we provide 4981 patches.
# The imges are RGB (hence 3 channels), whereas for the labels, the first channel is the instance map and the second channel is the classification map.
# Let's visualise a few patches along with the coresponding ground truth.

# this patch can be repeatedly executed to visualise a different patch!
rand_idx = np.random.randint(0, images.shape[0]) # select a random patch
patch_img = images[rand_idx] # 256x256x3
patch_lab = labels[rand_idx] # 256x256x2

# separate the instance map and classification map
patch_inst_map = patch_lab[..., 0]
patch_class_map = patch_lab[..., 1]

# visualise the data in a single plot
viz_dict = {"Image": patch_img, "Instance Map": patch_inst_map, "Classification Map": patch_class_map}
fig = plt.figure(figsize=(7,10))
count = 1
for img_name, img in viz_dict.items():
    ax = plt.subplot(1,3, count)
    plt.imshow(img)
    plt.title(img_name)
    plt.axis("off")
    count += 1
