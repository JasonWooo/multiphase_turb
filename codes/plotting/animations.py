"""
Make animations
"""

### import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.markers import MarkerStyle
import matplotlib.patheffects as pe
import cmasher as cmr
from tqdm import tqdm
from codes.funcs import *  # import everything in functions
from codes.timescales import *
import codes.timescales_8e3 as ts8e3
mpl.rcParams.update(mpl.rcParamsDefault)
from codes.jason import plotting_def, plot_prettier
plotting_def()


def concat_anim(fpath = '/ptmp/mpa/wuze/multiphase_turb/figures/animations/240709_0.6_16000',
                frame_rate = 30):
    """
    Concatenate a collection of frames into an animation
    """

    import cv2
    import os

    vidpath = f'{fpath}.avi'

    # Get the list of images in the folder
    images = [img for img in os.listdir(fpath) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Sort the images by name to maintain the correct order

    # Read the first image to get the dimensions
    first_image_path = os.path.join(fpath, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use 'MJPG' or other codecs
    out = cv2.VideoWriter(vidpath, fourcc, frame_rate, (width, height))

    for image in images:
        img_path = os.path.join(fpath, image)
        frame = cv2.imread(img_path)
        out.write(frame)  # Write out frame to video

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {vidpath}")