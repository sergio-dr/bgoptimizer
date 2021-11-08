# %%
import argparse
import os 
from xisf import XISF
import numpy as np
import tensorflow as tf
from tensorflow import keras

from bgmodel import BgModel
from spline import Spline
from imageprocessor import ImageProcessor

# For visualization only
import matplotlib.pyplot as plt

# For experiments only
import pandas as pd 

np.set_printoptions(precision=4)


bg_model_fits_comment = "Background-subtracted with github.com/sergio-dr/bg-model"


help_desc = """
Generates a spline based background model (by gradient descent optimization) for the input (linear) image. 
Both the background-subtracted image and background model are written to the specified output directory 
(appending '_bgSubtracted' and 'bgModel' suffixes to the original filename). A mask can be specified by
giving a (min, max) threshold range (the min value can be helpful to mask out missing values after 
registration, for example; the max value helps to ignore very bright regions when fitting the spline). 
"""

config_defaults = {
    'out_dirpath': '.',
    'downscaling-factor': 8, 'downscaling-func': 'median',
    'delinearization-quantile': 0.95,
    'N': 32,
    'O': 2,
    'threshold': (0.001, 0.95),
    'initializer': 'random', 
    'alpha': 5,    
    'B': 1,
    'lr': 0.001,
    'epochs': 1000,
}

parser = argparse.ArgumentParser(description=help_desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input_file", 
                    help="Input filename. Must be in XISF format, in linear state")
parser.add_argument("output_path", default=config_defaults['out_dirpath'],
                    help="Path for the output files")
parser.add_argument("-dx", "--downscaling-factor", type=int, default=config_defaults['downscaling-factor'], 
                    help="Image downscaling factor for spline fitting")
parser.add_argument("-df", "--downscaling-func", default=config_defaults['downscaling-func'], 
                    help="Image downscaling function ('median', 'mean') for spline fitting")
parser.add_argument("-dq", "--delinearization-quantile", type=float, default=config_defaults['delinearization-quantile'], 
                    help="Quantile mapped to 1.0 in image delinearization")                  
parser.add_argument("-N", default=config_defaults['N'], type=int, 
                    help="Number of control points of the spline")
parser.add_argument("-O", default=config_defaults['O'], type=int, 
                    help="Order of the spline (2=thin-plate; 3=bicubic)")
parser.add_argument("-tm", "--threshold-min", type=float, default=config_defaults['threshold'][0], 
                    help="A mask can be defined by giving a (min, max) range")
parser.add_argument("-tM", "--threshold-max", type=float, default=config_defaults['threshold'][1], 
                    help="A mask can be defined by giving a (min, max) range") 
parser.add_argument("-i", "--initializer", default=config_defaults['initializer'], 
                    help="The spline fitting could be initialized with 'random' train points, or arranging then in a 'grid'")
parser.add_argument("-a", "--alpha", type=float, default=config_defaults['alpha'], 
                    help="[Advanced] Factor that multiplies the 'negative background' and 'overshoot' penalties in the loss function")
parser.add_argument("-B", type=int, default=config_defaults['B'], 
                    help="[Advanced] Batch size for the optimization process")
parser.add_argument("-lr", type=float, default=config_defaults['lr'], 
                    help="[Advanced] Learning rate for the optimization process")
parser.add_argument("-e", "--epochs", type=int, default=config_defaults['epochs'], 
                    help="[Advanced] Maximum number of epochs for the optimization process")

args = parser.parse_args()
config = vars(args)

# %%
config['threshold'] = (config.pop('threshold_min'), config.pop('threshold_max'))

in_filepath = os.path.abspath(args.input_file)
out_dirpath = os.path.abspath(args.output_path)

in_name, ext = os.path.splitext( os.path.basename(in_filepath) )
out_filename = in_name + "_bgSubtracted" + ext
out_filepath = os.path.join(out_dirpath, out_filename)
bg_filename = in_name + "_bgModel" + ext
bg_filepath = os.path.join(out_dirpath, bg_filename)

print("\n\n__/ Arguments \__________")
arg_print_format = "%-24s: %s"
print(arg_print_format % ("Input file", in_filepath))
print(arg_print_format % ("Output path", out_dirpath))
for key, value in config.items():
  print(arg_print_format % (key, value))
print("\n")


# __/ Environment \__________

import platform
import tensorflow_addons as tfa
import skimage

print("\n\n__/ Environment \__________")
print("python:", platform.python_version())
print("tensorflow:", tf.__version__)
print("tensorflow_addons:", tfa.__version__)
print("keras:", keras.__version__)
print("numpy:", np.__version__)
print("skimage:", skimage.__version__)
print("\n")



# %%

# In the following: fr=full resolution, lin/nl = linear/non-linear

# Read the input image
xisf = XISF(in_filepath)
im_fr_lin = xisf.read_image(0)


# Preprocess the image (delinearize and downscale)
print("\n\n__/ Preprocessing \__________")
improc = ImageProcessor(config)
im_ds_nl = improc.fit_transform(im_fr_lin)
improc.plot_image_hist(im_ds_nl * Spline._generate_mask(im_ds_nl, config['threshold']), "Delinearized, downscaled, masked")


# Fit the background model on the delinearized, downsized version of the image
print("\n\n__/ Background modeling \__________")
bgmodel = BgModel(config)
bg_hat_ds_nl = bgmodel.fit_transform(im_ds_nl)

#   Training results
bgmodel.training_report()
bgmodel.spline_layer.plot_train_points()

#   Show fitted background model
improc.plot_image_hist(bg_hat_ds_nl, "Fitted background model")

#   Preview subtracted result
im_hat_ds_nl = im_ds_nl-bg_hat_ds_nl+improc.im_fr_nl_median
improc.plot_image_hist(im_hat_ds_nl, "Background-subtracted (downsized, delinearized)")


# Generate background model at full res, linearize it and subtract to the original image
print("\n\n__/ Full-size background model & subtracted image \__________")
bg_hat_fr_nl = bgmodel.interpolate_to(im_fr_lin.shape)
bg_hat_fr_lin = improc.inverse_transform(bg_hat_fr_nl)
im_hat_fr_lin = improc.subtract_safe(im_fr_lin, bg_hat_fr_lin)

#   Preview background-subtracted image
fig = plt.figure(figsize=(16,10))
plt.imshow(improc._delinearize(im_hat_fr_lin.copy(), 0.25)[...,0], cmap='gray')
fig.show()

# %%
# Write background-subtracted image and the background model to file
print("\n\n__/ Saving output files \__________")
os.makedirs(out_dirpath, exist_ok=True)

metadata = xisf.get_images_metadata()[0]
metadata['FITSKeywords'].setdefault('COMMENT', []).append({'value':'', 'comment': bg_model_fits_comment})

print(f"Writing {out_filepath}... ")
XISF.write(out_filepath, im_hat_fr_lin, metadata, xisf.get_file_metadata())
print("done.")

print(f"Writing {bg_filepath}... ")
XISF.write(bg_filepath, bg_hat_fr_lin)
print("done.")

plt.show()