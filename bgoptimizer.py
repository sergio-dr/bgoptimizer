# %%
"""
bgoptimizer -- Astrophotography background modeling/subtraction via spline gradient descent optimization

Copyright (C) 2021-2022 Sergio Díaz, sergiodiaz.eu

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import argparse
import os 
from xisf import XISF
import numpy as np
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show errors
import tensorflow as tf
from tensorflow import keras

from bgmodel import BgModel
from spline import Spline
from imageprocessor import ImageProcessor

# For visualization only
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

DEBUG_CMDLINE = None
#DEBUG_CMDLINE = "in\\orion\\O3.xisf out2 -dq 1.0 -p".split(" ") 


bg_model_fits_comment = "Background-subtracted with github.com/sergio-dr/bgoptimizer"


help_desc = """
Generates a spline based background model (by gradient descent optimization) for the input (linear) image. 
Both the background-subtracted image and background model are written to the specified output directory 
(appending '_bgSubtracted' and 'bgModel' suffixes to the original filename). A mask can be specified by
providing a grayscale mask image, or by providing a (min, max) threshold range (outside this range, 
pixels are masked). Fully masked pixels are ignored during optimization, so masks are helpful, e.g., for 
ignorig missing values after registration, or very bright regions, when fitting the spline. Only XISF files
are supported. 
"""

config_defaults = {
    'out_dirpath': '.',
    'downscaling_factor': 8, 'downscaling_func': 'median',
    'delinearization_quantile': 0.95,
    'N': 32,
    'O': 2,
    'threshold_min': 0.001, 
    'threshold_max': 1.0,
    'mask': None,
    'initializer': 'random', 
    'alpha': 5,    
    'B': 1,
    'lr': 0.001,
    'epochs': 1000,
}

parser = argparse.ArgumentParser(description=help_desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input_file", 
                    help="Input filename. Must be in XISF format, in linear state")
parser.add_argument("output_path", default=config_defaults['out_dirpath'], nargs='?',
                    help="Path for the output files (by default, the current directory)")
parser.add_argument("-dx", "--downscaling-factor", type=int, default=config_defaults['downscaling_factor'], 
                    help="Image downscaling factor for spline fitting")
parser.add_argument("-df", "--downscaling-func", default=config_defaults['downscaling_func'], 
                    help="Image downscaling function ('median', 'mean') for spline fitting")
parser.add_argument("-dq", "--delinearization-quantile", type=float, default=config_defaults['delinearization_quantile'], 
                    help="Quantile mapped to 1.0 in image delinearization")                  
parser.add_argument("-p", "--preview", action='store_true',
                    help="Don't fit the spline, just preview delinearization and masking") 
parser.add_argument("-N", default=config_defaults['N'], type=int, 
                    help="Number of control points of the spline")
parser.add_argument("-O", default=config_defaults['O'], type=int, 
                    help="Order of the spline (2=thin-plate; 3=bicubic)")
parser.add_argument("-tm", "--threshold-min", type=float, default=config_defaults['threshold_min'], 
                    help="A mask can be defined by giving a (min, max) range")
parser.add_argument("-tM", "--threshold-max", type=float, default=config_defaults['threshold_max'], 
                    help="A mask can be defined by giving a (min, max) range") 
parser.add_argument("-m", "--mask", type=str, default=config_defaults['mask'], 
                    help="Path to a mask file (grayscale)")
parser.add_argument("-i", "--initializer", default=config_defaults['initializer'], 
                    help="The spline fitting could be initialized with 'random' train points, or arranging them in a 'grid'")
parser.add_argument("-e", "--epochs", type=int, default=config_defaults['epochs'], 
                    help="Maximum number of epochs (iterations) for the optimization process")
parser.add_argument("-a", "--alpha", type=float, default=config_defaults['alpha'], 
                    help="[Advanced] Factor that multiplies the 'negative background' and 'overshoot' penalties in the loss function")
parser.add_argument("-B", type=int, default=config_defaults['B'], 
                    help="[Advanced] Batch size for the optimization process")
parser.add_argument("-lr", type=float, default=config_defaults['lr'], 
                    help="[Advanced] Learning rate for the optimization process")
args = parser.parse_args(DEBUG_CMDLINE)
config = vars(args)


# %%
config['threshold'] = (config.pop('threshold_min'), config.pop('threshold_max'))

in_filepath = os.path.abspath(config['input_file'])
out_dirpath = os.path.abspath(config['output_path'])

in_name, ext = os.path.splitext( os.path.basename(in_filepath) )
out_filename = in_name + "_bgSubtracted" + ext
out_filepath = os.path.join(out_dirpath, out_filename)
bg_filename = in_name + "_bgModel" + ext
bg_filepath = os.path.join(out_dirpath, bg_filename)

print("\n\n__/ Arguments \__________")
arg_print_format = "%-24s: %s"
for key, value in config.items():
  print(arg_print_format % (key, value))


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



# %%

print("\n\n__/ Preprocessing \__________")
# In the following: fr=full resolution, lin/nl = linear/non-linear

# Read the input image
xisf = XISF(in_filepath)
im_fr_lin = xisf.read_image(0)

# Preprocess the image (delinearize and downscale)
improc = ImageProcessor(config)
im_ds_nl = improc.fit_transform(im_fr_lin)

# Get mask
if config['mask']:
  mask = improc.downscale( XISF.read(config['mask']) )
  assert mask.shape == im_ds_nl.shape, f"Mask should have the same shape as the input image: {im_ds_nl.shape}"
  config['npmask'] = mask
else:
  mask = Spline._generate_mask(im_ds_nl, config['threshold'])

# Show image (masked)
improc.plot_image_hist(im_ds_nl * mask, "Delinearized, downscaled, masked")


# %%

# Fit the background model on the delinearized, downsized version of the image
print("\n\n__/ Background modeling \__________")

if config['preview']:
  print("Skipping spline fit, --preview requested.")
  plt.show()
  exit(0)

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
#improc.plot_image_hist(improc._delinearize(im_hat_fr_lin.copy(), 0.25), 'Bg-subtracted image preview')


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