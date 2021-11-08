# %%
import platform
import os
import time
from xisf import XISF
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from spline import Spline

# For preprocessing
import skimage
from skimage.measure import block_reduce

# For visualization only
import matplotlib.pyplot as plt
from skimage import io

# For experiments only
import pandas as pd 

# %%
print("\n\n__/ Environment \__________")

print("python:", platform.python_version())
print("tensorflow:", tf.__version__)
print("tensorflow_addons:", tfa.__version__)
print("keras:", keras.__version__)
print("numpy:", np.__version__)
print("skimage:", skimage.__version__)


# %%

# __/ Script parameters \__________
# TODO: command line args
base_dir = "S:\\src\\bg-model"
in_dir = "in\\pleyades"
in_filename = "masterLight_BINNING_1_FILTER_NoFilter_EXPTIME_120_3_integration_R.xisf" 
in_filepath = os.path.join(base_dir, in_dir, in_filename)
out_dir = "out\\pleyades"
out_filepath = os.path.join(base_dir, out_dir, in_filename)
bg_filepath = os.path.join(base_dir, out_dir, "bg_"+in_filename)

config = {
    'downscaling_factor': 8, 'downscaling_func': 'median',
    'delinearization_quantile': 0.95,
    'N': 32,
    'O': 2,
    'threshold': (0.001, 0.95),
    'initializer': 'random', 
    'alpha': 5,    
    'B': 1,
    'lr': 0.001,
    'epochs': 1000,
}

# %%

# __/ Preprocessing \__________

# min, med, max
def statistics(im, title=""):
    im_min, im_med, im_max = np.nanmin(im), np.nanmedian(im), np.nanmax(im)
    print(f"[{title.ljust(12)}] Min / Median / Max = {im_min:.4f} / {im_med:.4f} / {im_max:.4f}", end='')
    print("  CLIPPING!" if im_min < 0 or im_max > 1 else "")
    return im_min, im_med, im_max


def plot_image_hist(im, title=""):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16,14), gridspec_kw={'height_ratios': [4, 1]})
    _ = ax0.imshow(im[...,0], cmap='gray', vmin=0, vmax=1)
    _ = ax1.hist(im.ravel(), bins=100)
    fig.suptitle(title, fontsize=24)
    fig.tight_layout()


def downscale(data):
    downscaling_func = {
        'median': np.nanmedian,
        'mean': np.nanmean
    }[config['downscaling_func']]

    block_sz = (config['downscaling_factor'], config['downscaling_factor'], 1)
    return block_reduce(data, block_size=block_sz, func=downscaling_func)


# https://pixinsight.com/forum/index.php?threads/auto-histogram-settings-to-replicate-auto-stf.8205/#post-55143
# donde (sea m=bg_val):
#   mtf(0, m, r) = 0
#   mtf(m, m, r) = bg_target_val
#   mtf(1, m, r) = 1
def mtf(x, bg_val, bg_target_val=0.5):
    m, r = bg_val, 1/bg_target_val
    return ( (1-m)*x ) / ( (1-r*m)*x + (r-1)*m ) 


def imtf(y, bg_val, bg_target_val=0.5):
    m, r = bg_val, 1/bg_target_val
    return mtf(y, 1-m, (r-1)/r)


# Mirroring np.nan_to_num
def num_to_nan(data, num=0.0):
    data[data == num] = np.nan


def delinearize(data, bg_target_val=0.25):
    # Subtract pedestal
    pedestal = np.nanmin(data)
    data -= pedestal
    
    # Scale to [0,1] range, mapping the given quantile (instead of the max) to 1.0.
    # This blows out the highlights, but we are interested in the background!
    scale = np.nanquantile(data, q=config['delinearization_quantile']) 
    data /= scale
    data = data.clip(0.0, 1.0)

    # Estimate background value
    bg_val = np.nanmedian(data)

    return mtf(data, bg_val, bg_target_val), pedestal, scale, bg_val


def linearize(data, pedestal, scale, bg_val, bg_target_val=0.25):
    data = imtf(data, bg_val, bg_target_val)
    data *= scale
    data += pedestal
    return data



# %%

# __/ Custom loss \__________
def bg_loss_alpha(y_true, y_pred, model, alpha): # beta=0.1):
    # In this model, y_true is im, y_pred is the generated background model (spline)
    
    # Get mask and bg_val from the spline layer
    spline_layer = model.layers[1]
    mask = spline_layer.mask
    bg_val = spline_layer.bg_val
    
    # Apply mask (like in Spline.build())
    masked_y_true = mask*y_true + (1-mask)*bg_val
    
    # Residuals
    r = masked_y_true - y_pred
    abs_r = tf.math.abs(r)

    # Error loss
    error = tf.math.reduce_mean(abs_r, axis=-1) #+ tf.math.reduce_max(abs_r, axis=(1,2,3))

    # "Overshoot" penalty: if the estimated background is higher than the actual pixel value
    overshoot = tf.math.reduce_mean(abs_r - r, axis=-1)

    # Negative background penalty: if the estimated background is negative
    negative_bg = tf.math.reduce_mean(tf.math.abs(y_pred) - y_pred) 

    # Spline complexity penalty
    #complexity = tf.math.reduce_mean(tf.math.square(spline_layer.ww)) + tf.math.reduce_mean(tf.math.square(spline_layer.vw))

    return error + alpha*(overshoot + negative_bg) #+ beta*complexity
    #return tf.math.log(0.001 + error + alpha*(overshoot + negative_bg))


# __/ Callbacks \__________
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', 
    min_delta=0.00005, 
    patience=100,
    restore_best_weights=True,
    verbose=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.5,
    patience=50, 
    min_lr=0.0001,
    verbose=True
)

def lr_sched(epoch, lr):
    if epoch == 25:
        return 0.1 * lr
    else:
        return lr

lrsched = tf.keras.callbacks.LearningRateScheduler(lr_sched, verbose=True)


class PredictionCallback(tf.keras.callbacks.Callback): 
    def __init__(self):
        os.makedirs(out_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        X = np.zeros(config['B'])
        y_true = self.model.layers[1].im.numpy()
        y_pred = self.model.predict_on_batch(X).numpy()[0,...]
        final = y_true - y_pred
        final -= final.min()
        final /= final.max()
        im = (255 * final).astype(np.uint8)
        im_fname = os.path.join(out_dir, f"Epoch_{epoch:03d}.png")
        io.imsave(im_fname, im)    

prediction = PredictionCallback()

callbacks = [earlystop, reduce_lr] #, prediction] #, lrsched]


# __/ Model fit and predict (spline fitting) \__________
def fit_spline(im, config):
    # Mask definition
    threshold = config['threshold']

    # Spline complexity params
    N, O = config['N'], config['O']
    # Spline control points initialization
    initializer = config['initializer']
    # Spline regularization parameter for loss function
    alpha = config['alpha']

    # Training params
    B, lr, epochs = config['B'], config['lr'], config['epochs']

    # y_true is im
    im_orig = np.expand_dims(im, axis=0)
    y_true = im_orig.repeat(B, axis=0)

    # Dummy input
    X = np.zeros(B) 

    # Model
    x = keras.layers.Input(shape=(), name='input_layer', batch_size=B)
    y = Spline(im, mask=threshold, n_control_points=N, order=O, initializer=initializer)(x)
    model = keras.Model(inputs=x, outputs=y, name="bgmodel")
    model.summary()

    # Initial train_points positions
    #plot_train_points(model, im)
    #plt.show()

    # Model compilation with custom loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    bg_loss = lambda y_true, y_pred: bg_loss_alpha(y_true, y_pred, model, alpha) 
    model.compile(optimizer, loss=bg_loss)

    print("Fitting spline...")
    t_start = time.perf_counter()
    # Model fit
    history = model.fit(
        x=X, y=y_true, 
        epochs=epochs, 
        callbacks=callbacks,
        verbose=0
    )
    t_end = time.perf_counter()
    print(f"Done in {t_end-t_start:.2f} seconds")

    # Optimized background model 
    y_pred = model.predict(X)

    return y_pred[0,...], model, history


# __/ Draw spline train points \__________
def plot_train_points(model, im):
    h, w, _ = im.shape
    spline_layer = model.layers[1]
    train_points = spline_layer.train_points.numpy()[0]
    if spline_layer.mask is not None:
        x = im * spline_layer.mask
    else:
        x = im

    fig, ax = plt.subplots(figsize=(16,8))
    ax.imshow(x[...,0], cmap='gray')
    ax.plot(train_points[:,1]*w, train_points[:,0]*h, 'go', fillstyle='none')


# %%

# __/ Main script \__________

print("\n\n__/ Preprocessing \__________")

# Open original image
xisf = XISF(in_filepath)
im_orig = xisf.read_image(0)
_, im_orig_median, _ = statistics(im_orig, "Original")

# Preprocessing uses a copy
im = im_orig.copy()

# Ignore zero values (real data has some pedestal) by converting to NaN
num_to_nan(im)

# Delinearize to stretch the background
im, pedestal, scale, bg_val = delinearize(im)
_, im_median, _ = statistics(im, "Delinearized")

# Downscale
im = downscale(im)
_ = statistics(im, "Downscaled")

# Replace NaNs
np.nan_to_num(im, copy=False)

# Visualize preprocessed image
plot_image_hist(im, "Delinearized & downscaled")

# Preview mask
plot_image_hist(Spline._generate_mask(im, config['threshold']), "Mask")


# %%
print("\n\n__/ Background modeling \__________")

# Fit spline
bg_hat, model, history = fit_spline(im, config)

print(f"N, B, epochs, loss: {config['N']}, {config['B']}, {len(history.history['loss'])}, {min(history.history['loss']):.5f}")

plt.figure(figsize=(10, 3))
plt.plot(history.history['loss'], label='Loss')
plt.title('Loss')


# %%
# Visualize fitted spline (background model)
_ = statistics(bg_hat, "Bg model")
plot_image_hist(bg_hat, "Background model")


# %%
# Visualize final train points over the (masked) image
#plot_train_points(model, im)
spline_layer = model.layers[1]
spline_layer.plot_train_points()


# %%
plot_image_hist(im-bg_hat+im_median, "Bg subtracted (downsized, delinearized)")


# %%
print("\n\n__/ Full-size background model & subtracted image \__________")

# Generate the final background model by interpolating the trained spline to the original image size
t_start = time.perf_counter()
bg_fullres = spline_layer.interpolate(im_orig.shape, chunks=config['downscaling_factor']**2)
t_end = time.perf_counter()
print(f"Elapsed {t_end-t_start:.2f} seconds")

_ = statistics(bg_fullres, "Bg (full size)")
plot_image_hist(bg_fullres, "Background model (full size)")


# %%
# Linearize the background model...
bg_fullres_linear = linearize(bg_fullres, pedestal, scale, bg_val)
_ = statistics(bg_fullres_linear, "Bg (linear)")

# ... and subtract it from the original image
im_final = im_orig - bg_fullres_linear
im_final_min, _, _ = statistics(im_final, "Subtracted")

# Visualize out of range (negative, really) values
plt.figure(figsize=(16,10))
plt.imshow(-im_final.clip(-1,0)[...,0], cmap='gray')
plt.title("Pixels with negative value after subtraction")

# Apply pedestal so the final image has the same median value as the original
im_final -= im_final_min
im_final += im_orig_median
if im_final.max() > 1.0:
    im_final /= im_final.max()

_ = statistics(im_final, "Final")

# %%
plt.figure(figsize=(16,10))
plt.imshow(delinearize(im_final.copy(), 0.25)[0][...,0], cmap='gray')

# %%
os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

# Write final image and background model to file
XISF.write(out_filepath, im_final, xisf.get_images_metadata()[0], xisf.get_file_metadata())
XISF.write(bg_filepath, bg_fullres_linear)

# %%
# Experiment: variance 
# experiment = []
# for _ in range(30):
#    y_pred, model, history = fit_spline(im, config)
#    bg = y_pred[0,...]
#    final = im - bg + np.median(im)

#    data = {
#        'loss': min(history.history['loss']),
#        'epochs': len(history.history['loss']),
#        'min': final.min(),
#        'median': np.median(final),
#        'max': final.max()
#    }
#    experiment.append(data)

# df = pd.DataFrame(experiment)
# df[['loss']].plot()
# df.to_csv("%s_B%d_var.csv" % (in_filename, config['B']))


# %%
# Experiment: varying N
#experiment = []
#for N in [15, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
#    config['N'] = N
#    y_pred, model, history = fit_spline(im_orig, config)
#    bg = y_pred[0,...]
#    final = im_orig - bg
#
#    data = {
#        'N': N,
#        'loss': min(history.history['loss']),
#        'epochs': len(history.history['loss']),
#        'min': final.min()
#    }
#    experiment.append(data)
#
#df = pd.DataFrame(experiment).set_index("N")
#df[['loss']].plot()
#df.to_csv("%s_N.csv" % (filename,))
