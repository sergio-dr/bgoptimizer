# %%
%load_ext autoreload
%autoreload 2

import os
from xisf import XISF
import numpy as np
import tensorflow as tf
from tensorflow import keras
from spline import Spline

# For visualization only
import matplotlib.pyplot as plt
from PIL import Image 

# For experiments only
import pandas as pd 

# %%

# __/ Script parameters \__________
# TODO: command line args
filename = "Ha_nonlinear_median.xisf" # "Pleyades_L_nonlinear_median.xisf" 

config = {
    'N': 400,
    'O': 3,
    'threshold': 0.5,
    'B': 1,
    'alpha': 100,
    'lr': 0.001,
    'epochs': 5000,
}


# %%

# __/ Custom loss \__________
# TODO: meter esta función en la capa spline ??
def bg_loss_alpha(y_true, y_pred, model, alpha):
    # Residuals
    r = y_true - y_pred

    # Apply mask of residuals
    if model.layers[1].mask is not None:
        r *= model.layers[1].mask

    abs_r = tf.math.abs(r)

    # TODO: mae abs_r vs mse r*r ...
    # tf.math.log(1+r*r)
    # 2*tf.math.reciprocal( 1+tf.math.exp(-15*r*r))-1 # tipo sigmoide
    #   https://www.wolframalpha.com/input/?i=Plot%5B2%2F%281%2Be%5E%28-15*x%5E2%29%29-1%2C+x+%3D+-1+to+1%5D
    error = tf.math.reduce_mean( abs_r , axis=-1) 

    # TODO: nombrar estos penalties
    penalty = tf.math.reduce_mean(abs_r - r, axis=-1) / 2
    negative_bg = tf.math.reduce_mean(tf.math.abs(y_pred) - y_pred) / 2 # Éste aplica independientemente de la máscara

    return error + alpha*(penalty + negative_bg)


# __/ Callbacks \__________
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', 
    min_delta=0.00001, 
    patience=25,
    restore_best_weights=True,
    verbose=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.5,
    patience=10, 
    min_lr=0.00001,
    verbose=True
)

def lr_sched(epoch, lr):
    if epoch == 25:
        return 0.1 * lr
    else:
        return lr

lrsched = tf.keras.callbacks.LearningRateScheduler(lr_sched, verbose=True)


class PredictionCallback(tf.keras.callbacks.Callback):    
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict_on_batch(X)
    final = y_true[0,...,0] - y_pred[0,...,0]
    final -= final.min()
    final /= final.max()
    im = Image.fromarray( (255 * final).astype(np.uint8) )
    im.save("out\\Epoch_%03d.png" % (epoch,))

prediction = PredictionCallback()

callbacks = [earlystop, reduce_lr] #, prediction] #, lrsched]


# __/ Model fit and predict (spline fitting) \__________
def fit_spline(im, config):
    N, O, alpha, threshold = config['N'], config['O'],  config['alpha'], config['threshold']
    B, lr, epochs = config['B'], config['lr'], config['epochs']

    im_orig = np.expand_dims(im, axis=0)
    y_true = im_orig.repeat(B, axis=0)
    X = np.zeros(B) 
    #print(im_orig.shape, y_true.shape, X.shape)

    x = keras.layers.Input(shape=(), name='input_layer', batch_size=B)
    y = Spline(im, mask=threshold, n_control_points=N, order=O)(x)
    model = keras.Model(inputs=x, outputs=y, name="bgmodel")

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    bg_loss = lambda y_true, y_pred: bg_loss_alpha(y_true, y_pred, model, alpha) 
    model.compile(optimizer, loss=bg_loss)

    history = model.fit(
        x=X, y=y_true, 
        epochs=epochs, 
        callbacks=callbacks
    )

    y_pred = model.predict(X)

    return y_pred, model, history


# __/ Draw spline train points \__________
def plot_train_points(model, im):
    h, w, _ = im.shape
    train_points = model.layers[1].train_points.numpy()[0]
    if model.layers[1].mask is not None:
        x = im * model.layers[1].mask.numpy()
    else:
        x = im

    fig, ax = plt.subplots(figsize=(16,8))
    ax.imshow(x, cmap='gray')
    ax.plot(train_points[:,1]*w, train_points[:,0]*h, 'go', fillstyle='none')


# %%

# __/ Main script \__________
xisf = XISF(filename)
im_orig = xisf.read_image(0)

y_pred, model, history = fit_spline(im_orig, config)
plt.plot(history.history['loss'], label='Loss')

#%%
if model.layers[1].mask is not None:
    plt.imshow(model.layers[1].mask.numpy())

# %%
bg = y_pred[0,...]
plt.imshow(bg, cmap='gray', vmin=0, vmax=1)
print("Range: ", bg.min(), bg.max())

# %%
plot_train_points(model, im_orig)

# %%
plt.figure(figsize=(16,10))
plt.imshow(im_orig, cmap='gray')

# %%
final = im_orig - bg
plt.figure(figsize=(16,10))
plt.imshow(final, cmap='gray')

print("Range: ", final.min(), final.max())

# %%
print("N, B, epochs, loss: %d, %d, %d, %.5f" % (config['N'], config['B'], len(history.history['loss']), min(history.history['loss'])))

# %%
plt.imshow(-final.clip(-1,0), cmap='gray')
# TODO: salen valores negativos, es necesario hacer final -= final.min() para ajustar el 0. 

# %%
final -= final.min()
if final.max() > 1:
    final /= final.max()
# %%
XISF.write("final_%s" % (filename,), final, xisf.get_images_metadata()[0], xisf.get_file_metadata())


# %%
# Experiment: variance 
#experiment = []
#for _ in range(0, 15):
#    y_pred, model, history = fit_spline(im_orig, config)
#    bg = y_pred[0,...]
#    final = im_orig - bg#
#
#    data = {
#        'loss': min(history.history['loss']),
#        'epochs': len(history.history['loss']),
#        'min': final.min()
#    }
#    experiment.append(data)#
#
#df = pd.DataFrame(experiment)
#df[['loss']].plot()
#df.to_csv("%s_B%d_var.csv" % (filename, config['B']))


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
