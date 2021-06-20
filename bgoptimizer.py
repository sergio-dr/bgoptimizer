# %%
%load_ext autoreload
%autoreload 2

# %%
import os
from xisf import XISF
import numpy as np
import tensorflow as tf
from tensorflow import keras
from spline import Spline
import matplotlib.pyplot as plt

filename = "Ha_nonlinear_median.xisf"

N = 16
O = 3
B = 1 # TODO: batch_size, any effect on convergence?

alpha = 1

lr = 0.001
epochs = 1000

# %%
xisf = XISF(filename)
im_orig = xisf.read_image(0)
im_shape = im_orig.shape


# %%
x = keras.layers.Input(shape=(), name='input_layer', batch_size=B)
y = Spline(im_orig, control_points=N, order=O)(x)
model = keras.Model(inputs=x, outputs=y, name="bgmodel")

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# %%
def bg_loss_alpha(y_true, y_pred, alpha):
    r = y_pred - y_true
    abs_r = tf.math.abs(r)

    mse = tf.math.reduce_mean(r*r, axis=-1)
    penalty = tf.math.reduce_mean(abs_r - r, axis=-1)

    return mse + alpha*penalty

bg_loss = lambda y_true, y_pred: bg_loss_alpha(y_true, y_pred, alpha)

model.compile(optimizer, loss=bg_loss)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', 
    min_delta=0.0001, 
    patience=50,
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

callbacks = [earlystop, reduce_lr] #, lrsched]

# %%
im_orig = np.expand_dims(im_orig, axis=0)
y_true = im_orig.repeat(B, axis=0)
X = np.zeros(B) 
im_orig.shape, y_true.shape, X.shape

# %%
# Draw spline train points
import seaborn as sns
from matplotlib.patches import Rectangle

def plot_train_points():
    train_points = model.layers[1].train_points.numpy()[0]
    # values = _apply_interpolation...
    ax = sns.scatterplot(x=train_points[:,0], y=train_points[:,1]) 
    rect = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    _ = plt.gca().add_patch(rect)

plot_train_points()

# %%
history = model.fit(
    x=X, y=y_true, 
    epochs=epochs, 
    callbacks=callbacks
)

plt.plot(history.history['loss'], label='Loss')

# %%
y_pred = model.predict_on_batch(X).numpy()

plt.imshow(y_pred[0,...,0], cmap='gray', vmin=0, vmax=1)

# %%
plot_train_points()

# %%
plt.figure(figsize=(16,10))
plt.imshow(y_true[0,...,0], cmap='gray')

# %%
plt.figure(figsize=(16,10))
plt.imshow(y_true[0,...,0] - y_pred[0,...,0], cmap='gray')

# %%


