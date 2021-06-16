
# %%
import os
from xisf import XISF
import numpy as np
import tensorflow as tf
from tensorflow import keras
from spline import Spline
import matplotlib.pyplot as plt

filename = "Ha_nonlinear_r.xisf"

N = 16
O = 3
B = 4

alpha = 1

lr = 0.001
epochs = 10000

# %%
xisf = XISF()
xisf.read(filename)
im_orig = xisf.read_image(0)
im_shape = im_orig.shape
xisf.close()

# %%
#plt.imshow(im_orig[:,:,0])
im_orig = np.expand_dims(im_orig, axis=0)

# %%
y_true = im_orig.repeat(B, axis=0)
y_true.shape

# %%
x = keras.layers.Input(shape=(), name='input_layer', batch_size=B)
y = Spline(shape=im_shape, control_points=N, order=O)(x)
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
    patience=25,
    restore_best_weights=True,
    verbose=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.5,
    patience=5, 
    min_lr=0.0001,
    verbose=True
)

def lr_sched(epoch, lr):
    if epoch == 25:
        return 0.1 * lr
    else:
        return lr

lrsched = tf.keras.callbacks.LearningRateScheduler(lr_sched, verbose=True)

callbacks = [earlystop] #, reduce_lr, lrsched]

# %%
X = np.zeros(B) 
X.shape

# %%
history = model.fit(
    x=X, y=y_true, 
    epochs=epochs, 
    callbacks=callbacks
)

# %%
y_pred = model.predict_on_batch(X).numpy()

plt.imshow(y_pred[0,...,0], cmap='gray')

# %%
plt.plot(history.history['loss'], label='Loss')

# %%
plt.figure(figsize=(16,10))
plt.imshow(y_true[0,...,0], cmap='gray')

# %%
plt.figure(figsize=(16,10))
plt.imshow(y_true[0,...,0] - y_pred[0,...,0], cmap='gray')

# %%


