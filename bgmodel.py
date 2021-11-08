import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from spline import Spline
from skimage import io
import matplotlib.pyplot as plt


class BgModel:

    config_defaults = {
        'N': 32,
        'O': 2,
        'threshold': (0.001, 0.95),
        'initializer': 'random', 
        'alpha': 5,    
        'B': 1,
        'lr': 0.001,
        'epochs': 1000,
        'out_dirpath': '.'
    }

    def __init__(self, config):
        self.config = {**self.config_defaults, **config}
        self.model = None
        self.spline_layer = None
        self.history = {}


    def bg_loss_alpha(self, y_true, y_pred): # beta=0.1):
        # In this model, y_true is im, y_pred is the generated background model (spline)
        
        # Get mask and bg_val from the spline layer
        mask = self.spline_layer.mask
        bg_val = self.spline_layer.bg_val
        
        # Apply mask (like in Spline.build())
        masked_y_true = mask*y_true + (1-mask)*bg_val
        
        # Residuals
        r = masked_y_true - y_pred
        abs_r = tf.math.abs(r)

        # Error loss (TODO: may be not optimal for 3-channel images)
        error = tf.math.reduce_mean(abs_r, axis=-1)  #+ tf.math.reduce_max(abs_r, axis=(1,2,3))

        # "Overshoot" penalty: if the estimated background is higher than the actual pixel value
        overshoot = tf.math.reduce_mean(abs_r - r, axis=-1)

        # Negative background penalty: if the estimated background is negative
        negative_bg = tf.math.reduce_mean(tf.math.abs(y_pred) - y_pred) 

        # Spline complexity penalty
        #complexity = tf.math.reduce_mean(tf.math.square(self.spline_layer.ww)) + tf.math.reduce_mean(tf.math.square(self.spline_layer.vw))

        alpha = self.config['alpha']
        return error + alpha*(overshoot + negative_bg) #+ beta*complexity
        #return tf.math.log(0.001 + error + alpha*(overshoot + negative_bg))


    def _build_callbacks(self):
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
            def __init__(self, config):
                self.config = config
                os.makedirs(self.config['out_dirpath'], exist_ok=True)

            def on_epoch_end(self, epoch, logs={}):
                X = np.zeros(self.config['B'])
                y_true = self.model.layers[1].im.numpy()
                y_pred = self.model.predict_on_batch(X).numpy()[0,...]
                final = y_true - y_pred
                final -= final.min()
                final /= final.max()
                im = (255 * final).astype(np.uint8)
                im_fname = os.path.join(self.config['out_dirpath'], f"Epoch_{epoch:03d}.png")
                io.imsave(im_fname, im)    

        prediction = PredictionCallback(self.config)

        return [earlystop, reduce_lr] #, prediction] #, lrsched]


    # __/ Model fit and predict (spline fitting) \__________
    def fit_transform(self, im):
        # Mask definition
        threshold = self.config['threshold']

        # Spline complexity params
        N, O = self.config['N'], self.config['O']
        # Spline control points initialization
        initializer = self.config['initializer']
        # Spline regularization parameter for loss function
        alpha = self.config['alpha']

        # Training params
        B, lr, epochs = self.config['B'], self.config['lr'], self.config['epochs']

        # y_true is im (expanded in batch axis)
        y_true = np.expand_dims(im, axis=0).repeat(B, axis=0)

        # Dummy input
        X = np.zeros(B) 

        # Model
        x = keras.layers.Input(shape=(), name='input_layer', batch_size=B)
        y = Spline(im, mask=threshold, n_control_points=N, order=O, initializer=initializer)(x)
        self.model = keras.Model(inputs=x, outputs=y, name="bgmodel")
        self.spline_layer = self.model.layers[1]
        #self.model.summary()

        # Initial train_points positions
        # self.spline_layer.plot_train_points()

        # Model compilation with custom loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer, loss=self.bg_loss_alpha)

        print("Fitting spline...")
        t_start = time.perf_counter()
        # Model fit
        self.history = self.model.fit(
            x=X, y=y_true, 
            epochs=epochs, 
            callbacks=self._build_callbacks(),
            verbose=0
        )
        t_end = time.perf_counter()
        print(f"Done in {t_end-t_start:.2f} seconds")

        # Optimized background model 
        y_pred = self.model.predict(X)

        return y_pred[0,...]


    def training_report(self):
        cfg, h = self.config, self.history
        print(f"N, B, epochs, loss: {cfg['N']}, {cfg['B']}, {len(h.history['loss'])}, {min(h.history['loss']):.5f}")

        plt.figure(figsize=(10, 3))
        plt.plot(h.history['loss'], label='Loss')
        plt.title('Loss')
        plt.show()


    def interpolate_to(self, shape):
        t_start = time.perf_counter()
        bg_fr = self.spline_layer.interpolate(shape, chunks=self.config['downscaling_factor']**2)
        t_end = time.perf_counter()
        print(f"Elapsed {t_end-t_start:.2f} seconds")
        return bg_fr
