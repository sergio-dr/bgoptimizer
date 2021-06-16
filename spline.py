import tensorflow as tf
from tensorflow_addons.image.interpolate_spline import _apply_interpolation
from tensorflow import keras
import numpy as np

# TODO: documentar

class Spline(keras.layers.Layer):
    def __init__(self, shape, control_points=16, order=3, **kwargs):
        super(Spline, self).__init__(**kwargs)
        self.shape = shape
        self.control_points = control_points
        self.order = order

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        h, w, _ = self.shape

        # query_points: todos los puntos (x, y) = ([0:W], [0:H])
        x, y = np.arange(0, w), np.arange(0, h)
        self.query_points = tf.constant(np.array(np.meshgrid(x, y)).T.reshape(1, -1, 2).astype(np.float32)) 

        train_points_initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
        wvalues_initializer = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
        self.train_points = self.add_weight(
            shape=(1, self.control_points, 2), initializer=train_points_initializer, trainable=True
        )
        self.ww = self.add_weight(
            shape=(1, self.control_points, 1), initializer=wvalues_initializer, trainable=True
        )
        self.vw = self.add_weight(
            shape=(1, 3, 1), initializer=wvalues_initializer, trainable=True
        )

        super(Spline, self).build(input_shape)
      

    def call(self, inputs):
        h, w, _ = self.shape

        # tfa.image._apply_interpolation: interpolación spline poliarmónica.
        # Es una función auxiliar de tfa.image.interpolate_spline(); no podemos usar ésta
        # directamente ya que determina w y v realizando tf.linalg.solve() y ésta 
        # da problemas al encontrarse con matrices no invertibles. 
        #
        # query_points: `[b, m, d]` x values to evaluate the interpolation at.
        # train_points: `[b, n, d]` x values that act as the interpolation centers
        #     (the c variables in the wikipedia article).
        # w: `[b, n, k]` weights on each interpolation center.
        # v: `[b, d, k]` weights on each input dimension.        
        # Returns: `[b, m, k]` float `Tensor` of query values
        #
        # Donde:
        # b = batch_size
        # m = número de puntos a evaluar, w*h
        # d = 2 (dimensión del espacio de entrada, 2D)
        # n = número de puntos de control (hiperparámetro)
        # k = 1 (dimensión de los valores de salida)

        query_points = tf.repeat(self.query_points, self.batch_size, axis=0)

        # Escalamos los puntos de control (x,y)
        train_points = self.train_points * tf.constant([w, h], dtype=tf.float32) 
        train_points = tf.repeat(train_points, self.batch_size, axis=0)

        # Los pesos también necesitan escalado... pero cuál ??????????
        w_weights = tf.repeat(self.ww, self.batch_size, axis=0) / (w*w*h*h)
        v_weights = tf.repeat(self.vw, self.batch_size, axis=0)

        query_values = _apply_interpolation(
            query_points=self.query_points,
            train_points=train_points,
            w=w_weights,
            v=v_weights,
            order=self.order,
        )

        # Los valores tienen shape (B, W*H, 1) ordenados por columnas, filas;
        # hay que reestructurar a formato de imagen
        query_values = tf.reshape(query_values, (-1, w, h, 1))
        return tf.transpose(query_values, perm=[0, 2, 1, 3])

# TODO
    def get_config(self):
        config = super(Spline, self).get_config()
        config.update({"shape": self.shape})
        config.update({"order": self.order})
        return config