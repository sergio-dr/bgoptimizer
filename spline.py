import tensorflow as tf
from tensorflow_addons.image.interpolate_spline import _apply_interpolation, _solve_interpolation
from tensorflow import keras
import numpy as np

# TODO: batch_size, any effect on convergence?


class Spline(keras.layers.Layer):
    def __init__(self, img, control_points=16, order=3, **kwargs):
        super(Spline, self).__init__(**kwargs)
        self.img = img
        self.shape = img.shape
        self.control_points = control_points
        self.order = order

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        h, w, _ = self.shape 

        # query_points: generate every pixel position (x, y) = ([0:W], [0:H])
        # TODO: generate them on the ([0,1], [0,1]) range
        x, y = np.arange(0, w), np.arange(0, h)
        self.query_points = tf.constant(np.array(np.meshgrid(x, y)).T.reshape(1, -1, 2).astype(np.float32)) 

        # train_points: randomly distributed over ([0,1], [0,1]) range
        train_points_initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
        self.train_points = self.add_weight(
            shape=(1, self.control_points, 2), initializer=train_points_initializer, trainable=True
        )

        # ww, vw spline coefficients: initialize from actual image pixel values on the train_points (scaled
        # to the image size) for faster convergence (much faster than random initialization)
        train_coords = self.train_points.numpy().squeeze(axis=0)
        train_coords = (train_coords * np.array(self.shape[0:2])).astype('int') 
        train_values = np.expand_dims(self.img[ train_coords[:,0], train_coords[:,0], 0 ], axis=(0,-1))
        # We use _solve_interpolation for the initial spine coefficient values, 
        # see https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_spline
        ww, vw = _solve_interpolation(
            self.train_points, train_values, self.order, regularization_weight=0.1
        ) 
        self.ww = tf.Variable(initial_value=ww, trainable=True)
        self.vw = tf.Variable(initial_value=vw, trainable=True)

        super(Spline, self).build(input_shape)
      

    def call(self, inputs):
        h, w, _ = self.shape

        # tfa.image._apply_interpolation: interpolación spline poliarmónica.
        # Es una función auxiliar de tfa.image.interpolate_spline(); no podemos usar ésta
        # directamente durante fit() ya que determina w y v realizando tf.linalg.solve() y ésta 
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

        # Clip the train_points to the [0,1] box to constrain them
        # TODO: make this an option?
        train_points = keras.backend.clip(self.train_points, 0.0, 1.0)
        # Escalamos los puntos de control (x,y)        
        train_points = self.train_points * tf.constant([w, h], dtype=tf.float32) 
        train_points = tf.repeat(train_points, self.batch_size, axis=0)

        # Los pesos también necesitan escalado... pero cuál ??????????
        # TODO: don't scale query_values in init() and avoid w*w*h*h scaling here
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
        # TODO: generate query_points as y,x values instead to avoid transposing
        query_values = tf.reshape(query_values, (-1, w, h, 1))
        return tf.transpose(query_values, perm=[0, 2, 1, 3])

    # TODO: quitar, no le veo mucho sentido a almacenar esta capa, lo que interesa
    # del modelo es entrenarlo, no usarlo con predict. 
    def get_config(self):
        config = super(Spline, self).get_config()
        config.update({"shape": self.shape})
        config.update({"order": self.order})
        return config

    # TODO: sí puede ser interesante añadir un método para generar el spline sobre otro
    # conjunto de query_points (para la imagen en tamaño original, entiendiendo que el
    # entrenamiento se hará sobre una imagen reducida)
    def generate_spline(shape):
        # generar query_points según shape
        # use ww y vw y _apply_interpolation para generar spline
        pass
