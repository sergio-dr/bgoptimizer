import tensorflow as tf
from tensorflow_addons.image.interpolate_spline import _apply_interpolation, _solve_interpolation
from tensorflow import keras
import numpy as np

class Spline(keras.layers.Layer):
    """
    Generates a spline by poliharmonic interpolation.
    
    This implementation is based on tfa.image.interpolate_spline(), which
    makes the spline interpolation in two steps: first, it solves for a set
    of coefficients w and v from a set of train_points and train_values; then,
    it makes the actual interpolation by using those coefficients values at 
    the train_points and a list of query_points. 
    
    In this layer, the trainable weights are the train_point and the spline
    coefficients w and v. At the initialization of the layer, it uses the 
    solve phase, tfa.image._solve_interpolation(), on randomly defined 
    train_points over the image given as attribute im, to determine the 
    initial values of the weights w and v. 
    
    The layer uses these weights to interpolate a spline, evaluating it at 
    all pixels positions of the image im, i.e, the output of the layer
    is a image with the same shape as im.

    Attributes:
        im: image to fit the spline to; expected shape is (h, w, 1).
        control_points: number of control points for the spline (16 by default)
        order: 2 (thin-plate spline) o 3 (bicubic spline, default)

    Input shape:
        Ignored, except for the batch_size.

    Output shape:
        N-D tensor with shape: (batch_size, h, w, 1).
    """

    # References:
    #
    # See https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_spline
    #   query_points: [b, m, d] x values to evaluate the interpolation at.
    #   train_points: [b, n, d] x values that act as the interpolation centers
    #   w: [b, n, k] weights on each interpolation center.
    #   v: [b, d, k] weights on each input dimension.        
    #   query_values: [b, m, k] interpolation results
    #
    # Where:
    #   b = batch_size
    #   m = number of locations to evaluate (w*h in this case)
    #   d = input space dimension (2 in this case)
    #   n = number of train points (parameter of the layer)
    #   k = output values dimensions (1 in this case)

    def __init__(self, im, control_points=16, order=3, **kwargs):
        super(Spline, self).__init__(**kwargs)
        self.im = im
        self.shape = im.shape
        self.control_points = control_points
        self.order = order

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        h, w, _ = self.shape 

        # query_points: generate every pixel position, on the ([0,1], [0,1]) range
        y, x = np.linspace(0, 1, h, endpoint=False), np.linspace(0, 1, w, endpoint=False)
        self.query_points = tf.constant(np.array(np.meshgrid(y, x)).T.reshape(1, -1, 2).astype(np.float32)) 

        # train_points: randomly distributed over ([0,1], [0,1]) range
        train_points_initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
        self.train_points = self.add_weight(
            shape=(1, self.control_points, 2), initializer=train_points_initializer, trainable=True
        )

        # w, v spline coefficients: initialize from actual image pixel values on the train_points (scaled
        # to the image size) for faster convergence (much faster than random initialization)
        train_coords = self.train_points.numpy().squeeze(axis=0)
        train_coords = (train_coords * np.array(self.shape[:-1])).astype('int') # nearest pixel position
        train_values = np.expand_dims(self.im[ train_coords[:,0], train_coords[:,1], 0 ], axis=(0,-1))
        # We use tfa.image._solve_interpolation for the initial spine coefficient values
        ww, vw = _solve_interpolation(
            self.train_points, train_values, self.order, regularization_weight=0.1
        ) 
        self.ww = tf.Variable(initial_value=ww, trainable=True)
        self.vw = tf.Variable(initial_value=vw, trainable=True)

        super(Spline, self).build(input_shape)
      

    def call(self, inputs):
        h, w, _ = self.shape

        # Clip the train_points to the [0,1] box to constrain them
        # TODO: make this optional?
        train_points = keras.backend.clip(self.train_points, 0.0, 1.0)

        # Broadcast the weights and query_points to the batch_size
        train_points = tf.repeat(train_points, self.batch_size, axis=0)
        w_weights = tf.repeat(self.ww, self.batch_size, axis=0)# / (w*w*h*h)
        v_weights = tf.repeat(self.vw, self.batch_size, axis=0)
        query_points = tf.repeat(self.query_points, self.batch_size, axis=0)

        # Perform the interpolation, output shape is (batch_size, h*w, 1)
        query_values = _apply_interpolation(
            query_points=self.query_points,
            train_points=train_points,
            w=w_weights,
            v=v_weights,
            order=self.order,
        )

        # Reshape to (batch_size, h, w, 1)
        return tf.reshape(query_values, (-1, h, w, 1))

    # TODO: sí puede ser interesante añadir un método para generar el spline sobre otro
    # conjunto de query_points (para la imagen en tamaño original, entiendiendo que el
    # entrenamiento se hará sobre una imagen reducida)
    def generate_spline(shape):
        # generar query_points según shape
        # use ww y vw y _apply_interpolation para generar spline
        pass
