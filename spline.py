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

    You can specify a mask to avoid placing control points on high-valued pixels. 
    The mask is stored in the layer (Spline.mask) as a tensor so you can use also
    for applying it on custom loss functions. 

    order and initial_regularization correspond to tfa.image.interpolate_spline() 
    order and regularization_weight arguments, respectively. 

    Attributes:
        im: image (ndarray) to fit the spline to; expected shape is (h, w, 1).
        mask: if float, defines the threshold value for the mask; if tuple(float,float),
           defines minimum and maximum threshold values for the mask (useful for ignoring 
           missing values encoded below the pedestal); if ndarray, it has to have the same
           shape as im, and its values should be in the [0.0, 1.0] range. By default it is 
           1.0, i.e., no mask is used. 
        n_control_points: number of control points for the spline (64 by default)
        order: 2 (thin-plate spline, default) or 3 (bicubic spline)
        initializer: either 'random' or 'grid', for initial train points distribution
        initial_regularization: float for smoothness of the initial spline (0.1 by 
          default); 0.0 means no smoothness. 

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

    def __init__(self, im, mask=1.0, n_control_points=64, order=2, initializer='random', initial_spline_regularization=0.1, **kwargs):
        super(Spline, self).__init__(**kwargs)
        self.im = im

        if isinstance(mask, np.ndarray):
            if im.shape == mask.shape:
                self.mask = mask
            else:
                raise ValueError("if mask is a ndarray, it should have the same shape as im")
        elif isinstance(mask, tuple):
            try:
                self.mask = float(mask[0]), float(mask[1])
            except:
                raise ValueError("if mask is a tuple, it should be given as (float, float)")
        elif isinstance(mask, float):
            self.mask = (0.0, mask) # Mask as thresholds range (thr_min, thr_max) or ndarray
        else:
            raise ValueError("mask should be a float, tuple(float, foat), or ndarray")
        # From now on, self.mask is either a tuple (float, float) or a ndarray.

        self.shape = im.shape
        self.bg_val = np.median(self.im)        
        self.n_control_points = n_control_points
        self.order = order
        self.initializer = initializer
        self.initial_spline_regularization = initial_spline_regularization

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        h, w, _ = self.shape 

        # query_points: generate every pixel position, on the ([0,1], [0,1]) range
        y, x = np.linspace(0, 1, h, endpoint=False), np.linspace(0, 1, w, endpoint=False)
        query_points = np.array(np.meshgrid(y, x)).T.astype(np.float32)
        self.query_points = tf.constant(query_points.reshape(1, -1, 2)) # Add batch axis, convert to tensor 

        # if mask is given as a threshold range, convert it to ndarray
        if isinstance(self.mask, tuple): 
            thr_min, thr_max = self.mask
            self.mask = ((thr_min <= self.im) & (self.im <= thr_max)).astype(np.float32)
        
        # Checks
        assert isinstance(self.mask, np.ndarray) # At this point, mask should be a ndarray
        assert self.mask.shape == self.im.shape  # At this point, mask and shape should have the same shape
        
        if self.initializer == 'random':
            # Generate train_points randomly distributed over [0,1) box
            train_points_initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
            self.train_points = self.add_weight(
                shape=(1, self.n_control_points, 2), initializer=train_points_initializer, trainable=True
            )
        elif self.initializer == 'grid':
            n_points_per_axis = np.round(np.sqrt(self.n_control_points))
            gy, gx = np.linspace(0, h-1, n_points_per_axis)/h, np.linspace(0, w-1, n_points_per_axis)/w
            train_points = np.array(np.meshgrid(gy, gx)).T.astype(np.float32)
            #   TODO: add random jitter? training most probably jitter them anyway

            # Add batch axis, create layer weights for train_points
            self.train_points = tf.Variable(initial_value=train_points.reshape(1, -1, 2), trainable=True) 
        else:
            raise ValueError("initializer should be either 'random' or 'grid'")

        # Get pixel locations corresponding to the train_points
        train_coords = self.train_points.numpy().squeeze(axis=0)
        train_coords = (train_coords * np.array([h, w])).astype(np.int32) # nearest pixel position

        # Evaluate pixel values from the image and the mask, using train_coords pixel positions
        pixel_values = self.im[ train_coords[:,0], train_coords[:,1], 0 ]
        mask_values = self.mask[ train_coords[:,0], train_coords[:,1], 0 ]
        # Initial train_values are evaluated as follows:
        # - Fully unmasked pixels: actual pixel value
        # - Fully masked pixels: estimated background value (median)
        # - Partially masked pixels: linear combination of the above
        train_values = mask_values * pixel_values + (1-mask_values) * self.bg_val

        # Expand train_values for batch and output axis
        train_values = np.expand_dims(train_values, axis=(0,-1))

        # Convert im, mask to tensors
        self.im, self.mask = tf.constant(self.im), tf.constant(self.mask)       

        # We use tfa.image._solve_interpolation for the initial spine coefficient values
        # TODO: make regularization_weight an argument?
        ww, vw = _solve_interpolation(
            self.train_points, train_values, self.order, regularization_weight=self.initial_spline_regularization
        ) 
        # Create layer weights for w and v spline coefficients
        self.ww = tf.Variable(initial_value=ww, trainable=True)
        self.vw = tf.Variable(initial_value=vw, trainable=True)

        super(Spline, self).build(input_shape)
      

    def call(self, inputs):
        h, w, _ = self.shape

        # Broadcast the weights and query_points to the batch_size
        #   TODO: clip the train_points to the [0,1] box to constrain them?
        #     self.train_points --> tf.clip_by_value(self.train_points, 0.0, 1.0)
        train_points = tf.repeat(self.train_points, self.batch_size, axis=0)
        w_weights = tf.repeat(self.ww, self.batch_size, axis=0)
        v_weights = tf.repeat(self.vw, self.batch_size, axis=0)
        query_points = tf.repeat(self.query_points, self.batch_size, axis=0)

        # Perform the interpolation, output shape is (batch_size, h*w, 1)
        query_values = _apply_interpolation(
            query_points=query_points,
            train_points=train_points,
            w=w_weights,
            v=v_weights,
            order=self.order,
        )

        # Reshape to (batch_size, h, w, 1)
        return tf.reshape(query_values, (-1, h, w, 1))


    # Training is usually done on a downscaled image; this method allows to 
    # generate the trained spline on any shape = (h,w,...) (usually (h,w,1)). 
    #
    # Note: this has to be done in chunks, because _apply_interpolation()
    # gives an OOM error trying to MatMul if query_points is large.  
    # chunks = downscale_factor**2 avoids OOM, if the spline could train with that
    # downscale_factor.
    def interpolate(self, shape, chunks):
        h, w = shape[:2]

        # Generate query_points according to im.shape
        y, x = np.linspace(0, 1, h, endpoint=False), np.linspace(0, 1, w, endpoint=False)
        query_points = tf.constant(np.array(np.meshgrid(y, x)).T.reshape(1, -1, 2).astype(np.float32)) 

        # Use ww & vw on _apply_interpolation to generate the spline
        chunk_sz = np.ceil((h*w) / chunks).astype(np.int32)
        query_values = []
        for i in range(chunks):
            start = i*chunk_sz
            end = min(start + chunk_sz, query_points.shape[1])
            print(f"{100*i//chunks}..", end='')

            query_points_chunk = query_points[:,start:end,:]
            query_values_chunk = _apply_interpolation(
                query_points=query_points_chunk,
                train_points=self.train_points,
                w=self.ww,
                v=self.vw,
                order=self.order,
            )
            query_values.append(query_values_chunk) 
        query_values = tf.concat(query_values, axis=1)
        bg = tf.reshape(query_values, shape) 
        print("100%")

        return bg.numpy()
