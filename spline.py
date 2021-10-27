import tensorflow as tf
from tensorflow_addons.image.interpolate_spline import _apply_interpolation, _solve_interpolation
from tensorflow import keras
import numpy as np
import math

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

    Attributes:
        im: image (ndarray) to fit the spline to; expected shape is (h, w, 1).
        mask: if float, defines the threshold value for the mask; if ndarray, it has to
          have the same shape as im; if None (default), no mask is used.
        n_control_points: number of control points for the spline (16 by default)
        order: 2 (thin-plate spline) or 3 (bicubic spline, default)

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

    def __init__(self, im, mask=None, n_control_points=16, order=3, **kwargs):
        super(Spline, self).__init__(**kwargs)
        self.im = im
        self.mask = mask
        self.shape = im.shape
        self.n_control_points = n_control_points
        self.order = order

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        h, w, _ = self.shape 

        # query_points: generate every pixel position, on the ([0,1], [0,1]) range
        y, x = np.linspace(0, 1, h, endpoint=False), np.linspace(0, 1, w, endpoint=False)
        self.query_points = tf.constant(np.array(np.meshgrid(y, x)).T.reshape(1, -1, 2).astype(np.float32)) 

        # If mask is given as a threshold, create the real mask (as ndarray) from the image
        if isinstance(self.mask, float): 
            threshold = self.mask
            # TODO: (0.0 < self.im) es para evitar áreas sin señal, pero ´
            # sólo se aplica si se define mask como umbral
            self.mask = ((0.0 < self.im) & (self.im < threshold)).astype(np.float32) 
            # 1 on pixels below threshold, 0 otherwise

        # The mask (given as attr, or determined by threshold) is assumed to be an ndarray (image)
        # Generate the pixel locations not masked
        if self.mask is not None:
            assert(self.mask.shape == self.im.shape)
            idx = np.transpose(self.mask.squeeze(axis=-1).nonzero()) 
                # idx = ndarray with unmasked pixel locations, shape=(number of unmasked pixels, 2)
            #self.idx = idx # debug

            # Save the mask as a tensor, for later use in loss function
            self.mask = tf.constant(self.mask)
        else:
            idx = None

        # Generate train_coords (pixel locations) and then train_points (coords mapped to [0,1))
        try:
            # Sample randomly unmasked pixel locations
            n_unmasked_pixels = idx.shape[0]
            if n_unmasked_pixels == 0:
                print("Warning: empty mask, ignoring")
             
            random_sample = np.random.choice(n_unmasked_pixels, size=self.n_control_points, replace=False) 
                # replace=False so we don't repeat pixel locations            
            
            # Get sampled pixel locations
            train_coords = idx[random_sample]

            # Map coords to [0,1). Explicit float32 dtype.
            train_points = np.divide(train_coords, np.array([h, w]), dtype=np.float32) 
            # Expand for batch axis
            train_points = np.expand_dims(train_points, axis=0) 
            # Create layer weights for train_points
            self.train_points = tf.Variable(initial_value=train_points, trainable=True)

        except (AttributeError, ValueError):

            # ... then either
            #   mask==None (idx is None => AttributeError at idx.shape)
            # or 
            #   mask is empty (n_unmasked_pixels == 0 => ValueError at np.random.choice)
            # In these cases, generate train_points randomly distributed over ([0,1], [0,1]) range
            # TODO: generar train_coords como enteros y luego hacer común el código de generar train_points?
            train_points_initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
            self.train_points = self.add_weight(
                shape=(1, self.n_control_points, 2), initializer=train_points_initializer, trainable=True
            )
            train_coords = self.train_points.numpy().squeeze(axis=0)
            train_coords = (train_coords * np.array([h, w])).astype(np.int32) # nearest pixel position
        
        # w, v spline coefficients: initialize from actual image pixel values on the train_points 
        # for faster convergence (much faster than random initialization)
        # TODO: apply the mask here, to weight the initial train_values?
        train_values = self.im[ train_coords[:,0], train_coords[:,1], 0 ]
        # Expand for batch and output axis
        train_values = np.expand_dims(train_values, axis=(0,-1))

        # We use tfa.image._solve_interpolation for the initial spine coefficient values
        # TODO: make regularization_weight an argument?
        ww, vw = _solve_interpolation(
            self.train_points, train_values, self.order, regularization_weight=0.1
        ) 
        # Create layer weights for w and v spline coefficients
        self.ww = tf.Variable(initial_value=ww, trainable=True)
        self.vw = tf.Variable(initial_value=vw, trainable=True)

        super(Spline, self).build(input_shape)
      

    def call(self, inputs):
        h, w, _ = self.shape

        # Clip the train_points to the [0,1] box to constrain them
        # TODO: make this optional?
        #train_points = keras.backend.clip(self.train_points, 0.0, 1.0)
        train_points = self.train_points

        # Broadcast the weights and query_points to the batch_size
        train_points = tf.repeat(train_points, self.batch_size, axis=0)
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
        chunk_sz = math.ceil((h*w) / chunks)
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
