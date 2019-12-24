from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import layers
import warnings


#metric losses
def ML_loss(l=1.):
    warnings.warn("do not use smothed labels for ML_loss, only ones and zeros are available")
    def loss(ytr, ypr):
        val = K.max(K.switch(ytr <= 0.5, l - ypr, ypr), axis=-1)
        
        return K.mean(val)
    
    return loss


def SoftML_loss(l=1.):
    warnings.warn("do not use smothed labels for SoftML_loss, only ones and zeros are available")
    def loss(ytr, ypr):
        val = K.sum(K.switch(ytr >= 0.5, ypr, K.log(1 + K.exp(l - ypr))), axis=-1)
        
        return K.mean(val)
    
    return loss
                    
                    
#class Layer
class RBFDense(layers.Layer):
    """densely-connected RBF NN layer
    
    Convert inputs into cluster space
    
    Note: this layer is useful as the last layer of the network

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        model.add(RBFDense(10, inner_dim=None))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        
        inner_dim: Positive integer or None, dimensionality of A matrix(trainable)
            or set it as Identity matrix(non-trainable).
            
        norm_type: String "l1" or "l2", type of norm which will be used.
        
        use_gamma: Boolean, whether the layer uses a gamma vector to convert output to
            (gamma - distance), so you can use standart losses. Be careful 
            there are no scientifically correct experiments on the trained gamma =)
        
        use_bias: Boolean, whether the layer uses a bias vector.
        
        kernel_initializer: Initializer for the `kernel` weights matrix.
        
        bias_initializer: Initializer for the bias vector.
        
        gamma_initializer: Initializer for the gamma vector.
        
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
            
        bias_regularizer: Regularizer function applied to the bias vector.
        
        gamma_regularizer: Regularizer function applied to the gamma vector.
        
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
            
        bias_constraint: Constraint function applied to the bias vector.
        
        gamma_constraint: Constraint function applied to the gamma vector.

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """
    def __init__(self, units,
                 inner_dim=2,
                 use_gamma=False,
                 norm_type="l2",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 gamma_initializer='ones',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 gamma_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        
        if not ((isinstance(inner_dim, int) and inner_dim > 0) or inner_dim is None):
            raise RunTimeError("inner_dim must be an integer greater than 0 or None")
        self.inner_dim = inner_dim
            
        if not use_bias and inner_dim is None:
            raise RunTimeError("only one parameter use_bias or inner_dim may be False or None")
        self.use_bias = use_bias
        
        if not norm_type in ["l1", "l2"]:
            raise NotImplementError("use only 'l1' or 'l2' for norm_type")
        self.norm_type = norm_type
        
        self.units = units
        self.use_gamma = use_gamma
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        
        super(RBFDense, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        if not self.inner_dim is None:
            bias_shape = self.units * self.inner_dim
            
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_dim, self.units * self.inner_dim),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        else:
            bias_shape = input_dim * self.units
            self.kernel = None
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(bias_shape,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        
        if self.use_gamma:
            self.gamma = self.add_weight(name='gamma', 
                                         shape=(self.units,),
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        
        super(RBFDense, self).build(input_shape)


    def call(self, x):
        reshaping = self.inner_dim or int(x.shape[-1])
        if self.inner_dim is None:
            x = K.tile(x, tuple([1] * (len(x.shape)-1) + [self.units]))
        else:
            x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias, data_format='channels_last')
            
        if self.norm_type == "l2":
            x = K.square(x)
            x = K.expand_dims(x, -1)
            shape = [K.shape(x)[k] for k in range(len(x.shape))]
            shape[-2] = self.units
            shape[-1] = reshaping
            x = K.reshape(x, shape)
            x = K.sum(x, axis=-1)
            x = K.sqrt(x)
        else:
            x = K.abs(x)
            x = K.expand_dims(x, -1)
            shape = [K.shape(x)[k] for k in range(len(x.shape))]
            shape[-2] = self.units
            shape[-1] = reshaping
            x = K.reshape(x, shape)
            x = K.sum(x, axis=-1)
        
        if self.use_gamma:
            x = self.gamma - x
        
        return x


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


    def get_config(self):
        config = {
            'units': self.units,
            'inner_dim': self.inner_dim,
            'use_gamma': self.use_gamma,
            'norm_type': self.norm_type,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(RBFDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    



class SoftMin(layers.Layer):
    """Siftmin activation layer
    
    Note: use it only for RBF-layer without gamma

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        model.add(RBFDense(10, inner_dim=None))
        model.add(SoftMin(lambd=1., rejection_class=True))
    ```

    # Arguments
        
        rejection_class: Boolean, added extra class, if we need to reject sample.
    
        lambd: Positive Float, rejection coeficient, ignore if rejection_class=False.
        
        axis: positive integer, axis through whitch calculate softmin.
        
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
        If rejection_class=True, would have `(batch_size, units+1)`
    """
    def __init__(self, rejection_class=False, lambd=1., axis=-1, **kwargs):
        self.rejection_class = rejection_class
        if rejection_class:
            assert lambd > 0, "lambd must be greater than 0."
            self.lambd = lambd
        self.axis = axis
        super(SoftMin, self).__init__(**kwargs)
    
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        last = int(output_shape[self.axis])
        if self.rejection_class:
            last += 1
        output_shape[self.axis] = last
        return tuple(output_shape)
    
    
    def call(self, x):
        if self.rejection_class:
            shape = [K.shape(x)[k] for k in range(len(x.shape))]
            shape[self.axis] = 1
            x = K.concatenate([x, K.ones(shape) * self.lambd])
        
        x = K.exp(-x)
        
        sx = K.sum(x, axis=self.axis, keepdims=True)
        
        softmax = x / sx
        
        return softmax


    def get_config(self):
        config = {
            'rejection_class': self.rejection_class,
            'lambd': self.lambd,
            'axis': self.axis,
        }
        base_config = super(RBFDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    

                    