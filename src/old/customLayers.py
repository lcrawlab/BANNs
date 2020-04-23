import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import backend as K
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow_probability.python.math import random_rademacher

from tensorflow.python.eager import context
from tensorflow.python.framework import *
from tensorflow.python.keras import activations,constraints,initializers,regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import *
from tensorflow.python.ops import *
from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.util.tf_export import keras_export
# from tensorflow.python.distributions.bernoulli import Bernoulli

class Deterministic(tf.keras.layers.Layer):
  def __init__(self,
               units,
               mask,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(Deterministic, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.units = int(units)
    self.mask=mask
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs)
    rank = common_shapes.rank(inputs)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.mask*self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      # Cast the inputs to self.dtype, which is the variable dtype. We do not
      # cast if `should_cast_variables` is True, as in that case the variable
      # will be automatically casted to inputs.dtype.
      if not self._mixed_precision_policy.should_cast_variables:
        inputs = math_ops.cast(inputs, self.dtype)
      outputs = gen_math_ops.mat_mul(inputs, self.mask*self.kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(Dense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class Probabilistic(tf.keras.layers.Layer):
  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,

      slab_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      slab_posterior_tensor_fn=lambda d: d.sample(),
      slab_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      slab_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),

      spike_posterior_fn=tfd.Uniform(low=tf.math.log(1/1000), high=tf.math.log(1.0)),
      spike_posterior_tensor_fn=lambda d: d.sample(),
      spike_prior_fn=tfd.Uniform(low=tf.math.log(1/1000), high=tf.math.log(1.0)),
      spike_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),

      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None, ### WHY???
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):

    super(Probabilistic, self).__init__(
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.units = units
    self.activation = tf.keras.activations.get(activation)
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

    self.slab_posterior_fn = slab_posterior_fn
    self.slab_posterior_tensor_fn = slab_posterior_tensor_fn
    self.slab_prior_fn = slab_prior_fn
    self.slab_divergence_fn = slab_divergence_fn

    self.spike_posterior_fn = spike_posterior_fn
    self.spike_posterior_tensor_fn = spike_posterior_tensor_fn
    self.spike_prior_fn = spike_prior_fn
    self.spike_divergence_fn = spike_divergence_fn

    self.bias_posterior_fn = bias_posterior_fn
    self.bias_posterior_tensor_fn = bias_posterior_tensor_fn
    self.bias_prior_fn = bias_prior_fn
    self.bias_divergence_fn = bias_divergence_fn

    # super(_DenseVariational, self)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    in_size = tf.compat.dimension_value(input_shape.with_rank_at_least(2)[-1])
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # BUILD SLAB DISTRIBUTION
    self.slab_posterior = self.slab_posterior_fn(
        dtype, [in_size, self.units], 'slab_posterior',
        self.trainable, self.add_variable)
    if self.slab_prior_fn is None:
      self.slab_prior = None
    else:
      self.slab_prior = self.slab_prior_fn(
          dtype, [in_size, self.units], 'slab_prior',
          self.trainable, self.add_variable)
    self._built_slab_divergence = False


    # BUILD SPIKE DISTRIBUTION
    self.spike_posterior = self.spike_posterior_fn(
        dtype, [in_size, self.units], 'spike_posterior',
        self.trainable, self.add_variable)

    if self.spike_prior_fn is None:
      self.spike_prior = None
    else:
      self.spike_prior = self.spike_prior_fn(
          dtype, [in_size, self.units], 'spike_prior',
          self.trainable, self.add_variable)
    self._built_spike_divergence = False

    # BUILD BIAS DISTRIBUTION
    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs):
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)

    # PERFORM THE TENSOR COMPUTATIONS FOR FEED-FORWARD
    outputs = self._apply_variational_kernel(inputs) #!! self._apply_variational_spike_slab(inputs)
    outputs = self._apply_variational_bias(outputs)
    if self.activation is not None:
      outputs = self.activation(outputs)

    # COMPUTE THE VARIATION LOSS
    if not self._built_slab_divergence:
      self._apply_divergence(self.slab_divergence_fn,
                             self.slab_posterior,
                             self.slab_prior,
                             self.slab_posterior_tensor,
                             name='divergence_slab')
      self._built_slab_divergence = True

    if not self._built_spike_divergence:
      self._apply_divergence(self.spike_divergence_fn,
                             self.spike_posterior,
                             self.spike_prior,
                             self.spike_posterior_tensor,
                             name='divergence_spike')
      self._built_spike_divergence = True

    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    """Returns the config of the layer:
    A layer config is a Python dictionary (serializable) containing the
    configuration of a layer. The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    Returns:
      config: A Python dictionary of class keyword arguments and their
        serialized values.
    """
    config = {
        'units': self.units,
        'activation': (tf.keras.activations.serialize(self.activation)
                       if self.activation else None),
        'activity_regularizer':
            tf.keras.initializers.serialize(self.activity_regularizer),
    }
    function_keys = [
        'slab_posterior_fn',
        'slab_posterior_tensor_fn',
        'slab_prior_fn',
        'slab_divergence_fn',
        'spike_posterior_fn',
        'spike_posterior_tensor_fn',
        'spike_prior_fn',
        'spike_divergence_fn',
        'bias_posterior_fn',
        'bias_posterior_tensor_fn',
        'bias_prior_fn',
        'bias_divergence_fn',
    ]
    for function_key in function_keys:
      function = getattr(self, function_key)
      if function is None:
        function_name = None
        function_type = None
      else:
        function_name, function_type = tfp_layers_util.serialize_function(
            function)
      config[function_key] = function_name
      config[function_key + '_type'] = function_type
    base_config = super(_DenseVariational, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    """Creates a layer from its config:
    This method is the reverse of `get_config`, capable of instantiating the
    same layer from the config dictionary.

    Args: config: A Python dictionary, typically the output of `get_config`.

    Returns:
      layer: A layer instance.
    """
    config = config.copy()
    function_keys = [
        'slab_posterior_fn',
        'slab_posterior_tensor_fn',
        'slab_prior_fn',
        'slab_divergence_fn',
        'spike_posterior_fn',
        'spike_posterior_tensor_fn',
        'spike_prior_fn',
        'spike_divergence_fn',
        'bias_posterior_fn',
        'bias_posterior_tensor_fn',
        'bias_prior_fn',
        'bias_divergence_fn',
    ]
    for function_key in function_keys:
      serial = config[function_key]
      function_type = config.pop(function_key + '_type')
      if serial is not None:
        config[function_key] = tfp_layers_util.deserialize_function(
            serial,
            function_type=function_type)
    return cls(**config)

  def _apply_variational_kernel(self, inputs):

    # Just some warning messages based on checking the distribution types for spike and slab variables:
    if (not isinstance(self.slab_posterior, tfd.Independent) or
        not isinstance(self.slab_posterior.distribution, tfd.Normal)):
      raise TypeError(
          '`Probabilistic layer` requires '
          '`slab_posterior_fn` produce an instance of '
          '`tfd.Independent(tfd.Normal)` '
          '(saw: \"{}\").'.format(self.slab_posterior.name)) ## REPLICATE THIS FOR BOTH
    
    if (not isinstance(self.spike_posterior, tfd.Independent) or
        not isinstance(self.spike_posterior.distribution, tfd.Normal)):
      raise TypeError(
          '`Probabilistic layer` requires '
          '`spike_posterior_fn` produce an instance of '
          '`tfd.Independent(tfd.Normal)` '
          '(saw: \"{}\").'.format(self.spike_posterior.name)) 

    self.bernoullis=tfd.Bernoulli(logits=self.spike_posterior.distribution.loc)
    self.kernel_posterior_affine = tfd.Normal(
        loc=self._matmul(inputs, self.self.bernoullis*self.slab_posterior.distribution.loc),
        scale=tf.sqrt(self._matmul(
            tf.square(inputs),
            tf.square(self.self.bernoullis*self.slab_posterior.distribution.scale))))

    self.kernel_posterior_affine_tensor = (
        self.slab_posterior_tensor_fn(self.kernel_posterior_affine)) #samples from posterior_affine

    self.slab_posterior_tensor = None ###!!!??
    self.spike_posterior_tensor = None ###!!!??

    return self.kernel_posterior_affine_tensor

  def _apply_variational_bias(self, inputs):
    if self.bias_posterior is None:
      self.bias_posterior_tensor = None
      return inputs
    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    return tf.nn.bias_add(inputs, self.bias_posterior_tensor)

  def _apply_divergence(self, divergence_fn, posterior, prior,
                        posterior_tensor, name):
    if (divergence_fn is None or
        posterior is None or
        prior is None):
      divergence = None
      return
    divergence = tf.identity(
        divergence_fn(
            posterior, prior, posterior_tensor),
        name=name)
    self.add_loss(divergence)

  def _matmul(self, inputs, kernel):
    if inputs.shape.ndims <= 2:
      return tf.matmul(inputs, kernel)
    # To handle broadcasting, we must use `tensordot`.
    return tf.tensordot(inputs, kernel, axes=[[-1], [0]])

  # def _mulElementWise(self,spike,kernel):
  #   return 

