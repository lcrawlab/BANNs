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
import numpy as np
# from tensorflow.python.distributions.bernoulli import Bernoulli

class SNP_Layer(tf.keras.layers.Layer):
  def __init__(self,
               units,
               mask,
               activation=None,
               use_bias=True,
               prob_initializer='RandomUniform',
               kernel_initializer='RandomNormal',
               bias_initializer='zeros',
               prob_regularizer=None,
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               prob_constraint=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(SNP_Layer, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    # self.last_dim=last_dim
    self.units = int(units)
    self.mask=mask
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.prob_initializer = initializers.get(prob_initializer)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.prob_regularizer = regularizers.get(prob_regularizer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.prob_constraint = constraints.get(prob_constraint)
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

    self.prob=self.add_weight(
        'prob',
        shape=[last_dim, self.units],
        initializer=self.prob_initializer,
        regularizer=self.prob_regularizer,
        constraint=self.prob_constraint,
        dtype=self.dtype,
        trainable=True)
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
      # logits=tf.math.log(self.prob/(1-se lf.prob))
      # temperature=0.2
      dist = tfp.distributions.Logistic(self.prob, 2)
      samples = dist.sample()
      sigmoid_samples = tf.sigmoid(samples)
      #self.spike= tfp.distributions.RelaxedBernoulli(temperature=0.8, probs=self.prob).sample() #Normal(loc=self.prob,scale=0.0 )
      #tf.nn.relu(self.prob-0.4) #tfp.distributions.RelaxedBernoulli(temperature=0.1, probs=[])
      #mask=tf.random.normal(mean=self.prob, shape=[self.last_dim, self.units])
      # mask = tfp.distributions.Normal(loc=self.prob,scale=0.0 ).sample(1000)#mask = tfd.Normal(loc=self.prob, scale=0.01).sample()
      outputs = standard_ops.tensordot(inputs, (sigmoid_samples*self.mask)*self.kernel, [[rank - 1], [0]])
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
      # spike=tfp.distributions.RelaxedBernoulli(temperature=0.1, probs=self.prob).sample()
      #mask=tf.random.normal(mean=self.prob, shape=[self.last_dim, self.units])
      #self.spike= tfp.distributions.RelaxedBernoulli(temperature=0.8, probs=self.prob).sample()
      # logits=tf.math.log(self.prob/(1-self.prob))
      # temperature=0.2
      dist = tfp.distributions.Logistic(self.prob, 2)
      samples = dist.sample()
      sigmoid_samples = tf.sigmoid(samples)
      #self.spike=tfp.distributions.RelaxedBernoulli(temperature=0.1, probs=0.1).sample(1000)
      # RelaxedBernoulli(temperature=0.8, probs=self.prob).sample() #tfd.Normal(loc=self.prob, scale=0.01).sample()
      outputs = gen_math_ops.mat_mul(inputs, (sigmoid_samples*self.mask)*self.kernel)
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


class Gene_Layer(tf.keras.layers.Layer):
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               prob_initializer='RandomUniform',
               kernel_initializer='RandomNormal',
               bias_initializer='zeros',
               prob_regularizer=None,
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               prob_constraint=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(Gene_Layer, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    # self.last_dim=last_dim
    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.prob_initializer = initializers.get(prob_initializer)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.prob_regularizer = regularizers.get(prob_regularizer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.prob_constraint = constraints.get(prob_constraint)
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

    self.prob=self.add_weight(
        'prob',
        shape=[last_dim, self.units],
        initializer=self.prob_initializer,
        regularizer=self.prob_regularizer,
        constraint=self.prob_constraint,
        dtype=self.dtype,
        trainable=True)
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
      # logits=tf.math.log(self.prob/(1-se lf.prob))
      # temperature=0.2
      dist = tfp.distributions.Logistic(self.prob, 0.5)
      samples = dist.sample()
      sigmoid_samples = tf.sigmoid(samples)
      #self.spike= tfp.distributions.RelaxedBernoulli(temperature=0.8, probs=self.prob).sample() #Normal(loc=self.prob,scale=0.0 )
      #tf.nn.relu(self.prob-0.4) #tfp.distributions.RelaxedBernoulli(temperature=0.1, probs=[])
      #mask=tf.random.normal(mean=self.prob, shape=[self.last_dim, self.units])
      # mask = tfp.distributions.Normal(loc=self.prob,scale=0.0 ).sample(1000)#mask = tfd.Normal(loc=self.prob, scale=0.01).sample()
      outputs = standard_ops.tensordot(inputs, sigmoid_samples*self.kernel, [[rank - 1], [0]])
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
      # spike=tfp.distributions.RelaxedBernoulli(temperature=0.1, probs=self.prob).sample()
      #mask=tf.random.normal(mean=self.prob, shape=[self.last_dim, self.units])
      #self.spike= tfp.distributions.RelaxedBernoulli(temperature=0.8, probs=self.prob).sample()
      # logits=tf.math.log(self.prob/(1-self.prob))
      # temperature=0.2
      dist = tfp.distributions.Logistic(self.prob, 0.5)
      samples = dist.sample()
      sigmoid_samples = tf.sigmoid(samples)
      #self.spike=tfp.distributions.RelaxedBernoulli(temperature=0.1, probs=0.1).sample(1000)
      # RelaxedBernoulli(temperature=0.8, probs=self.prob).sample() #tfd.Normal(loc=self.prob, scale=0.01).sample()
      outputs = gen_math_ops.mat_mul(inputs, sigmoid_samples*self.kernel)
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

