# Code reference - https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/ops/rnn_cell_impl.py#L484-L614
"""Module implementing RNN Cells.
This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class GRUCell(LayerRNNCell):
    """Gated Recurrent Unit cell.
    Note that this cell is not optimized for performance. Please use
    `tf.contrib.cudnn_rnn.CudnnGRU` for better performance on GPU, or
    `tf.contrib.rnn.GRUBlockCellV2` for better performance on CPU.
    Args:
        num_units: int, The number of units in the GRU cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables in an
            existing scope.  If not `True`, and the existing scope already has the
            given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
            projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
        name: String, the name of the layer. Layers with the same name will share
            weights, but to avoid mistakes we require reuse=True in such cases.
        dtype: Default dtype of the layer (default of `None` means use the type of
            the first input). Required when `build` is called before `call`.
        **kwargs: Dict, keyword named properties for common layer attributes, like
            `trainable` etc when constructing the cell from configs of get_config().
            References:
        Learning Phrase Representations using RNN Encoder Decoder for Statistical
        Machine Translation:
            [Cho et al., 2014]
            (https://aclanthology.coli.uni-saarland.de/papers/D14-1179/d14-1179)
            ([pdf](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf))
    """

    @deprecated(None, "This class is equivalent as tf.keras.layers.GRUCell,"
                      " and will be replaced by that in Tensorflow 2.0.")
    def __init__(self,
                 num_units,
                 activation=None,
                 dropout_rate=0.2,
                 training=False,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(GRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                "performance on GPU.", self)
        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._dropout_rate = dropout_rate
        self._training = training
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(self._bias_initializer
                         if self._bias_initializer is not None else
                         init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(self._bias_initializer
                         if self._bias_initializer is not None else
                         init_ops.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        _check_rnn_cell_input_dtypes([inputs, state])

        # If the cell is training, we should apply the given dropout
        if self._training:
            dropout_rate = self._dropout_rate
        else:
            dropout_rate = 0.

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)  # sigmoid

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)

        # Apply dropout as in paper "Recurrent Dropout without Memory Loss"
        c = tf.nn.dropout(c, rate=dropout_rate)

        new_h = u * state + (1 - u) * c
        return new_h, new_h

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "kernel_initializer": initializers.serialize(self._kernel_initializer),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _check_rnn_cell_input_dtypes(inputs):
    """Check whether the input tensors are with supported dtypes.
    Default RNN cells only support floats and complex as its dtypes since the
    activation function (tanh and sigmoid) only allow those types. This function
    will throw a proper error message if the inputs is not in a supported type.
    Args:
        inputs: tensor or nested structure of tensors that are feed to RNN cell as
            input or state.
        Raises:
            ValueError: if any of the input tensor are not having dtypes of float or
            complex.
    """
    for t in nest.flatten(inputs):
        _check_supported_dtypes(t.dtype)


def _check_supported_dtypes(dtype):
    if dtype is None:
        return
    dtype = dtypes.as_dtype(dtype)
    if not (dtype.is_floating or dtype.is_complex):
        raise ValueError("RNN cell only supports floating point inputs, " "but saw dtype: %s" % dtype)
