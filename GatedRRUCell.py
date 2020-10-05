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
import numpy as np
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


def inv_sigmoid(y):
    return np.log(y / (1 - y))


# residual_weight = 0.95  # r
# candidate_weight = np.sqrt(1 - residual_weight ** 2) * 0.25  # h
# S_initial_value = inv_sigmoid(residual_weight)


class RRUCell(LayerRNNCell):
    """Residual Recurrent Unit cell.
    Note that this cell is not optimized for performance. It might be wise to implement it with
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

    def __init__(self,
                 num_units,
                 group_size=32,
                 activation=None,
                 reuse=None,
                 dropout_rate=0.1,
                 residual_weight_initial_value=0.95,  # in range (0 - 1]
                 name=None,
                 dtype=None,
                 **kwargs):
        super(RRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "We could try to implement RRU with tf.contrib.cudnn_rnn.CudnnGRU as base for better "
                "performance on GPU.", self)
        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = None
        self._bias_initializer = tf.zeros_initializer()
        self._group_size = group_size
        self._dropout_rate = dropout_rate
        assert residual_weight_initial_value > 0 and residual_weight_initial_value <= 1
        self.residual_weight_initial_value = residual_weight_initial_value

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
        total = input_depth + self._num_units
        n_middle_maps = 2 * total  # TODO: find the optimal value
        self._Z_kernel = self.add_variable(
            "Z/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[total, n_middle_maps],
            initializer=self._kernel_initializer)
        self._Z_bias = self.add_variable(
            "Z/%s" % _BIAS_VARIABLE_NAME,
            shape=[n_middle_maps],
            initializer=self._bias_initializer)
        self.S_bias_variable = self.add_variable(
            "S/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer = init_ops.constant_initializer(inv_sigmoid(self.residual_weight_initial_value / 1.5), dtype=self.dtype))
        self.S_bias = tf.sigmoid(self.S_bias_variable)*1.5
        self._W_kernel = self.add_variable(
            "W/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[n_middle_maps, self._num_units*2],
            initializer=self._kernel_initializer)
        self._W_bias = self.add_variable(
            "W/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units*2],
            initializer=self._bias_initializer)
        self._W_mul = self.add_variable(
            "W_smul/%s"% _BIAS_VARIABLE_NAME,
            shape=(),
            initializer=tf.zeros_initializer())

        self.prev_state_weight = self.add_variable(  # TODO: check if needed
            "prev_state_weight/%s" % _BIAS_VARIABLE_NAME,
            shape=(),
            initializer=tf.ones_initializer())

        self.built = True

    def call(self, inputs, state):
        """Residual recurrent unit (RRU) with nunits cells."""
        _check_rnn_cell_input_dtypes([inputs, state])

        # LOWER PART OF THE CELL
        # Concatenate input and last state
        # state_drop = tf.nn.dropout(state, rate = self._dropout_rate)
        state_drop = state
        input_and_state = array_ops.concat([inputs, state_drop], 1)  # Inputs are batch_size x depth

        # Go through first transformation – Z
        after_z = math_ops.matmul(input_and_state, self._Z_kernel) + self._Z_bias

        # Do group normalization
        # group_size = self._group_size
        # # Do normalization
        # if group_size is None or group_size < 1 or group_size >= after_z.shape[1]:
        #     # Do instance normalization
        #     after_norm = instance_norm(after_z)
        # else:
        #     # Do group normalization
        #     after_norm = group_norm(after_z, group_size)

        # Do instance normalization
        after_norm = instance_norm(after_z)

        # Do GELU activation
        after_gelu = gelu(after_norm)
        # Do ReLU activation
        # after_gelu = tf.nn.relu(after_norm)

        # Go through the second transformation - W
        after_w = math_ops.matmul(after_gelu, self._W_kernel) + self._W_bias
        after_w, gate = tf.split(after_w, 2, axis=-1)
        gate = tf.sigmoid(gate+1)

        # Merge upper and lower parts
        # final = math_ops.sigmoid(self._S_bias) * state + after_w * candidate_weight
        # final = state * self.S_bias + after_w * self._W_mul#*np.sqrt(1.0/200)
        final = state*gate+after_w*(1-gate)

        return final, final

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "kernel_initializer": initializers.serialize(self._kernel_initializer),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(RRUCell, self).get_config()
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
        raise ValueError("RRU cell only supports floating point inputs, " "but saw dtype: %s" % dtype)


def gelu(x):
    return x * tf.sigmoid(1.702 * x)


def group_norm(cur, group_size):
    shape = tf.shape(cur)  # runtime shape
    n_units = cur.get_shape().as_list()[-1]  # static shape
    n_groups = n_units//group_size
    assert group_size*n_groups == n_units
    cur = tf.reshape(cur, [-1]+[n_groups]+[group_size])
    cur = instance_norm(cur)
    cur = tf.reshape(cur, shape)
    return cur


def instance_norm(cur):
    """Normalize each element based on variance"""
    variance = tf.reduce_mean(tf.square(cur), [-1], keepdims=True)
    cur = cur * tf.rsqrt(variance + 1e-6)
    return cur
