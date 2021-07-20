# Code reference - https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/ops/rnn_cell_impl.py#L484-L614
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, _check_supported_dtypes, _check_rnn_cell_input_dtypes

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class RRUCell(LayerRNNCell):
    """Residual Recurrent Unit (RRU) cell.
    Args:
        num_units: int, The number of units in the RRU cell.
        output_size: int, The size of the RRU output.
        relu_layers: int, The number of Z transformations in the RRU cell.
        middle_layer_size_multiplier: int, The size multiplier for transformation layer in the RRU cell.
        dropout_rate: float, The dropout rate used after the W transformation.
        training: boolean, To let know whether the RRU cell is in training or not.
        reuse: (optional) Python boolean describing whether to reuse variables in an
            existing scope.  If not `True`, and the existing scope already has the
            given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will share
            weights, but to avoid mistakes we require reuse=True in such cases.
        dtype: Default dtype of the layer (default of `None` means use the type of
            the first input). Required when `build` is called before `call`.
        **kwargs: Dict, keyword named properties for common layer attributes, like
            `trainable` etc when constructing the cell from configs of get_config().
    """

    def __init__(self,
                 num_units,
                 output_size=256,
                 relu_layers=1,
                 middle_layer_size_multiplier=2,
                 dropout_rate=0.5,
                 training=False,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(RRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        self._output_size = output_size
        self._relu_layers = relu_layers
        self._middle_layer_size_multiplier = middle_layer_size_multiplier
        self._dropout_rate = dropout_rate
        self._training = training
        self._kernel_initializer = None
        self._bias_initializer = tf.zeros_initializer()

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_size

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        total = input_depth + self._num_units
        n_middle_maps = round(self._middle_layer_size_multiplier * total)
        self.mul_lr_multiplier = 10.  # So adam optimizer uses 10 times larger learning rate for this parameter
        self._J_kernel = []
        self._J_bias = []
        for i in range(self._relu_layers):
            if i == 0:  # The first Z transformation has a different shape
                j_kernel = self.add_variable(
                    "J/%s" % _WEIGHTS_VARIABLE_NAME,
                    shape=[total, n_middle_maps],
                    initializer=self._kernel_initializer)
                j_bias = self.add_variable(
                    "J/%s" % _BIAS_VARIABLE_NAME,
                    shape=[n_middle_maps],
                    initializer=self._bias_initializer)
            else:
                j_kernel = self.add_variable(
                    f"J{i + 1}/%s" % _WEIGHTS_VARIABLE_NAME,
                    shape=[n_middle_maps, n_middle_maps],
                    initializer=self._kernel_initializer)
                j_bias = self.add_variable(
                    f"J{i + 1}/%s" % _BIAS_VARIABLE_NAME,
                    shape=[n_middle_maps],
                    initializer=self._bias_initializer)
            self._J_kernel.append(j_kernel)
            self._J_bias.append(j_bias)
        self.S_bias_variable = self.add_variable(
            "S/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=init_ops.constant_initializer(inv_sigmoid(np.random.uniform(0.01, 0.99, size=self._num_units))
                                                      / self.mul_lr_multiplier, dtype=self.dtype))
        self._W_kernel = self.add_variable(
            "W/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[n_middle_maps, self._num_units + self._output_size],
            initializer=self._kernel_initializer)
        self._W_bias = self.add_variable(
            "W/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units + self._output_size],
            initializer=self._bias_initializer)
        self._Z_ReZero = self.add_variable(
            "W_smul/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=tf.zeros_initializer())

        self.built = True

    def call(self, inputs, state):
        _check_rnn_cell_input_dtypes([inputs, state])

        # If the cell is training, we should apply the given dropout
        if self._training:
            dropout_rate = self._dropout_rate
        else:
            dropout_rate = 0.

        # Concatenate input and last state (for faster calculation)
        input_and_state = array_ops.concat([inputs, state], 1)  # Inputs are batch_size x depth

        # Go through first transformation(s) - J
        j_start = input_and_state  # This will hold the info that each J transformation has to transform
        for i in range(self._relu_layers):
            # Multiply the matrices
            after_j = math_ops.matmul(j_start, self._J_kernel[i]) + self._J_bias[i]

            if i == 0:  # For the first J transformation do normalization
                # Do instance normalization
                after_j = instance_norm(after_j)

            # Do ReLU activation
            after_activation = tf.nn.relu(after_j)

            # Update the j_start variable with the newest values
            j_start = after_activation

        # Apply dropout
        after_dropout = tf.nn.dropout(j_start, rate=dropout_rate)

        # Go through the second transformation - W (W^c and W^o from the paper)
        after_w = math_ops.matmul(after_dropout, self._W_kernel) + self._W_bias

        # Calculate the output (o_t)
        output = after_w[:, self._num_units:]

        # Calculate the candidate value (c)
        candidate = after_w[:, 0:self._num_units]

        # Calculate the state's final values (h_t)
        final_state = state * tf.sigmoid(self.S_bias_variable * self.mul_lr_multiplier) + candidate * self._Z_ReZero

        return output, final_state

    def zero_state(self, batch_size, dtype):
        value = super().zero_state(batch_size, dtype)
        initial = np.asarray([1] + [0] * (self._num_units - 1)) * np.sqrt(self._num_units) * 0.25
        return value + initial

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "output_size": self._output_size,
            "reuse": self._reuse,
            "training": self._training,
            "relu_layers": self._relu_layers,
            "middle_layer_size_multiplier": self._middle_layer_size_multiplier,
            "dropout_rate": self._dropout_rate
        }
        base_config = super(RRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def instance_norm(cur):
    """Normalize each element based on variance"""
    variance = tf.reduce_mean(tf.square(cur), [-1], keepdims=True)
    cur = cur * tf.rsqrt(variance + 1e-6)
    return cur


def inv_sigmoid(y):
    return np.log(y / (1 - y))
