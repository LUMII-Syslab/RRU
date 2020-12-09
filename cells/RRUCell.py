# Code reference - https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/ops/rnn_cell_impl.py#L484-L614
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
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
        z_transformations: int, The number of Z transformations in the RRU cell.
        middle_layer_size_multiplier: int, The size multiplier for transformation layer in the RRU cell.
        dropout_rate: float, The dropout rate used after the W transformation.
        residual_weight_initial_value: float, Value of the residual weight (must be in range (0 - 1]).
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
                 z_transformations=1,
                 middle_layer_size_multiplier=2,  # TODO: find the optimal value (this goes to n_middle_maps)
                 dropout_rate=0.5,
                 residual_weight_initial_value=0.95,
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
        self._z_transformations = z_transformations
        self._middle_layer_size_multiplier = middle_layer_size_multiplier
        self._dropout_rate = dropout_rate
        assert 0 < residual_weight_initial_value <= 1
        self._residual_weight_initial_value = residual_weight_initial_value
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
        self._Z_kernel = []
        self._Z_bias = []
        for i in range(self._z_transformations):
            if i == 0:  # The first Z transformation has a different shape
                z_kernel = self.add_variable(
                    "Z/%s" % _WEIGHTS_VARIABLE_NAME,
                    shape=[total, n_middle_maps],
                    initializer=self._kernel_initializer)
                z_bias = self.add_variable(
                    "Z/%s" % _BIAS_VARIABLE_NAME,
                    shape=[n_middle_maps],
                    initializer=self._bias_initializer)
            else:
                z_kernel = self.add_variable(
                    f"Z{i + 1}/%s" % _WEIGHTS_VARIABLE_NAME,
                    shape=[n_middle_maps, n_middle_maps],
                    initializer=self._kernel_initializer)
                z_bias = self.add_variable(
                    f"Z{i + 1}/%s" % _BIAS_VARIABLE_NAME,
                    shape=[n_middle_maps],
                    initializer=self._bias_initializer)
            self._Z_kernel.append(z_kernel)
            self._Z_bias.append(z_bias)
        self.S_bias_variable = self.add_variable(
            "S/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=init_ops.constant_initializer(inv_sigmoid(self._residual_weight_initial_value / 1.5), dtype=self.dtype))
        self.S_bias = tf.sigmoid(self.S_bias_variable) * 1.5
        self._W_kernel = self.add_variable(
            "W/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[n_middle_maps, self._num_units + self._output_size],
            initializer=self._kernel_initializer)
        self._W_bias = self.add_variable(
            "W/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units + self._output_size],
            initializer=self._bias_initializer)
        self._W_mul = self.add_variable(
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

        # Concatenate input and last state
        input_and_state = array_ops.concat([inputs, state], 1)  # Inputs are batch_size x depth

        # Go through first transformation(s) - Z
        z_start = input_and_state  # This will hold the info that each Z transformation has to transform
        for i in range(self._z_transformations):
            # Multiply the matrices
            after_z = math_ops.matmul(z_start, self._Z_kernel[i]) + self._Z_bias[i]

            if i == 0:  # For the first Z transformation do normalization
                # Do instance normalization
                after_z = instance_norm(after_z)

            # Do ReLU activation
            after_activation = tf.nn.relu(after_z)

            # Update the z_start variable with the newest values
            z_start = after_activation

        # Apply dropout
        after_dropout = tf.nn.dropout(z_start, rate=dropout_rate)

        # Go through the second transformation - W
        after_w = math_ops.matmul(after_dropout, self._W_kernel) + self._W_bias

        # Calculate the output
        output = after_w[:, self._num_units:]

        # Calculate the state's final values
        candidate = after_w[:, 0:self._num_units]

        final_state = state * self.S_bias + candidate * self._W_mul

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
            "z_transformations": self._z_transformations,
            "middle_layer_size_multiplier": self._middle_layer_size_multiplier,
            "dropout_rate": self._dropout_rate,
            "residual_weight_initial_value": self._residual_weight_initial_value
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
