import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell

# Code adapted from https://github.com/ribeiromiranda/sru
# Junier B Oliva, Barnabás Póczos, and Jeff Schneider. The statistical recurrent unit. In International Conference on Machine Learning, pages 2671–2680, 2017.
# https://arxiv.org/abs/1703.00381
class SRUCell(LayerRNNCell):
    def __init__(self, num_stats, mavg_alphas, recur_dims,
                 learn_alphas=False, linear_out=False, include_input=False,
                 activation=tf.nn.relu, **kwargs):
        self._num_stats = num_stats
        #        self._output_dims = output_dims
        self._recur_dims = recur_dims
        if learn_alphas:
            init_logit_alphas = -tf.log(1.0 / mavg_alphas - 1)
            logit_alphas = K.variable(init_logit_alphas)
            self._mavg_alphas = K.reshape(K.sigmoid(logit_alphas), [1, -1, 1])
        else:
            self._mavg_alphas = K.reshape(mavg_alphas, [1, -1, 1])
        self._nalphas = len(mavg_alphas)
        self._linear_out = linear_out
        self._activation = activation
        self._include_input = include_input

        super(SRUCell, self).__init__(**kwargs)

    @property
    def units(self):
        return self.state_size

    @property
    def output_size(self):
        return self.state_size

    @property
    def state_size(self):
        return int(self._nalphas * self._num_stats)

    def build(self, input_shape):
        bias_start = 0.0
        if self._recur_dims > 0:
            self.recur_feats_matrix = self.add_weight(shape=(self.state_size, self._recur_dims), initializer='uniform',
                                                      name='recur_feats_matrix')
            self.recur_feats_bias = self.add_weight(shape=(self._recur_dims,),
                                                    initializer=tf.keras.initializers.Constant(0), name='recur_feats_bias')
            # self.recur_feats_bias = K.constant([bias_start], shape=(self._recur_dims,))

        rows = input_shape[-1]
        if self._recur_dims > 0:
            rows += self._recur_dims
        self.stats_matrix = self.add_weight(shape=(rows, self._num_stats), initializer='uniform', name='stats_matrix')
        self.stats_bias = self.add_weight(shape=(self._num_stats,), initializer=tf.keras.initializers.Constant(0),
                                          name='stats_bias')
        # self.stats_bias = K.constant(bias_start, shape=(self._num_stats,))

        rows = self.state_size
        if self._include_input > 0:
            rows += input_shape[-1]
        self.output_matrix = self.add_weight(shape=(rows, self.output_size), initializer='uniform',
                                             name='output_matrix')
        self.output_bias = self.add_weight(shape=(self.output_size,), initializer=tf.keras.initializers.Constant(0),
                                           name='output_bias')
        # self.output_bias = K.constant(bias_start, shape=(self.output_size,))

        self.built = True

    def call(self, inputs, states):

        if self._recur_dims > 0:
            recur_output = self._activation(self._linear(
                states, self.recur_feats_matrix, self.recur_feats_bias
            ))
            stats = self._activation(self._linear(
                [inputs, recur_output], self.stats_matrix, self.stats_bias
            ))
        else:
            stats = self._activation(self._linear(
                inputs, self.stats_matrix, self.stats_bias
            ))

        state_tensor = K.reshape(states, [-1, self._nalphas, self._num_stats])
        stats_tensor = K.reshape(stats, [-1, 1, self._num_stats])
        out_state = K.reshape(self._mavg_alphas * state_tensor + (1 - self._mavg_alphas) * stats_tensor,
                              [-1, self.state_size])

        # Compute the output.
        if self._include_input:
            output_vars = [out_state, inputs]
        else:
            output_vars = out_state

        output = self._linear(
            output_vars, self.output_matrix, self.output_bias
        )
        if not self._linear_out:
            output = self._activation(output)

        return output, out_state

    def _linear(self, args, matrix, bias):
        if type(args) == list:
            res = K.dot(K.concatenate(args, 1), matrix)
        else:
            res = K.dot(args, matrix)

        if bias is None:
            return res

        return res + bias