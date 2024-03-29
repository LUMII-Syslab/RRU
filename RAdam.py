import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import clip_ops

__all__ = ['RAdamOptimizer']


class RAdamOptimizer(optimizer.Optimizer):
    """RAdam optimizer.

    According to the paper
    [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-7,
                 L2_decay=0.,
                 amsgrad=False,
                 total_steps=0,
                 warmup_proportion=0.1,
                 min_lr=0.,
                 use_locking=False,
                 name="RAdam",
                 decay_vars = None,
                 L1_decay=0.0,
                 clip_gradients = False, clip_multiplier=3.0, clip_epsilon=1e-2):
        r"""Construct a new Adam optimizer.

        Args:
            learning_rate: A Tensor or a floating point value.    The learning rate.
            beta1: A float value or a constant float tensor. The exponential decay
                rate for the 1st moment estimates.
            beta2: A float value or a constant float tensor. The exponential decay
                rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability. This epsilon is
                "epsilon hat" in the Kingma and Ba paper (in the formula just before
                Section 2.1), not the epsilon in Algorithm 1 of the paper.
            L2_decay: A floating point value. Weight decay for each param.
            amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
                the paper "On the Convergence of Adam and beyond".
            total_steps: An integer. Total number of training steps.
                Enable warmup by setting a positive value.
            warmup_proportion: A floating point value. The proportion of increasing steps.
            min_lr: A floating point value. Minimum learning rate after warmup.
            name: Optional name for the operations created when applying gradients.
                Defaults to "Adam".    @compatibility(eager) When eager execution is
                enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each be
                a callable that takes no arguments and returns the actual value to use.
                This can be useful for changing these values across different
                invocations of optimizer functions. @end_compatibility
            **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
                `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
                gradients by value, `decay` is included for backward compatibility to
                allow time inverse decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super(RAdamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay = L2_decay
        self._L1_decay = L1_decay
        self._amsgrad = amsgrad
        self._total_steps = float(total_steps)
        self._warmup_proportion = warmup_proportion
        self._min_lr = min_lr
        self._initial_weight_decay = L2_decay
        self._initial_total_steps = total_steps
        self.clip_multiplier = clip_multiplier
        self.clip_epsilon = clip_epsilon
        self.clip_gradients = clip_gradients
        self.clip_multiplier_t = ops.convert_to_tensor(self.clip_multiplier, name="clip_multiplier")
        self.clip_epsilon_t = ops.convert_to_tensor(self.clip_epsilon, name="clip_epsilon")

        self._lr_t = None
        self._step_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._weight_decay_t = None
        self._total_steps_t = None
        self._warmup_proportion_t = None
        self._min_lr_t = None
        self.reg_vars = set(decay_vars) if decay_vars is not None else set()

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("step", graph=graph),
                    self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots_internal(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=1.0, name="step", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2, name="beta2_power", colocate_with=first_var)
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            if self._amsgrad:
                self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)
        weight_decay = self._call_if_callable(self._weight_decay)
        total_steps = self._call_if_callable(self._total_steps)
        warmup_proportion = self._call_if_callable(self._warmup_proportion)
        min_lr = self._call_if_callable(self._min_lr)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        self._weight_decay_t = ops.convert_to_tensor(weight_decay, name="weight_decay")
        self._total_steps_t = ops.convert_to_tensor(total_steps, name="total_steps")
        self._warmup_proportion_t = ops.convert_to_tensor(warmup_proportion, name="warmup_proportion")
        self._min_lr_t = ops.convert_to_tensor(min_lr, name="min_lr")

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        tvars = list(zip(*grads_and_vars))[1]
        self._create_slots_internal(tvars)

        # if self.clip_gradients:
        #     _, _, beta2_power = self._get_beta_accumulators()
        #     norm_list =[self.get_slot(var, "v") for var in tvars]
        #     clip_norm = tf.add_n([tf.reduce_sum(v) for v in norm_list])
        #     clip_norm = clip_norm / (1.0 - beta2_power)
        #     clip_norm = tf.sqrt(clip_norm) * self.clip_multiplier_t + self.clip_epsilon_t
        #     grads = list(zip(*grads_and_vars))[0]
        #     grads, cur_norm = tf.clip_by_global_norm(grads, clip_norm)
        #     grads_and_vars = list(zip(grads, tvars))

        return super().apply_gradients(grads_and_vars, global_step, name)

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        step, beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

        if self._initial_total_steps > 0:
            total_steps = math_ops.cast(self._total_steps_t, var.dtype.base_dtype)
            warmup_proportion = math_ops.cast(self._warmup_proportion_t, var.dtype.base_dtype)
            min_lr = math_ops.cast(self._min_lr_t, var.dtype.base_dtype)
            warmup_steps = total_steps * warmup_proportion
            decay_steps = math_ops.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                step <= warmup_steps,
                lr_t * (step / warmup_steps),
                lr_t + decay_rate * math_ops.minimum(step - warmup_steps, decay_steps),
            )

        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        v = self.get_slot(var, "v")

        # if var in self.reg_vars:
        #     var_scaled = var*(math_ops.sqrt(v / (1.0 - beta2_power)) + epsilon_t)
        #     proj = var_scaled*(tf.reduce_sum(var_scaled*grad)/(tf.reduce_sum(tf.square(var_scaled)))+1e-16)
        #     grad = grad-proj

        if self.clip_gradients:
            clipVal = math_ops.sqrt(tf.reduce_sum(v) / (1.0 - beta2_power)) * self.clip_multiplier_t + self.clip_epsilon_t
            grad = clip_ops.clip_by_norm(grad, clipVal)

        sma_inf = 2.0 / (1.0 - beta2_t) - 1.0
        sma_t = sma_inf - 2.0 * step * beta2_power / (1.0 - beta2_power)

        m = self.get_slot(var, "m")

        v_t = state_ops.assign(v, beta2_t * v + (1.0 - beta2_t) * math_ops.square(grad), use_locking=self._use_locking)
        v_corr_t = math_ops.sqrt(v_t / (1.0 - beta2_power)) + epsilon_t
        grad_corr = grad/v_corr_t

        # if var in self.reg_vars:
        #     var_scaled = var#var*(math_ops.sqrt(v / (1.0 - beta2_power)) + epsilon_t)
        #     proj = var_scaled*(tf.reduce_sum(var_scaled*grad_corr)/(tf.reduce_sum(tf.square(var_scaled)))+1e-16)
        #     grad_corr = grad_corr - proj

        m_t = state_ops.assign(m, beta1_t * m + (1.0 - beta1_t) * grad_corr, use_locking=self._use_locking)
        m_corr_t = m_t / (1.0 - beta1_power)

        r_t = math_ops.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                            (sma_t - 2.0) / (sma_inf - 2.0) *
                            sma_inf / sma_t)

        var_t = tf.where(sma_t >= 5.0, r_t * m_corr_t , m_corr_t)
        # with tf.device('/cpu:0'):
        #     tf.summary.scalar("sma_t", tf.reduce_mean(sma_t))
        #     tf.summary.scalar("r_t", tf.reduce_mean(r_t))
        #     #tf.summary.scalar("sma_inf", tf.reduce_mean(sma_inf))
        # if var in self.reg_vars:
        #     proj = var*(tf.reduce_sum(var*var_t)/(tf.reduce_sum(tf.square(var)))+1e-16)
        #     var_t = var_t-proj
        #     print(var)

        if var in self.reg_vars:
            #val_norm = tf.reduce_mean(tf.abs(var))
            #var_scaled = var# * v_corr_t / tf.reduce_mean(v_corr_t)
            # with tf.device('/cpu:0'):
            #     tf.summary.histogram("variation", v_corr_t / tf.reduce_mean(v_corr_t))
            if self._initial_weight_decay > 0.0:
                var_t += math_ops.cast(self._weight_decay_t, var.dtype.base_dtype) * var
            if self._L1_decay > 0.0:
                var_t += math_ops.cast(self._L1_decay, var.dtype.base_dtype) * math_ops.sign(var)

        with tf.control_dependencies([var_t]):
            var_update = state_ops.assign_sub(var, lr_t * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        return control_flow_ops.group(*updates)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        step, beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

        if self._initial_total_steps > 0:
            total_steps = math_ops.cast(self._total_steps_t, var.dtype.base_dtype)
            warmup_proportion = math_ops.cast(self._warmup_proportion_t, var.dtype.base_dtype)
            min_lr = math_ops.cast(self._min_lr_t, var.dtype.base_dtype)
            warmup_steps = total_steps * warmup_proportion
            decay_steps = math_ops.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                step <= warmup_steps,
                lr_t * (step / warmup_steps),
                lr_t + decay_rate * math_ops.minimum(step - warmup_steps, decay_steps),
            )

        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        v = self.get_slot(var, "v")

        if self.clip_gradients:
            clipVal = math_ops.sqrt(tf.reduce_sum(v) / (1.0 - beta2_power)) * self.clip_multiplier_t + self.clip_epsilon_t
            grad = clip_ops.clip_by_norm(grad, clipVal)

        sma_inf = 2.0 / (1.0 - beta2_t) - 1.0
        sma_t = sma_inf - 2.0 * step * beta2_power / (1.0 - beta2_power)

        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        m_corr_t = m_t / (1.0 - beta1_power)

        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        if self._amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat, math_ops.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = math_ops.sqrt(vhat_t / (1.0 - beta2_power)) + epsilon_t
        else:
            v_corr_t = math_ops.sqrt(v_t / (1.0 - beta2_power)) + epsilon_t

        r_t = math_ops.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                            (sma_t - 2.0) / (sma_inf - 2.0) *
                            sma_inf / sma_t)

        var_t = tf.where(sma_t >= 5.0, r_t * m_corr_t / v_corr_t, m_corr_t)

        if var in self.reg_vars:
            if self._initial_weight_decay > 0.0:
                var_t += math_ops.cast(self._weight_decay_t, var.dtype.base_dtype) * var
            if self._L1_decay > 0.0:
                var_t += math_ops.cast(self._L1_decay, var.dtype.base_dtype) * math_ops.sign(var)

        var_update = state_ops.assign_sub(var, lr_t * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self._amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies([resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            step, beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_step = step.assign(step + 1.0, use_locking=self._use_locking)
                update_beta1 = beta1_power.assign(beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_step, update_beta1, update_beta2], name=name_scope)
