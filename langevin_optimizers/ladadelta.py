import tensorflow as tf
from .base import LangevinOptimizer


class LAdadelta(LangevinOptimizer):
    def __init__(self, learning_rate=0.01, sigma=0.001, beta_1=0.95, beta_2=0.95, diagonal_bias=1e-6, name="LAdadelta", **kwargs):
        super().__init__(learning_rate, sigma, name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.diagonal_bias = diagonal_bias

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "delta_rms")
            self.add_slot(var, "grad_rms")

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        coefficients, lr_t, sigma = self._get_coefficients(grad, var, apply_state)
        delta_rms_var = self.get_slot(var, "delta_rms")
        grad_rms_var = self.get_slot(var, "grad_rms")
        new_grad_rms = self.beta_2*grad_rms_var + (1-self.beta_2)*tf.square(grad)
        preconditioner = tf.sqrt(delta_rms_var + tf.cast(self.diagonal_bias, grad.dtype))*tf.math.rsqrt(new_grad_rms + tf.cast(self.diagonal_bias, grad.dtype))
        new_delta = - lr_t*preconditioner*grad + tf.random.normal(shape=tf.shape(grad),stddev=tf.math.sqrt(lr_t)*tf.sqrt(preconditioner)*sigma)
        new_var = var + new_delta
        new_delta_rms = self.beta_1*delta_rms_var + (1-self.beta_1)*tf.square(new_delta)
        delta_rms_var.assign(new_delta_rms)
        grad_rms_var.assign(new_grad_rms)
        var.assign(new_var)



class LayerLAdadelta(LangevinOptimizer):
    def __init__(self, learning_rate=0.01, sigma=0.001, beta_1=0.95, beta_2=0.95, diagonal_bias=1e-6, langevin_layers=[], name="LAdadelta", **kwargs):
        super().__init__(learning_rate, sigma, name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.diagonal_bias = diagonal_bias
        self.langevin_layers = langevin_layers

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "delta_rms")
            self.add_slot(var, "grad_rms")

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        coefficients, lr_t, sigma = self._get_coefficients(grad, var, apply_state)
        delta_rms_var = self.get_slot(var, "delta_rms")
        grad_rms_var = self.get_slot(var, "grad_rms")
        new_grad_rms = self.beta_2*grad_rms_var + (1-self.beta_2)*tf.square(grad)
        preconditioner = tf.sqrt(delta_rms_var + tf.cast(self.diagonal_bias, grad.dtype))*tf.math.rsqrt(new_grad_rms + tf.cast(self.diagonal_bias, grad.dtype))
        new_delta = - lr_t*preconditioner*grad + tf.random.normal(shape=tf.shape(grad),stddev=tf.math.sqrt(lr_t)*tf.sqrt(preconditioner)*sigma)*var._langevin
        new_var = var + new_delta
        new_delta_rms = self.beta_1*delta_rms_var + (1-self.beta_1)*tf.square(new_delta)
        delta_rms_var.assign(new_delta_rms)
        grad_rms_var.assign(new_grad_rms)
        var.assign(new_var)