import tensorflow as tf
from .base import LangevinOptimizer


class LSGD(LangevinOptimizer):
    def __init__(self, learning_rate=0.001, sigma=0.001, name="SGOptimizer", **kwargs):
        super().__init__(learning_rate, sigma, name, **kwargs)

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        coefficients, lr_t, sigma = self._get_coefficients(grad, var, apply_state)
        local_step = tf.cast(self.iterations + 1, var.dtype.base_dtype)
        new_var = var - lr_t*grad + sigma*tf.math.sqrt(lr_t)*tf.random.normal(shape=tf.shape(grad))
        var.assign(new_var)


class LayerLSGD(LangevinOptimizer):
    def __init__(self, learning_rate=0.001, sigma=0.001, langevin_layers=[], name="SGOptimizer", **kwargs):
        super().__init__(learning_rate, sigma, name, **kwargs)
        self.langevin_layers = langevin_layers

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        coefficients, lr_t, sigma = self._get_coefficients(grad, var, apply_state)
        local_step = tf.cast(self.iterations + 1, var.dtype.base_dtype)
        new_var = var - lr_t*grad + sigma*tf.math.sqrt(lr_t)*tf.random.normal(shape=tf.shape(grad))*var._langevin
        var.assign(new_var)