import tensorflow as tf
from .base import LangevinOptimizer


class LRMSprop(LangevinOptimizer):
    def __init__(self, learning_rate=0.001, sigma=0.001, alpha=0.9, diagonal_bias=1e-6, name="LRMSprop", **kwargs):
        super().__init__(learning_rate, sigma, name, **kwargs)
        self.alpha = alpha
        self.diagonal_bias = diagonal_bias

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms")

    @tf.function
    # @tf.autograph.experimental.do_not_convert
    def _resource_apply_dense(self, grad, var, apply_state=None):
        coefficients, lr_t, sigma = self._get_coefficients(grad, var, apply_state)
        rms_var = self.get_slot(var, "rms")
        new_rms = self.alpha*rms_var + (1-self.alpha)*tf.square(grad)
        preconditioner = 1./(self.diagonal_bias + tf.math.sqrt(new_rms))
        stddev = sigma*tf.math.sqrt(lr_t*preconditioner)
        new_var = var - lr_t*preconditioner*grad + tf.random.normal(shape=tf.shape(grad), stddev=stddev)
        rms_var.assign(new_rms)
        var.assign(new_var)



class LayerLRMSprop(LangevinOptimizer):
    def __init__(self, learning_rate=0.001, sigma=0.001, alpha=0.9, diagonal_bias=1e-6, langevin_layers=[], name="LayerLRMSprop", **kwargs):
        super().__init__(learning_rate, sigma, name, **kwargs)
        self.alpha = alpha
        self.diagonal_bias = diagonal_bias
        self.langevin_layers = langevin_layers
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms")
            
    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        coefficients, lr_t, sigma = self._get_coefficients(grad, var, apply_state)
        rms_var = self.get_slot(var, "rms")
        new_rms = self.alpha*rms_var + (1-self.alpha)*tf.square(grad)
        preconditioner = 1./(self.diagonal_bias + tf.math.sqrt(new_rms))
        stddev = sigma*tf.math.sqrt(lr_t*preconditioner)
        new_var = var - lr_t*preconditioner*grad + tf.random.normal(shape=tf.shape(grad), stddev=stddev)*var._langevin
        rms_var.assign(new_rms)
        var.assign(new_var)

