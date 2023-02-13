import tensorflow as tf
from .base import LangevinOptimizer


class LAdam(LangevinOptimizer):
    def __init__(self, learning_rate=0.001, sigma=0.001, beta_1=0.9, beta_2=0.999, diagonal_bias=1e-6, name="LAdam", **kwargs):
        super().__init__(learning_rate, sigma, name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.diagonal_bias = diagonal_bias

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var,"m")
            self.add_slot(var,"v")

    @tf.function
    # @tf.autograph.experimental.do_not_convert
    def _resource_apply_dense(self, grad, var, apply_state=None):
        coefficients, lr_t, sigma = self._get_coefficients(grad, var, apply_state)
        local_step = tf.cast(self.iterations + 1, var.dtype.base_dtype)
        m_var = self.get_slot(var, "m")
        v_var = self.get_slot(var, "v")
        new_m = self.beta_1*m_var + (1.-self.beta_1)*grad
        new_v = self.beta_2*v_var + (1.-self.beta_2)*tf.square(grad)
        v_hat = new_v/(1.-tf.pow(self.beta_2, local_step))
        preconditioner =  1./ ((1.-tf.pow(self.beta_1, local_step)) * (self.diagonal_bias + tf.math.sqrt(v_hat)))
        stddev = sigma*tf.math.sqrt(lr_t*preconditioner)
        new_var = var - lr_t*preconditioner*new_m + tf.random.normal(shape=tf.shape(grad),stddev=stddev)
        m_var.assign(new_m)
        v_var.assign(new_v)
        var.assign(new_var)



class LayerLAdam(LangevinOptimizer):
    def __init__(self, learning_rate=0.001, sigma=0.001, beta_1=0.9, beta_2=0.999, diagonal_bias=1e-6, langevin_layers=[], name="LAdam", **kwargs):
        super().__init__(learning_rate, sigma, name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.diagonal_bias = diagonal_bias
        self.langevin_layers = langevin_layers

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var,"m")
            self.add_slot(var,"v")

    @tf.function
    # @tf.autograph.experimental.do_not_convert
    def _resource_apply_dense(self, grad, var, apply_state=None):
        coefficients, lr_t, sigma = self._get_coefficients(grad, var, apply_state)
        local_step = tf.cast(self.iterations + 1, var.dtype.base_dtype)
        m_var = self.get_slot(var, "m")
        v_var = self.get_slot(var, "v")
        new_m = self.beta_1*m_var + (1.-self.beta_1)*grad
        new_v = self.beta_2*v_var + (1.-self.beta_2)*tf.square(grad)
        v_hat = new_v/(1.-tf.pow(self.beta_2, local_step))
        preconditioner =  1./ ((1.-tf.pow(self.beta_1, local_step)) * (self.diagonal_bias + tf.math.sqrt(v_hat)))
        stddev = sigma*tf.math.sqrt(lr_t*preconditioner)
        new_var = var - lr_t*preconditioner*new_m + tf.random.normal(shape=tf.shape(grad),stddev=stddev)*var._langevin
        m_var.assign(new_m)
        v_var.assign(new_v)
        var.assign(new_var)




