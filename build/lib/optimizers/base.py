import tensorflow as tf


class LangevinOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate, sigma, name='LangevinOptimizer', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("sigma", kwargs.get("sigma", sigma))

    def _decayed_sigma(self, var_dtype):
        sigma = self._get_hyper("sigma", var_dtype)
        if isinstance(sigma, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            sigma = tf.cast(sigma(local_step), var_dtype)
        return sigma
    
    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        if "sigma" in self._hyper:
            sigma = tf.identity(self._decayed_sigma(var_dtype))
            apply_state[(var_device, var_dtype)]["sigma"] = sigma
    
    def _get_coefficients(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        lr_t, sigma = coefficients["lr_t"], coefficients["sigma"]
        return coefficients, lr_t, sigma




def set_langevin(model):
    if hasattr(model.optimizer, 'langevin_layers'):
        langevin_layers = model.optimizer.langevin_layers

        for k in range(len(model.layers)):
            
            for weight in model.layers[k]._trainable_weights:
                if k in langevin_layers:
                    weight._langevin = 1.
                else:
                    weight._langevin = 0.

            for trackable in model.layers[k]._self_tracked_trackables: # careful, need this version for sub-models
                for weight in trackable._trainable_weights:
                    if k in langevin_layers:
                        weight._langevin = 1.
                    else:
                        weight._langevin = 0.
