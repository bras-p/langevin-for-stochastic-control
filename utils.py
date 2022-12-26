import tensorflow as tf

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred[:,0])

def sq_avg(y_true, y_pred):
    return tf.reduce_mean(y_pred[:,0]**2)

def initialize_to_zero(model):
    for layer in model.layers:
        zeros = [tf.zeros(tf.shape(weight)) for weight in layer.weights]
        layer.set_weights(zeros)

def initialize_to_normal(model, sigma):
    for layer in model.layers:
        normal_init = [tf.random.normal(tf.shape(weight), stddev=sigma) for weight in layer.weights]
        layer.set_weights(normal_init)
