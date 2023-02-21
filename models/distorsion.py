import tensorflow as tf


class KMeans():

    def __init__(
        self,
        K = 2,
        p = 2,
    ):
        self.K = K
        self.p = p
    
    def getModel(self):
        model = tf.keras.Sequential([
            Distorsion(K=self.K, p=self.p, keepdims=True)
        ])
        return model



class Distorsion(tf.keras.layers.Layer):

    def __init__(
        self,
        K = 2,
        p = 2,
        keepdims = False
    ):
        super().__init__()
        self.K = K
        self.p = p
        self.keepdims = keepdims

    def build(self, input_shape):
        dim = input_shape[-1]
        self.centroids = self.add_weight("centroids", shape=[self.K, dim])
        self.built = True

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=1)
        distances = tf.reduce_sum((tf.math.abs(x - self.centroids))**self.p, axis=-1)
        distorsion = tf.reduce_min(distances, axis=1, keepdims=self.keepdims)
        return distorsion

    


