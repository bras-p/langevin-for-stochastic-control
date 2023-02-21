import tensorflow as tf
from .base import DatasetLoader

class DataLoaderFromMap(DatasetLoader):
    def __init__(
        self,
        N_train = 100,
        N_test = 1000,
        batch_size = 512,
        get_X0 = None,
    ):
        super().__init__(N_train, N_test, batch_size)
        self.get_X0 = get_X0
    
    def loadData(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (self.get_X0(batch_size=self.N_train), tf.zeros((self.N_train, 1)))
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        ds_test = tf.data.Dataset.from_tensor_slices(
            (self.get_X0(batch_size=self.N_test), tf.zeros((self.N_test, 1)))
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return ds, ds_test
