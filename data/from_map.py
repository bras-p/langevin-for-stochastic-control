import tensorflow as tf
from .base import DatasetLoader

class DataLoaderFromMap(DatasetLoader):
    def __init__(
        self,
        N_train = 100,
        N_test = 1000,
        batch_size = 512,
        get_X0 = None
    ):
        super().__init__(N_train, N_test, batch_size)
        self.get_X0 = get_X0
    
    def loadData(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((self.N_train, self.dim)), tf.zeros((self.N_train, 1)) )
        ).batch(self.batch_size).map(lambda x,y: (self.get_X0(x), y)).prefetch(tf.data.AUTOTUNE)

        ds_test = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((self.N_test, self.dim)), tf.zeros((self.N_test, 1)) )
        ).batch(self.batch_size).map(lambda x,y: (self.get_X0(x), y)).prefetch(tf.data.AUTOTUNE)

        return ds, ds_test
