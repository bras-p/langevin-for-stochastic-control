import tensorflow as tf
from .base import DatasetLoader


class ConstantLoader(DatasetLoader):
    def __init__(
        self,
        X0,
        N_train = 100,
        N_test = 1000,
        batch_size = 512,
    ):
        super().__init__(N_train, N_test, batch_size)
        self.X0 = X0

    def loadData(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (tf.tile(tf.reshape(self.X0, (1, self.X0.shape[-1])), tf.constant([self.N_train, 1], dtype=tf.int32)), tf.zeros((self.N_train, 1)) )
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        ds_test = tf.data.Dataset.from_tensor_slices(
            (tf.tile(tf.reshape(self.X0, (1, self.X0.shape[-1])), tf.constant([self.N_test, 1], dtype=tf.int32)), tf.zeros((self.N_test, 1)) )
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return ds, ds_test


class ConstantMultipleLoader(DatasetLoader):
    def __init__(
        self,
        X0_list,
        N_train=100,
        N_test=1000,
        batch_size=512,
    ):
        super().__init__(N_train, N_test, batch_size)
        self.X0_list = X0_list
    
    def loadData(self):
        xs = [tf.data.Dataset.from_tensor_slices(tf.tile(tf.reshape(self.X0_list[k], (1, self.X0_list[k].shape[-1])), tf.constant([self.N_train, 1], dtype=tf.int32)))
            for k in range(len(self.X0_list))]
        y = tf.data.Dataset.from_tensor_slices(tf.zeros((self.N_train, 1)))
        ds = tf.data.Dataset.zip((tuple(xs),y)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        xs = [tf.data.Dataset.from_tensor_slices(tf.tile(tf.reshape(self.X0_list[k], (1, self.X0_list[k].shape[-1])), tf.constant([self.N_test, 1], dtype=tf.int32)))
            for k in range(len(self.X0_list))]
        y = tf.data.Dataset.from_tensor_slices(tf.zeros((self.N_test, 1)))
        ds_test = tf.data.Dataset.zip((tuple(xs),y)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return ds, ds_test
