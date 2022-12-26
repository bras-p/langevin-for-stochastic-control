from abc import ABC, abstractmethod

class DatasetLoader(ABC):
    def __init__(
        self,
        N_train = 100,
        N_test = 1000,
        batch_size = 512,
    ):
        self.N_train = N_train
        self.N_test = N_test
        self.batch_size = batch_size

    @abstractmethod
    def loadData(self):
        pass