from abc import ABC, abstractmethod


class ModelBuilder(ABC):
    def __init__(
        self,
        T = 1.,
        N_euler = 10,
        dim = 1
    ):
        self.T = T
        self.N_euler = N_euler
        self.h = T/N_euler
        self.dim = dim
        

    @abstractmethod
    def getCtrlModel(self):
        pass

    @abstractmethod
    def getModel(self):
        pass


