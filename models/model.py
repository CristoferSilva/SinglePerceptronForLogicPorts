from abc import ABC, abstractmethod

class model(ABC):

    @abstractmethod
    def fit(self, data, W, bias):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def calculate_cost(self, X, Y):
        pass