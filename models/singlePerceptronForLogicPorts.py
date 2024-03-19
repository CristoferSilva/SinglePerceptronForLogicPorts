import numpy as np
from models.model import model
from utils.activationFunctions import heaviside_step_function

class SinglePerceptronForLogicPorts(model):
    def __init__(self, activation_function = heaviside_step_function) -> None:
        self.W = []
        self.interactions = None
        self.activation_function = activation_function

    def predict(self, X):
        dot_product = X @ self.W.T
        y_hat = self.activation_function(dot_product)
        return y_hat

    def fit(self, data, W=[], bias = 1.0, learning_rate = 1.0, interactions = np.inf):
        X, Y = self.__get_X_Y(data, bias)
        N = len(Y) # N X D

        self.interactions = interactions
        current_interaction = 0

        if len(W) == 0:
            self.W = np.zeros(N)
        else:
            self.W = W
            
        while True:
            current_interaction += 1

            for i in range(N):
                self.__update_weights(self.W, X[i], Y[i], learning_rate)

            if self.__stop_criteria(X, Y, current_interaction):
                break

        print('The training ended | Final Weights= ' + str(self.W))

    def __update_weights(self, W, x, y, learning_rate):
        y_hat = self.predict(x)

        if y != y_hat:
            print('old_weight = ' + str(W))

            if y < y_hat:
                W = learning_rate * (W - x)
            elif y > y_hat:
                W = learning_rate * (W + x)

            self.W = W
            print('new_weight = ' + str(W))

    def calculate_cost(self, X, Y):
        number_of_predicitons = len(Y)
        incorrect_predicitons = 0

        for i in range(number_of_predicitons):
            y_hat = self.predict(X[i])
            y = Y[i]
            if y != y_hat:
                incorrect_predicitons += 1

        return incorrect_predicitons/number_of_predicitons

    def __stop_criteria(self, X, Y, current_interaction):
        return self.calculate_cost(X,Y) == 0 or self.interactions == current_interaction
    
    def __get_X_Y(self, data_matrix, bias):
        X = []
        Y = []
        for line in data_matrix:
            X.append(np.append(bias, line[:2]))
            Y.append(line[2])
        return np.array(X), np.array(Y)


