import numpy as np


class NeuralNetworkClassifier:
    def __init__(
        self,
        *,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        num_layer: int = 3,
        num_neurons: int = 20,
        l2_regularization: float = 0
    ):

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.num_layer = num_layer
        self.num_neurons = num_neurons
        self.l2_regularization = l2_regularization

    def __str__(self):
        pass

    def forward_propagation(self):
        pass

    def backward_propagation(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_prob(self):
        pass

    def get_params(self):
        pass
