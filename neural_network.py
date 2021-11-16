import numpy as np


class NeuralNetworkClassifier:
    """ """
    def __init__(
        self,
        *,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        num_layer: int = 3,
        num_neurons: int = 20,
        l2_regularization: float = 0
    ):

        # type check
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(f"The max_iter should be positive integer, got {max_iter}")
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError(f"The learning_rate should be positive float number, got {learning_rate}")
        if not isinstance(num_layer, int) or num_layer <= 0:
            raise ValueError(f"The num_layer should be positive integer, got {num_layer}")
        if not isinstance(num_neurons, int) or num_neurons <= 0:
            raise ValueError(f"The num_neurons should be positive integer, got {num_neurons}")
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError(f"The learning_rate should not be negative value, got {l2_regularization}")

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.num_layer = num_layer
        self.num_neurons = num_neurons
        self.l2_regularization = l2_regularization

    def __str__(self):
        pass

    def forward_propagation(self):
        """ """
        pass

    def backward_propagation(self):
        """ """
        pass

    def fit(self):
        """ """
        pass

    def predict(self):
        """ """
        pass

    def predict_prob(self):
        """ """
        pass

    def get_params(self):
        """ """
        pass
