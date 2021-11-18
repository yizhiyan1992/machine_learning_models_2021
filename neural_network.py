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
        self.w = dict()
        self.b = dict()

    def __str__(self):
        pass

    def forward_propagation(self, x: np.ndarray):
        """ perform forward-propagation from the first layer to the last for one time

        Parameters
        ----------
        x: array-like
            input features X

        Returns
        ----------
        cur_z: array-like
            the output from last layer (should be 1xm for binary classification)
        """
        cur_z = x
        for layer in range(self.num_layer):
            cur_w = self.w['layer_' + str(layer)]
            cur_b = self.b['layer_' + str(layer)]
            cur_a = self.linear_transform(cur_z, cur_w, cur_b)
            next_z = self.activation_function(cur_a)
            cur_z = next_z
        return cur_z

    @staticmethod
    def linear_transform(z: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ perform linear transformation w.T*x+b

        Parameters
        ----------
        z: array-like
            output from last layer
        w: array-like
            weights between current layer and next layer
        b: array-like
            biases between current layer and next layer

        Returns
        ----------
            a: array-like
        """
        a = np.dot(w.T, z) + b
        return a

    @staticmethod
    def activation_function(a: np.ndarray, activate_func: str = 'sigmoid') -> np.ndarray:
        """ perform activation function for each layer when forward propagating

        Parameters
        ----------
        a: array-like
            input value for activation
        activate_func: {'sigmoid','relu'} default = sigmoid
            the type of activation function

        Return
        ----------
            array-like
        """
        if activate_func == 'sigmoid':
            return 1 / (1 + np.exp(-a))
        elif activate_func == 'relu':
            pass

    def backward_propagation(self):
        """ """
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        """ train the model with training data

        Parameters
        ----------
        x: array like
            Features of the training data, the shape should be (n,m), where m is the number of samples
            and n is the number of features.

        y: array like
            Labels of the training data, the shape should be (,m)

        Returns
        ----------
        estimator: type?
        """

        # check input data
        if not isinstance(x, np.ndarray):
            raise ValueError(f"The input feature X should be array-like, got {type(x)}")
        if not isinstance(y, np.ndarray):
            raise ValueError(f"The input label y should be array-like, got {type(y)}")
        y = y.reshape((1, -1))
        if x.shape[1] != y.shape[1]:
            raise ValueError(f"the sample size of feature X and label y should match. got {x.shape} and {y.shape}")
        n_features, n_samples = x.shape

        # generate parameters W and b
        for layer in range(self.num_layer):
            if layer == 0:
                self.w['layer_' + str(layer)] = np.random.normal(0, 1, (n_features, self.num_neurons))
                self.b['layer_' + str(layer)] = np.random.normal(0, 1, (self.num_neurons, 1))
            elif layer == self.num_layer - 1:
                self.w['layer_' + str(layer)] = np.random.normal(0, 1, (self.num_neurons, 1))
                self.b['layer_' + str(layer)] = np.random.normal(0, 1, (1, 1))
            else:
                self.w['layer_' + str(layer)] = np.random.normal(0, 1, (self.num_neurons, self.num_neurons))
                self.b['layer_' + str(layer)] = np.random.normal(0, 1, (self.num_neurons, 1))

        #for key,val in self.w.items():
        #    print(key,val.shape)
        #for key,val in self.b.items():
        #    print(key,val.shape)
        self.forward_propagation(x)
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
