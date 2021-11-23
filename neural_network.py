import numpy as np
import matplotlib.pyplot as plt

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

    def forward_propagation(self, x: np.ndarray, y: np.ndarray):
        """ perform forward-propagation from the first layer to the last for one time

        Parameters
        ----------
        x: array-like
            input features X
        y: array-like
            input labels y
        Returns
        ----------
        loss: float
            the loss value of all samples
        intermediate_values: dict
            the temporary dict used to save a and z values in different layers, which will be used for backward-prop
        """
        intermediate_values = dict()
        cur_a = x
        intermediate_values['a_layer_-1'] = x
        for layer in range(self.num_layer):
            cur_w = self.w['layer_' + str(layer)]
            cur_b = self.b['layer_' + str(layer)]
            next_z = self.linear_transform(cur_a, cur_w, cur_b)
            if layer != self.num_layer -1:
                next_a = self.activation_function(next_z)
            else:
                next_a = 1/(1+np.exp(-next_z))
            intermediate_values['z_layer_' + str(layer)] = next_z
            intermediate_values['a_layer_' + str(layer)] = next_a
            cur_a = next_a
        loss = self.loss_function(y, cur_a)
        return loss, intermediate_values

    @staticmethod
    def linear_transform(a: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ perform linear transformation w.T*x+b

        Parameters
        ----------
        a: array-like
            output from last layer
        w: array-like
            weights between current layer and next layer
        b: array-like
            biases between current layer and next layer

        Returns
        ----------
            z: array-like
        """
        z = np.dot(w, a) + b
        return z

    @staticmethod
    def activation_function(z: np.ndarray, activate_func: str = 'sigmoid') -> np.ndarray:
        """ perform lineaer transform and activation function for each layer when forward propagating

        Parameters
        ----------
        z: array-like
            input value for activation
        activate_func: {'sigmoid','relu'} default = sigmoid
            the type of activation function

        Return
        ----------
            array-like
        """
        if activate_func == 'sigmoid':
            #return 1 / (1 + np.exp(-z))
            return np.tanh(z)
        elif activate_func == 'relu':
            pass

    @staticmethod
    def loss_function(y: np.ndarray, y_pre: np.ndarray) -> float:
        """ use least square error function to calculate loss values

        Parameters
        ----------
        y: array-like
            label of samples (shape=(1,m))
        y_pre: array-like
            predicted values of samples (shape=(1,m))

        Returns
        ----------
        loss: float
        """
        m = y.shape[1]
        cost_sum = np.multiply(np.log(y_pre), y) + np.multiply((1 - y), np.log(1 - y_pre))
        cost = - np.sum(cost_sum) / m
        cost = float(np.squeeze(cost))
        assert isinstance(cost, float)
        return cost

    @staticmethod
    def backward_propagation(y: np.ndarray, intermediate_values: dict, w_dict: dict) -> dict:
        """ Calculate the gradients of w and b for each layer
        """
        derivative_dict = dict()
        num_layer = len(intermediate_values) // 2
        #a = intermediate_values['a_layer_' + str(num_layer - 1)]
        #d_a = np.mean(a - y)

        a = intermediate_values['a_layer_1']
        # 1 layer
        #z = intermediate_values['z_layer_0']
        #d_z = d_a * (1 - z) * z
        d_z = a - y
        a_pre = intermediate_values['a_layer_0']
        d_w = np.dot(d_z, a_pre.T) * (1/y.shape[1])
        derivative_dict['w_layer_1'] = d_w
        d_b = np.sum(d_z, axis=1, keepdims=True) * (1/y.shape[1])
        derivative_dict['b_layer_1'] = d_b
        z = intermediate_values['z_layer_0']
        d_z = np.dot(w_dict['layer_1'].T, d_z)*(1-z**2)
        a_pre = intermediate_values['a_layer_-1']
        d_w = np.dot(d_z, a_pre.T) * (1/y.shape[1])
        derivative_dict['w_layer_0'] = d_w
        d_b = np.sum(d_z, axis=1, keepdims=True)* (1/y.shape[1])
        derivative_dict['b_layer_0'] = d_b

        """
        for i in reversed(range(num_layer)):
            z = intermediate_values['z_layer_' + str(i)]
            d_z = d_a * (1 - z) * z
            a_pre = intermediate_values['a_layer_' + str(i - 1)]
            d_w = np.dot(d_z, a_pre.T)
            derivative_dict['w_layer_' + str(i)] = d_w
            d_b = np.mean(d_z, axis=1, keepdims=True)
            derivative_dict['b_layer_' + str(i)] = d_b
            w = w_dict['layer_' + str(i)]
            d_a = np.dot(w.T, d_z)
        """
       # print(derivative_dict)
        return derivative_dict

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

        # generate parameters W and b w.shape = [neuron_next, neuron_cur], b.shape = [neuron_next, 1]
        for layer in range(self.num_layer):
            if layer == 0:
                self.w['layer_' + str(layer)] = np.random.randn(self.num_neurons, n_features)*0.01
                self.b['layer_' + str(layer)] = np.zeros((self.num_neurons, 1))
            elif layer == self.num_layer - 1:
                self.w['layer_' + str(layer)] = np.random.randn(1, self.num_neurons, )*0.01
                self.b['layer_' + str(layer)] = np.zeros((1, 1))
            else:
                self.w['layer_' + str(layer)] = np.random.randn(self.num_neurons, self.num_neurons)*0.01
                self.b['layer_' + str(layer)] = np.zeros((self.num_neurons, 1))
        for i in self.w.values():
            print(i.shape,'w shape')
        loss_list = []
        for ite in range(2000):
            loss, inter_val = self.forward_propagation(x, y)
            loss_list.append(loss)
            derivative_dict = self.backward_propagation(y, inter_val, self.w)
            print('loss',loss)
            # update gradient
            for layer in range(self.num_layer):
                self.w['layer_' + str(layer)] -= self.learning_rate * derivative_dict['w_layer_' + str(layer)]
                self.b['layer_' + str(layer)] -= self.learning_rate * derivative_dict['b_layer_' + str(layer)]

        plt.plot(loss_list)
        plt.show()
    def predict(self):
        """ """
        pass

    def predict_prob(self):
        """ """
        pass

    def get_params(self):
        """ """
        pass
