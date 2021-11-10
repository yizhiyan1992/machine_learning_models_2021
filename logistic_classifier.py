import numpy as np


class logistic_classifier:
    def __init__(
        self,
        max_iter:int = 1000,
        learning_rate: float = 0.1,
    ):

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w = np.empty((0, 0))
        self.loss_list = []

    def __str__(self):
        return f"the logistic classifier with max_iter:{self.max_iter}, " \
               f"learning_rate:{self.learning_rate}."

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.w)
        y_pred = 1 / (1 + np.exp(-z))
        assert y_pred.shape == (x.shape[0], 1)
        return y_pred

    def loss_function(self, y: np.ndarray, y_pre: np.ndarray) -> float:
        """
        logistic regression model uses cross entropy loss function.
        """
        val = np.sum(-(1/len(y))*(y*np.log(y_pre) + (1-y)*np.log(1-y_pre)))
        return val

    def gradient_function(self, x: np.ndarray, y: np.ndarray, y_pre: np.ndarray) -> np.ndarray:
        grad = np.dot(x.T, (1/len(y))*(y_pre-y))
        assert grad.shape == (x.shape[1], 1)
        return grad

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        given X (m,n) and y (m,), train the logistic regressor
        ---
        return the classifier
        """

        # reformat and test shapes of x and y
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        y = y.reshape((-1, 1))
        if x.shape[0] != y.shape[0]:
            raise ValueError("please check the size of training data.")

        # initialize the weight
        self.w = np.zeros((x.shape[1], 1))
        # gradient descent
        for i in range(self.max_iter):
            y_pred = self.sigmoid(x)
            loss_val = self.loss_function(y, y_pred)
            self.loss_list.append(loss_val)
            gradient = self.gradient_function(x, y, y_pred)
            self.w -= self.learning_rate*gradient
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
        prob = self.sigmoid(x)
        return prob

    def predict(self, x: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(x)
        return np.where(prob > 0.5, 1, 0)

    def get_accuracy_score(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        y = y.reshape((-1, 1))
        y_pred = y_pred.reshape((-1, 1))
        return np.sum(y == y_pred)/len(y)
