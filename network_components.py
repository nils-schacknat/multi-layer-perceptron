from abc import ABC, abstractmethod
import numpy as np
rng = np.random.default_rng()


# **** Abstract classes - start ***************************
class NetworkComponent(ABC):
    def __init__(self):
        self.gradient = None

    def empty_gradient(self):
        self.gradient = None

    @abstractmethod
    def __call__(self, input, track_gradient=False):
        pass

    @abstractmethod
    def compose_gradients(self, upstream_gradient):
        pass


class Loss(ABC):
    def __init__(self):
        self.gradient = None

    def empty_gradient(self):
        self.gradient = None

    @abstractmethod
    def __call__(self, input, target, track_gradient=False):
        pass

# **** Abstract classes - end *****************************


class LinearLayer(NetworkComponent):
    """
    Implements a linear layer (biases are absorbed into the weight matrix), including the gradient function.
    """
    def __init__(self, in_features, out_features):
        # Initialize the weight matrix using "Xavier initialization"
        super().__init__()
        self.W = rng.normal(loc=0, scale=1/out_features**.5, size=(out_features, in_features+1))
        # Initialize biases with 0
        # self.W[:, -1] = 0

        # Downstream gradient w.r.t. the weights
        self.downstream_gradient = None

    def __call__(self, X, track_gradient=False):
        X = np.vstack((X, np.ones(X.shape[1])))
        Z = np.dot(self.W, X)

        if track_gradient:
            # (gradient w.r.t. the weights, gradient w.r.t. the input)
            self.gradient = (X, self.W[:, :-1])
        return Z

    def compose_gradients(self, upstream_gradient):
        # Downstream gradient w.r.t. the weights
        gradient_weights = upstream_gradient @ self.gradient[0].T
        # Downstream gradient w.r.t. the input
        gradient_input = self.gradient[1].T @ upstream_gradient

        self.downstream_gradient = gradient_weights
        return gradient_input

    def empty_gradient(self):
        self.gradient = None
        self.downstream_gradient = None

    def update_params(self, delta_W):
        self.W += delta_W


class ReLU(NetworkComponent):
    """
    Implements the ReLU activation function, including the gradient function.
    """
    def __call__(self, Z, track_gradient=False):
        if track_gradient:
            self.gradient = np.where(Z < 0, 0, 1)

        return np.maximum(Z, 0)

    def compose_gradients(self, upstream_gradient):
        # Downstream gradient w.r.t. the input
        return upstream_gradient * self.gradient


class SoftmaxCrossEntropyLoss(Loss):
    """
    Implements softmax combined with the cross entropy loss, including the gradient function.
    """
    def __call__(self, Z, Y, track_gradient=False):
        P = self.softmax(Z)
        ce = self.cross_entropy(Y, P)
        loss = np.mean(ce)

        if track_gradient:
            self.gradient = (P - Y) / Y.shape[1]

        return loss

    @staticmethod
    def softmax(Z):
        Z_exp = np.exp(Z)
        return Z_exp / np.sum(Z_exp, axis=0)

    @staticmethod
    # Hide errors for log(0) = -inf and 0 * -inf = nan
    @np.errstate(divide='ignore', invalid='ignore')
    def cross_entropy(Y, P):
        ce = -np.sum(Y * np.log(P), axis=0)
        return np.nan_to_num(ce, nan=0.0, posinf=np.inf, neginf=-np.inf)


if __name__ == '__main__':
    L = LinearLayer(in_features=3, out_features=4)
    X = rng.random((3, 5))
    Z = L(X)
    # print(L(X))

    loss = SoftmaxCrossEntropyLoss()
    P = loss.softmax(Z)
    # print(P, np.sum(P, axis=0))
    Y = np.zeros(P.shape)
    Y[0, :] = 1
    # print(loss.cross_entropy(Y, P))
