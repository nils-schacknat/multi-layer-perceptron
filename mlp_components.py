import numpy as np
rng = np.random.default_rng()


class LinearLayer:
    """
    Implements a linear layer (biases are absorbed into the weight matrix), including the gradient function.
    """
    def __init__(self, in_features, out_features):
        self.optimize = True
        # Initialize the weight matrix using "Xavier initialization"
        self.Weights = rng.normal(loc=0, scale=1/out_features**.5, size=(out_features, in_features+1))
        # Initialize biases with 0
        self.Weights[:, -1] = 0

    def __call__(self, X):
        return np.dot(
            self.Weights,
            np.vstack((X, np.ones(X.shape[1])))
        )

    def gradient(self, X):
        return [self.gradient_input(X), self.gradient_params(X)]

    # ∂W*X / ∂W, gradient with respect to parameters
    def gradient_params(self, X):
        return np.vstack((X, np.ones(X.shape[1])))

    # ∂W*X / ∂X, gradient with respect to input
    def gradient_input(self, X):
        return self.Weights[:, :-1]


class ReLU:
    """
    Implements the ReLU activation function, including the gradient function.
    """
    def __call__(self, Z):
        return np.where(Z < 0, 0, Z)

    # ∂ReLU / ∂Z for each sample
    def gradient(self, Z):
        return np.where(Z < 0, 0, 1)


class SoftmaxCrossEntropyLoss:
    """
    Implements softmax combined with the cross entropy loss, including the gradient function.
    """
    def __call__(self, Z, Y):
        """

        :param Z: Input Matrix
        :param Y: Matrix with true probabilities, same shape as Z
        :return: Loss
        """
        P = self.softmax(Z)
        ce = self.cross_entropy(Y, P)
        loss = np.mean(ce)
        return loss

    # ∂Loss / ∂Z for each sample
    def gradient(self, Z, Y):
        return self.softmax(Z) - Y

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
