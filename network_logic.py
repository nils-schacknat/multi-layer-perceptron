import numpy as np


class Model:
    def __init__(self, *components, loss_func):
        self.components = components
        self.loss_func = loss_func

    def __call__(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, X.shape[0])

        for comp in self.components:
            X = comp(X)

        return X

    def fit(self, X, Y):
        if len(X.shape) == 1:
            X = X.reshape(1, X.shape[0])

        # Forward pass
        # Compute loss and local gradients
        for comp in self.components:
            X = comp(X, track_gradient=True)

        pred = np.argmax(X, axis=0)
        target = np.argmax(Y, axis=0)
        acc = np.sum(pred == target) / pred.shape[-1]

        loss = self.loss_func(X, Y, track_gradient=True)
        upstream_gradient = self.loss_func.gradient

        # Backward pass
        # Compute downstream gradients
        for comp in reversed(self.components):
            downstream_gradient = comp.compose_gradients(upstream_gradient)
            upstream_gradient = downstream_gradient

        # Optimize and empty gradients
        for comp in self.components:
            if hasattr(comp, 'downstream_gradient'):
                delta = -comp.downstream_gradient * 0.001
                comp.update_params(delta)

            comp.empty_gradient()

        self.loss_func.empty_gradient()

        return loss, acc


if __name__ == '__main__':
    from network_components import *
    from sample_data import *

    # # Get sample data
    # X, Y = gaussian()
    # # Transform Y into probabilities
    # Y_ = np.zeros((2, Y.shape[0]))
    # Y_[0, Y] = 1
    # Y_[1, 1-Y] = 0

    X = np.array([
        [-1, 1, 3, -2],
        [-2, 1, 2, -1]
    ])
    Y = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1]
    ])

    # # Define model architecture
    # linear1 = LinearLayer(in_features=2, out_features=3)
    # relu = ReLU()
    # linear2 = LinearLayer(in_features=3, out_features=2)
    # smax_ce_loss = SoftmaxCrossEntropyLoss()

    # linear1.W = np.array([
    #     [0.98, 0.063, 0.01],
    #     [0.81, 0.21, 0.02],
    #     [-0.44, 0.24, -0.013]
    # ])
    #
    # linear2.W = np.array([
    #     [0.14, -0.10, 1.39, -0.01],
    #     [-1.20, 0.64, -0.23,  0.023]
    # ])

    # model = Model(
    #     linear1,
    #     relu,
    #     linear2,
    #     loss_func=smax_ce_loss
    # )

    linear1 = LinearLayer(in_features=2, out_features=2)
    smax_ce_loss = SoftmaxCrossEntropyLoss()
    model = Model(
        linear1,
        loss_func=smax_ce_loss
    )

    loss = model.fit(X, Y)

