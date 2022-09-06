
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

        # Compute loss and local gradients
        print(X.shape)
        for comp in self.components:
            print(comp.__class__)
            X = comp(X, track_gradient=True)
            print(X.shape)

        loss = self.loss_func(X, Y, track_gradient=True)
        print('**********************')

        for comp in self.components:
            print(comp.__class__)
            if type(comp.gradient) == tuple:
                print([s.shape for s in comp.gradient])
            else:
                print(comp.gradient.shape)

        print(self.loss_func.__class__)
        print(self.loss_func.gradient.shape)

        return loss


if __name__ == '__main__':
    from network_components import *
    from sample_data import *

    # Get sample data
    X, Y = gaussian()
    # Transform Y into probabilities
    Y_ = np.zeros((2, Y.shape[0]))
    Y_[0, Y] = 1
    Y_[1, 1-Y] = 0

    # Define model architecture
    linear1 = LinearLayer(in_features=2, out_features=4)
    relu = ReLU()
    linear2 = LinearLayer(in_features=4, out_features=2)
    smax_ce_loss = SoftmaxCrossEntropyLoss()

    model = Model(
        linear1,
        relu,
        linear2,
        loss_func=smax_ce_loss
    )

    model.fit(X[:, :10].reshape(2, 10), Y_[:, :10].reshape(2, 10))
