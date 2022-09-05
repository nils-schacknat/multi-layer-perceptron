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

        # Compute loss and gradients
        gradient_dict = dict()
        for i, comp in enumerate(self.components):
            print(comp.__class__)
            # If the component has parameters to optimize
            if hasattr(comp, 'optimize') and comp.optimize:
                # Get gradient w.r.t. input and gradient w.r.t. the component parameters
                gradient, gradient_comp = comp.gradient(X)
                gradient_dict[i] = gradient_comp
            else:
                gradient = comp.gradient(X)

            # Compose the gradients
            for k in gradient_dict.keys():
                if i != k:
                    print(gradient_dict[k].shape, gradient.shape)
                    gradient_dict[k] *= gradient

            X = comp(X)

        gradient_loss = self.loss_func.gradient(X, Y)
        loss = self.loss_func(X, Y)

        # Finally, add gradient of the loss
        for k in gradient_dict.keys():
            gradient_dict[k] *= gradient_loss

        print(gradient_dict)

        return loss


if __name__ == '__main__':
    from mlp_components import *
    from sample_data import *

    # Get sample data
    X, Y = gaussian()

    # Define model architecture
    linear1 = LinearLayer(in_features=2, out_features=3)
    relu = ReLU()
    linear2 = LinearLayer(in_features=3, out_features=2)
    loss_func = SoftmaxCrossEntropyLoss()

    model = Model(
        linear1,
        relu,
        linear2,
        loss_func=loss_func
    )

    model.fit(X, Y)
