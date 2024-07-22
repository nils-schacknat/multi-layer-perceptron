import numpy as np
from tqdm import tqdm


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

    def fit(self, X, Y, num_epochs, batch_size, lr, get_predictions=None):
        num_batches = int(np.ceil(X.shape[-1] / batch_size))
        loss_list = []
        acc_list = []
        prediction_list = []

        with tqdm(total=num_epochs,
                  ascii=' 123456789#',
                  desc=f'#Epoch: ',
                  postfix=f'Acc = 0'
                  ) as progress:
            for e in range(num_epochs):
                num_correct = 0
                loss = 0

                for i in range(num_batches):
                    start = i*batch_size
                    stop = min(X.shape[-1], (i+1)*batch_size)

                    loss_, acc_ = self.fit_batch(X[:, start:stop], Y[:, start:stop], lr=lr)
                    num_correct += (stop-start) * acc_
                    loss += (stop-start) * loss_

                if get_predictions is not None:
                    prediction_list.append(get_predictions(self))

                acc = num_correct / X.shape[-1]
                loss /= X.shape[-1]
                acc_list.append(acc)
                loss_list.append(loss)

                progress.set_postfix_str(f'Acc = {acc:.3f}')
                progress.update()

        if get_predictions is not None:
            return loss_list, acc_list, prediction_list
        else:
            return loss_list, acc_list

    def fit_batch(self, X, Y, lr):
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
                delta = -comp.downstream_gradient * lr
                comp.update_params(delta)

            comp.empty_gradient()

        self.loss_func.empty_gradient()

        return loss, acc


if __name__ == '__main__':
    from network_components import *
    from sample_data import *

    # Get sample data
    X, Y = gaussian()
    # Transform Y into probabilities
    Y_ = np.zeros((2, Y.shape[0]))
    Y_[1, Y == 1] = 1
    Y_[0, Y == 0] = 1

    linear1 = LinearLayer(in_features=2, out_features=2)
    smax_ce_loss = SoftmaxCrossEntropyLoss()
    model = Model(
        linear1,
        loss_func=smax_ce_loss
    )

    acc_list, loss_list = model.fit(X, Y_, 10, 32, .01)
