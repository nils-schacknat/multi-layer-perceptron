import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


cdict = {
    'red':   [[0.0, 0.0, 0.0],
              [0.5, 1.0, 1.0],
              [1.0, 1.0, 1.0]],
    'green': [[0.0, 0.0, 0.7],
              [0.45, 1.0, 1.0],
              [0.55, 1.0, 1.0],
              [1.0, 0.4, 0.0]],
    'blue':  [[0.0, 1.0, 1.0],
              [0.5, 1.0, 1.0],
              [1.0, 0.0, 0.0]]
}
cmap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

linewidth = .3
num_levels = 256
dpi = 160
style_dict = {
    'figure.dpi': dpi,
    'font.size': 8,
    'axes.linewidth': linewidth,
    'image.aspect': 'equal',
    'image.cmap': cmap
}
mpl.rcParams.update(style_dict)


def get_grid_predictions(x_min, x_max, nx, y_min, y_max, ny, model):
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((
        xx.flatten(),
        yy.flatten()
    ))

    preds = model(points)[0].reshape(ny, nx)

    return x, y, preds


def plot_grid_prediction(grid_prediction, X, Y, epoch, acc, dpi=dpi):
    x, y, preds = grid_prediction

    fig, ax = plt.subplots(dpi=dpi)

    cf = plt.contourf(x, y, preds, levels=np.linspace(0, 1, num_levels))
    cb = plt.colorbar(ticks=np.linspace(0, 1, 6))
    plt.scatter(X[0], X[1], c=Y, edgecolors='black', linewidths=.15, zorder=2)
    title = ax.text(
        .05, .95, f'Epoch: {epoch},  Accuracy: {acc:.3f}',
        fontfamily='monospace', transform=ax.transAxes, ha="left", va="top",
        bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 5, 'linewidth': linewidth}
    )

    # Change linewidths
    ax.tick_params(width=linewidth)
    cb.ax.tick_params(width=linewidth)

    return fig, ax, cf, title


def update_grid_prediction(grid_prediction, epoch, acc, ax, cf, title):
    x, y, preds = grid_prediction

    for coll in cf.collections:
        coll.remove()
    cf = ax.contourf(x, y, preds, levels=np.linspace(0, 1, num_levels))
    title.set_text(f'Epoch: {epoch},  Accuracy: {acc:.3f}')

    return cf

