import numpy as np
rng = np.random.default_rng()


def gaussian(sigma=.3, num_samples=128):
    """
    Create two separate noisy point clouds belonging to two classes.

    :param sigma: Standard deviation
    :param num_samples: Number of samples per class
    :return: Point pairs (Nx2 numpy array) and labels
    """
    # Class 0, x and y coordinates
    c0_x = rng.normal(-1, sigma, num_samples)
    c0_y = rng.normal(-1, sigma, num_samples)
    c0_points = np.vstack((c0_x, c0_y))

    # Class 1, x and y coordinates
    c1_x = rng.normal(1, sigma, num_samples)
    c1_y = rng.normal(1, sigma, num_samples)
    c1_points = np.vstack((c1_x, c1_y))

    # Concatenate classes
    points = np.hstack((c0_points, c1_points))
    labels = np.repeat([0, 1], points.shape[1]/2)

    # Shuffle sample points and labels
    shuffle_idx = np.arange(points.shape[1])
    rng.shuffle(shuffle_idx)

    return points[:, shuffle_idx], labels[shuffle_idx]


def x_or(sigma=.3, num_samples=250):
    """
    Create four separate noisy point clouds belonging to two classes.

    :param sigma: Standard deviation
    :param num_samples: Number of samples per class
    :return: Point pairs (Nx2 numpy array) and labels
    """
    # Class 0, x and y coordinates
    c0_x_0 = rng.normal(-1, sigma, num_samples//4)
    c0_y_0 = rng.normal(-1, sigma, num_samples//4)
    c0_x_1 = rng.normal(1, sigma, num_samples//4)
    c0_y_1 = rng.normal(1, sigma, num_samples//4)
    c0_points = np.vstack((
        np.hstack((c0_x_0, c0_x_1)),
        np.hstack((c0_y_0, c0_y_1))
    ))

    # Class 1, x and y coordinates
    c1_x_0 = rng.normal(-1, sigma, num_samples // 4)
    c1_y_0 = rng.normal(1, sigma, num_samples // 4)
    c1_x_1 = rng.normal(1, sigma, num_samples // 4)
    c1_y_1 = rng.normal(-1, sigma, num_samples // 4)
    c1_points = np.vstack((
        np.hstack((c1_x_0, c1_x_1)),
        np.hstack((c1_y_0, c1_y_1))
    ))

    # Concatenate classes
    points = np.hstack((c0_points, c1_points))
    labels = np.repeat([0, 1], points.shape[1]/2)

    # Shuffle sample points and labels
    shuffle_idx = np.arange(points.shape[1])
    rng.shuffle(shuffle_idx)

    return points[:, shuffle_idx], labels[shuffle_idx]


def circle(sigma=.3, num_samples=250):
    """
    Create two separate noisy point clouds (inner circle and outer circle surface) belonging to two classes.

    :param sigma: Standard deviation
    :param num_samples: Number of samples per class
    :return: Point pairs (Nx2 numpy array) and labels
    """
    # Class 0, inside the circle, generate angles + radii + noise
    c0_theta = rng.random(num_samples) * np.pi * 2
    c0_r = rng.random(num_samples)
    c0_x = np.cos(c0_theta) * c0_r + rng.random(num_samples) * sigma
    c0_y = np.sin(c0_theta) * c0_r + rng.random(num_samples) * sigma
    c0_points = np.vstack((c0_x, c0_y))

    # Class 1, on circle surface, generate angles + noise
    c1_theta = rng.random(num_samples) * np.pi * 2
    c1_x = 1.5 * np.cos(c1_theta) + rng.random(num_samples) * sigma
    c1_y = 1.5 * np.sin(c1_theta) + rng.random(num_samples) * sigma
    c1_points = np.vstack((c1_x, c1_y))

    # Concatenate classes
    points = np.hstack((c0_points, c1_points))
    labels = np.repeat([0, 1], points.shape[1] / 2)

    # Shuffle sample points and labels
    shuffle_idx = np.arange(points.shape[1])
    rng.shuffle(shuffle_idx)

    return points[:, shuffle_idx], labels[shuffle_idx]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    points, labels = gaussian()

    plt.scatter(points[0], points[1], c=labels)
    plt.gca().set_aspect('equal')
    plt.show()
