import numpy as np
from sklearn.neighbors import KernelDensity


def kde_sampling(base_points):
    kde = KernelDensity(kernel="gaussian", bandwidth=0.15).fit(base_points)
    return np.concatenate([kde.sample(n_samples=len(base_points))])


def gaussian_sampling(base_points):
    points = np.asarray(base_points)
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)

    samples = np.random.multivariate_normal(mean, cov, size=len(base_points))
    return np.concatenate([samples])
