import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eps = 1e-6

def empirical_distribution(data, plot=False, show=False):
    data = np.sort(data)
    n = len(data)
    if plot or show:
        plt.plot(data, np.arange(1, n + 1), 'o')
        plt.xlabel('x')
        plt.ylabel('P(X <= x)')
        if show: plt.show()
    return data, np.arange(1, n + 1) / n

def find_nth_quantile(data, n):
    data, cdf = empirical_distribution(data)
    greater = data[cdf > n][0]
    return greater

def sigmoid(x, limit, inv=False):
    if inv: return limit * np.log(x / (limit - x))
    else: return limit / (1 + np.exp(-(x / limit)))

def normalize(X, y, mode='std'):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_max = X.max(axis=0)
    y_mean = y.mean()
    y_std = y.std()
    y_max = y.max()
    if mode == 'std':
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std
        return X, y, (X_mean, X_std), (y_mean, y_std)
    elif mode == 'dev':
        quantile = 0.9
        denormalizers = []
        for i in range(X.shape[1]):
            val0 = find_nth_quantile(X[:, i], 0)
            val1 = find_nth_quantile(X[:, i], quantile)
            affineX = 0.1 + (X[:, i] - val0) / (val1 - val0 + eps)
            X[:, i] = np.tanh(affineX)
            denormalizers.append([val0, val1])
        val0 = min(find_nth_quantile(y, 0), 1e5)
        val1 = find_nth_quantile(y, quantile)
        affiney = 0.1 + (y - val0) / (val1 - val0 + eps)
        y = np.tanh(affiney)
        # y = affiney
        return X, y, np.array(denormalizers).T, (val0, val1)
    elif mode == 'max':
        X = X / X_max
        y = y / y_max
        return X, y, X_max, y_max
    else:
        raise ValueError('Invalid normalization mode')

def denormalize(y, y_normalizer, mode='std'):
    if mode == 'std':
        y_mean, y_std = y_normalizer
        y = y * y_std + y_mean
        return y
    elif mode == 'dev':
        y_min, y_max = y_normalizer
        div, mod = np.divmod(y, 1.0)
        y = (y_max - y_min + eps) * (div + np.arctanh(mod) - 0.1) + y_min
        # y = (y_max - y_min + eps) * (y - 0.1) + y_min
        return y
    elif mode == 'max':
        y = y * y_normalizer
        return y
    else:
        raise ValueError('Invalid normalization mode')

def plot_r2_vs_param(results, param_name, show=False):
    plt.figure()
    plt.plot(results[param_name], results['mean_test_score'], 'o')
    plt.xlabel(param_name.removeprefix('param_'))
    plt.ylabel('R2')
    if show: plt.show()

def analyze_errors(y, y_pred, name, aux=None, plot=False, show=False):
    if aux is not None:
        y = aux[0]
        y_pred = (1 - y_pred) * aux[1]
        error = y_pred - y
        abs_errors = np.abs(error)
        rel_errors = np.abs(error / y)
    else:
        error = y_pred - y
        abs_errors = np.abs(error)
        rel_errors = np.abs(error)# / y)
    print('[%s] absolute errors: %3.3f, %3.3f, %3.3f' % (name, abs_errors.min(), abs_errors.mean(), abs_errors.max()))
    print('[%s] relative errors: %3.3f, %3.3f, %3.3f' % (name, rel_errors.min(), rel_errors.mean(), rel_errors.max()))
    if plot or show:
        plt.figure()
        plt.plot(y, y_pred, 'o')
        plt.title(name)
        plt.xlabel('y_%s' % name)
        plt.ylabel('y_pred_%s' % name)
        if show: plt.show()