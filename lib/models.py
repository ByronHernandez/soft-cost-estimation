import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold

from .utils import normalize, denormalize, analyze_errors, plot_r2_vs_param, empirical_distribution

def random_forest_experiment(X, y, aux=None):
    empirical_distribution(y, plot=False, show=False)
    X_norm, y_norm, X_normalizer, y_normalizer = normalize(X, y, mode='max')

    model = RandomForestRegressor(criterion='squared_error')
    # param_grid = {"n_estimators": list([1, 10, 20, 30, 40, 50, 60]),
    #               "min_samples_leaf": list([1, 10, 20, 30, 40, 50, 60]),}
    param_grid = {"n_estimators": list(range(1, 61, 3)),
                  "min_samples_leaf": list(range(1, 61, 3)),}
    grid_search = GridSearchCV(model, param_grid, cv=KFold(5), error_score='raise', scoring='max_error', 
                                                               return_train_score=True, n_jobs=16, verbose=1)
    grid_search.fit(X_norm, y_norm)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)
    print(grid_search.best_estimator_)

    results = pd.DataFrame(grid_search.cv_results_)
    plot_r2_vs_param(results, 'param_n_estimators', show=True)
    plot_r2_vs_param(results, 'param_min_samples_leaf', show=True)

    y_pred_norm = grid_search.best_estimator_.predict(X_norm)
    y_pred = denormalize(y_pred_norm, y_normalizer, mode='max')
    analyze_errors(y_norm, y_pred_norm, 'normalized', aux=None, plot=False, show=False)
    analyze_errors(y, y_pred, 'original', aux=aux, plot=False, show=False)

    return grid_search.best_estimator_


def neural_network_experiment(X, y, aux=None):
    X_norm, y_norm, X_normalizer, y_normalizer = normalize(X, y, mode='max')
    model = MLPRegressor(hidden_layer_sizes=(50, 15, 5), max_iter=5000)
    param_grid = {"activation": ['relu'], # [‘identity', ‘logistic', ‘tanh', ‘relu'],
                  "solver": ['lbfgs'], # ['lbfgs', 'sgd', 'adam'],
                  "alpha": [1e-1, 1e-2], #[1e-1, 1e-2, 1e-3],
                  "learning_rate": ['constant', 'adaptive'], #['constant', 'invscaling', ‘adaptive'],
                  "learning_rate_init": [1e-1, 1e-2] # [1e-1, 1e-2, 1e-3],
                  }
    grid_search = GridSearchCV(model, param_grid, cv=KFold(5), error_score='raise', scoring='max_error', 
                                                               return_train_score=True, n_jobs=16, verbose=1)
    grid_search.fit(X_norm, y_norm)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)
    print(grid_search.best_estimator_)

    y_pred_norm = grid_search.best_estimator_.predict(X_norm)
    y_pred = denormalize(y_pred_norm, y_normalizer, mode='max')
    analyze_errors(y_norm, y_pred_norm, 'normalized', aux=None, plot=False, show=False)
    analyze_errors(y, y_pred, 'original', aux=aux, plot=False, show=False)
    
    return grid_search.best_estimator_

def ransac_neural_experiment(X, y, aux=None):
    X_norm, y_norm, X_normalizer, y_normalizer = normalize(X, y, mode='max')
    model = MLPRegressor(hidden_layer_sizes=(50, 15, 5), max_iter=5000, activation='relu', 
                         solver='lbfgs', alpha=1e-2, learning_rate='constant', learning_rate_init=1e-2)
    # 
    N = len(y_norm)
    gamma = 0.2
    inlier_indexes = np.array([])

    for i in range(500):
        indexes_r = np.array(list(set(range(len(X_norm))) - set(inlier_indexes)))             # remaining indexes
        indexes_i = np.random.choice(indexes_r, max(1, len(indexes_r) // 4), replace=False)   # random indexes
        indexes_c = np.array(list(set(range(len(X_norm))) - set(indexes_i)))                  # complementary indexes
        indexes_i = np.append(indexes_i, inlier_indexes).astype(int)                          # inliers + random (new) indexes
        X_i = X_norm[indexes_i]
        # X_c = X_norm[indexes_c]
        y_i = y_norm[indexes_i]
        # y_c = y_norm[indexes_c]
        model.fit(X_i, y_i)
        y_pred_norm_i = model.predict(X_norm)
        y_pred_i = denormalize(y_pred_norm_i, y_normalizer, mode='max')

        error = np.abs(((1 - y_pred_i)*aux[1] - aux[0]) / aux[0])

        # error = np.abs((y_pred_i - y)) # / y)
        
        inlier_indexes_i = np.where(error < gamma)[0]
        if len(inlier_indexes_i) > len(inlier_indexes):
            inlier_indexes = inlier_indexes_i
            print('Iteration %03d: %d inliers' % (i, len(inlier_indexes)))
        if len(inlier_indexes) > N * 0.9: break

    X_norm_f = X_norm[inlier_indexes]
    y_norm_f = y_norm[inlier_indexes]
    y_f = y[inlier_indexes]
    model.fit(X_norm_f, y_norm_f)
    y_pred_norm = model.predict(X_norm_f)
    y_pred_f = denormalize(y_pred_norm, y_normalizer, mode='max')
    analyze_errors(y_norm_f, y_pred_norm, 'normalized', aux=None, plot=False, show=False)
    analyze_errors(y_f, y_pred_f, 'original', aux=(aux[0][inlier_indexes], aux[1][inlier_indexes]), plot=False, show=False)

        

if __name__ == "__main__":
    pass
