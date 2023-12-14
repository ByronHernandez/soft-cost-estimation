import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

from multiprocessing import Pool

from .utils import analyze_errors, plot_r2_vs_param


class Experiment():
    def __init__(self, X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv):
        self.X = X
        self.y = y
        self.actual_cost = actual_cost
        self.estimated_cost = estimated_cost
        # Create the train and test sets
        self.X_train = X[train_indexes]
        self.y_train = y[train_indexes]
        self.estimated_cost_train = estimated_cost[train_indexes]
        self.actual_cost_train = actual_cost[train_indexes]
        self.X_test = X[test_indexes]
        self.y_test = y[test_indexes]
        self.estimated_cost_test = estimated_cost[test_indexes]
        self.actual_cost_test = actual_cost[test_indexes]
        self.normalize()

    def normalize(self):
        # Obtain min, max, std and mean of the train set
        self.X_train_min = np.min(self.X_train, axis=0)
        self.X_train_max = np.max(self.X_train, axis=0)
        self.X_train_std = np.std(self.X_train, axis=0)
        self.X_train_mean = np.mean(self.X_train, axis=0)
        self.y_train_min = np.min(self.y_train)
        self.y_train_max = np.max(self.y_train)
        self.y_train_std = np.std(self.y_train)
        self.y_train_mean = np.mean(self.y_train)
        # Normalize the train set
        self.X_train_normalized = self.X_train.copy() / self.X_train_max
        self.y_train_normalized = self.y_train.copy() / self.y_train_max
        # Normalize the test set
        self.X_test_normalized = self.X_test.copy() / self.X_train_max
        self.y_test_normalized = self.y_test.copy() / self.y_train_max

    def denormalize(self, y):
        return y * self.y_train_max
    
    def run(self):
        self.grid_search.fit(self.X_train_normalized, self.y_train_normalized)
        self.best_params = self.grid_search.best_params_
        self.best_score = self.grid_search.best_score_
        self.best_model = self.grid_search.best_estimator_
        rel_errors_test = self.analyze_errors()
        return rel_errors_test
    
    def analyze_errors(self):
        self.y_train_pred_normalized = self.best_model.predict(self.X_train_normalized)
        self.y_train_pred = self.denormalize(self.y_train_pred_normalized)
        rel_errors_train_normalized = analyze_errors(self.y_train_normalized, self.y_train_pred_normalized, 
                      'normalized gain - train', None,                                                plot=False, show=False)
        rel_errors_train = analyze_errors(self.y_train, self.y_train_pred, 
                      'actual cost     - train', (self.actual_cost_train, self.estimated_cost_train), plot=False, show=False)
        
        self.y_test_pred_normalized = self.best_model.predict(self.X_test_normalized)
        self.y_test_pred = self.denormalize(self.y_test_pred_normalized)
        rel_errors_test_normalized = analyze_errors(self.y_test_normalized, self.y_test_pred_normalized, 
                       'normalized gain  - test', None,                                              plot=False, show=False)
        rel_errors_test = analyze_errors(self.y_test, self.y_test_pred, 
                       'actual cost      - test', (self.actual_cost_test, self.estimated_cost_test), plot=False, show=False)

        return rel_errors_test


class RandomForestExperiment(Experiment):
    def __init__(self, X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv):
        super().__init__(X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv)
        # Create the model
        self.model = RandomForestRegressor(criterion='squared_error')
        self.param_grid = {
                            "n_estimators": list(range(1, 61, 3)), 
                            "min_samples_leaf": list(range(1, 61, 3)),
                          }
        self.cv = cv
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=cv, error_score='raise', verbose=1,
                                        scoring='max_error', return_train_score=True, n_jobs=16)


class NeuralNetworkExperiment(Experiment):
    def __init__(self, X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv):
        super().__init__(X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv)
        # Create the model
        self.model = MLPRegressor(hidden_layer_sizes=(50, 15, 5), max_iter=5000)
        self.param_grid = { 
                            "activation": ['relu'], # [‘identity', ‘logistic', ‘tanh', ‘relu'],
                            "solver": ['lbfgs'], # ['lbfgs', 'sgd', 'adam'],
                            "alpha": [1e-1, 1e-2], #[1e-1, 1e-2, 1e-3],
                            "learning_rate": ['constant', 'adaptive'], #['constant', 'invscaling', ‘adaptive'],
                            "learning_rate_init": [1e-1, 1e-2] # [1e-1, 1e-2, 1e-3],
                          }
        self.cv = cv
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=cv, error_score='raise', verbose=1,
                                        scoring='max_error', return_train_score=True, n_jobs=16)


class NeuralRANSACExperiment(Experiment):
    def __init__(self, X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv):
        super().__init__(X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv)

        self.model = model = MLPRegressor(hidden_layer_sizes=(50, 15, 5), max_iter=5000, activation='relu', 
                         solver='lbfgs', alpha=1e-2, learning_rate='constant', learning_rate_init=1e-2)

    def run(self):
        N = len(self.y_train)
        P = 16
        M = 50
        gamma = 0.3
        
        inlier_indexes = np.array([])
        for i in range(M):
            # indexes_r = np.array(list(set(range(N)) - set(inlier_indexes)))                       # remaining indexes
            # indexes_i = np.random.choice(indexes_r, max(1, len(indexes_r) // 2), replace=False)   # random indexes
            # indexes_i = np.append(indexes_i, inlier_indexes).astype(int)                          # inliers + random (new) indexes
            indexes_i = np.random.choice(range(N), N //2 , replace=False)
            X_i = self.X_train_normalized[indexes_i]
            y_i = self.y_train_normalized[indexes_i]
            self.model.fit(X_i, y_i)
            y_test_pred_norm_i = self.model.predict(self.X_test_normalized)
            y_test_pred_i = self.denormalize(y_test_pred_norm_i)

            # error = np.abs((self.y_test - y_test_pred_i)/(1 - self.y_test))
            error = np.abs(((1 - y_test_pred_i) * self.estimated_cost_test - self.actual_cost_test) / self.actual_cost_test)
            
            inlier_indexes_i = np.where(error < gamma)[0]
            if len(inlier_indexes_i) > len(inlier_indexes):
                inlier_indexes = inlier_indexes_i
                # print('Iteration %03d: %d inliers' % (i, len(inlier_indexes)))
            if len(inlier_indexes) == N: break
        
        print('Final number of inliers:', len(inlier_indexes))
        self.best_model = self.model
        self.best_model.fit(self.X_train_normalized[inlier_indexes], self.y_train_normalized[inlier_indexes])
        return self.analyze_errors()


        


        

        
