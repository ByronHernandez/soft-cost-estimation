from lib.analysis import initial_analysis, completeness_analysis, correlation_analysis
from lib.models import random_forest_experiment, neural_network_experiment, ransac_neural_experiment

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'data/seera_raw.xlsx'
    original_seera = initial_analysis(path, verbose=False)
    clean_seera, rows_to_keep = completeness_analysis(path, verbose=False)
    correlation_analysis(clean_seera, rows_to_keep, output='% project gain (loss)', show=False)
    print('Number of rows to keep:', len(rows_to_keep))
    print('Number of rows:', len(clean_seera.data))

    # Prepare the data for the experiments
    gain_cost = clean_seera.data['% project gain (loss)'].values.astype(float)
    actual_cost = clean_seera.data['Actual cost'].values.astype(float)
    estimated_cost = clean_seera.data['Estimated cost'].values.astype(float)
    clean_seera.delete_column('Actual cost')
    clean_seera.delete_column('Estimated cost')
    clean_seera.delete_column('% project gain (loss)')
    
    X = clean_seera.data.values.astype(float)
    y = gain_cost

    forest = random_forest_experiment(X, y, (actual_cost, estimated_cost))
    neural = neural_network_experiment(X, y, (actual_cost, estimated_cost))
    ransac = ransac_neural_experiment(X, y, (actual_cost, estimated_cost))
