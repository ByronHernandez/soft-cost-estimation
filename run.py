from lib.analysis import initial_analysis, completeness_analysis, correlation_analysis
from lib.models import RandomForestExperiment, NeuralNetworkExperiment, NeuralRANSACExperiment

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


if __name__ == '__main__':
    path = 'data/seera_raw.xlsx'
    original_seera = initial_analysis(path, verbose=False)
    clean_seera, rows_to_keep = completeness_analysis(path, verbose=False)
    # correlation_analysis(clean_seera, rows_to_keep, output='% project gain (loss)', show=False)
    print('Number of rows to keep:', len(rows_to_keep))
    print('Number of rows:', len(clean_seera.data))

    # Prepare the data for the experiments
    gain_cost = clean_seera.data['% project gain (loss)'].values.astype(float)
    actual_cost = clean_seera.data['Actual cost'].values.astype(float)
    estimated_cost = clean_seera.data['Estimated cost'].values.astype(float)
    clean_seera.delete_column('Actual cost')
    clean_seera.delete_column('Estimated cost')
    clean_seera.delete_column('% project gain (loss)')
    
    # Monte Carlo Experiment
    N = 0
    forest_errors = np.array([])
    neural_errors = np.array([])
    ransac_errors = np.array([])

    # load from txt in output folder
    forest_errors = np.loadtxt('output/errors/forest_errors.txt')
    neural_errors = np.loadtxt('output/errors/neural_errors.txt')
    ransac_errors = np.loadtxt('output/errors/ransac_errors.txt')
    for i in range(N):
        # Split the data into train and test
        test_indexes = np.random.choice(range(len(actual_cost)), len(actual_cost) // 3, replace=False)
        train_indexes = list(set(range(len(actual_cost))) - set(test_indexes))
        X = clean_seera.data.values.astype(float)
        y = gain_cost

        cv = KFold(n_splits=4, shuffle=True, random_state=42)

        forest = RandomForestExperiment(X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv)
        neural = NeuralNetworkExperiment(X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv)
        ransac = NeuralRANSACExperiment(X, y, actual_cost, estimated_cost, train_indexes, test_indexes, cv)

        print('Iteration', i)
        forest_errors = np.append(forest_errors, forest.run())
        neural_errors = np.append(neural_errors, neural.run())
        ransac_errors = np.append(ransac_errors, ransac.run())

    # save as txt in output folder
    np.savetxt('output/errors/forest_errors.txt', forest_errors, fmt='%.4f')
    np.savetxt('output/errors/neural_errors.txt', neural_errors, fmt='%.4f')
    np.savetxt('output/errors/ransac_errors.txt', ransac_errors, fmt='%.4f')

    # Plot the results
    plt.figure(figsize=(6, 4))

    boxprops = dict(color='blue')
    medianprops = dict(color='red', linewidth=2)
    box_width = 0.15

    bp = plt.boxplot([forest_errors, neural_errors, ransac_errors], showfliers=False,
                    labels=['Random Forest', 'Neural Network', 'Neural RANSAC'],
                    boxprops=boxprops, medianprops=medianprops, patch_artist=True,
                    widths=[box_width]*3)

    # Set the fill color for each box
    for box in bp['boxes']:
        box.set(facecolor='lightblue')

    plt.ylabel('Relative Error')
    plt.title('Results')
    plt.savefig('output/monte_carlo.png')
    plt.savefig('output/monte_carlo.eps')
    plt.show()

