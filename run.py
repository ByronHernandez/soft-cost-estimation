from lib.analysis import initial_analysis, completeness_analysis, correlation_analysis, columns, out_columns, categorical_columns
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
    N = 1
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
                    labels=['Random Forest', # \n%.3f$\pm$%.3f' % (forest_errors.mean(), forest_errors.std()),
                            'Neural Network', #\n%.3f$\pm$%.3f' % (neural_errors.mean(), neural_errors.std()),
                            'Neural RANSAC' # \n%.3f$\pm$%.3f' % (ransac_errors.mean(), ransac_errors.std())
                            ],
                    boxprops=boxprops, medianprops=medianprops, patch_artist=True,
                    widths=[box_width]*3)

    # Set the fill color for each box
    for box in bp['boxes']:
        box.set(facecolor='lightblue')

    # plt.ylabel('Relative Error')
    # plt.title('Results')
    # plt.savefig('output/monte_carlo.png')
    # plt.savefig('output/monte_carlo.eps')
    # plt.show()

    def case_study(original_seera, gain_cost, idx, experiment, actual_cost, estimated_cost, name):
        Xidx = original_seera.data.loc[idx:idx+1].values.astype(float)
        yidx = gain_cost[idx:idx+1]

        Xidx_normalized, yidx_normalized = experiment.normalize_test(Xidx, yidx)
        yidx_pred_normalized = experiment.best_model.predict(Xidx_normalized)
        yidx_pred = experiment.denormalize(yidx_pred_normalized)
        cost_pred = (1 - yidx_pred) * estimated_cost[idx]

        print('\nProject ID: %d, %s' % (idx + 1, name))
        print('Estimated cost:', estimated_cost[idx])
        print('Actual cost:', actual_cost[idx])
        print('Predicted cost:', cost_pred[0])
        # print('Gain:', gain_cost[idx])
        # print('Predicted gain:', yidx_pred[0])
        print('Relative error:', np.abs(cost_pred[0] - actual_cost[idx]) / actual_cost[idx])

    # Prapare data for study cases:
    actual_cost = original_seera.read_from_sheet("General Information", "Actual incurred costs ")
    estimated_cost = original_seera.read_from_sheet("General Information", "Contract price ")
    original_seera.add_column('Actual cost', 'General Information', actual_cost)
    original_seera.add_column('Estimated cost', 'General Information', estimated_cost)

    for i in range(len(original_seera.data)):
        for column in original_seera.names:
            if original_seera.data.loc[i, column] in ['?', 'Not exist']: 
                original_seera.data.loc[i, column] = 1
            else: 
                original_seera.data.loc[i, column] = float(original_seera.data.loc[i, column])

    for col in columns + out_columns + categorical_columns:
        original_seera.delete_column(col)

    gain_cost = original_seera.data['% project gain (loss)'].values.astype(float)
    actual_cost = original_seera.data['Actual cost'].values.astype(float)
    estimated_cost = original_seera.data['Estimated cost'].values.astype(float)

    original_seera.delete_column('% project gain (loss)')
    original_seera.delete_column('Actual cost')
    original_seera.delete_column('Estimated cost')

    id, name = 5, 'biggest outlier, '
    case_study(original_seera, gain_cost, id-1, forest, actual_cost, estimated_cost, name + 'forest')
    case_study(original_seera, gain_cost, id-1, neural, actual_cost, estimated_cost, name + 'neural')
    case_study(original_seera, gain_cost, id-1, ransac, actual_cost, estimated_cost, name + 'ransac')

    id, name = 15, 'missing actual, '
    case_study(original_seera, gain_cost, id-1, forest, actual_cost, estimated_cost, name + 'forest')
    case_study(original_seera, gain_cost, id-1, neural, actual_cost, estimated_cost, name + 'neural')
    case_study(original_seera, gain_cost, id-1, ransac, actual_cost, estimated_cost, name + 'ransac')

    id, name = 20, 'biggest gain, '
    case_study(original_seera, gain_cost, id-1, forest, actual_cost, estimated_cost, name + 'forest')
    case_study(original_seera, gain_cost, id-1, neural, actual_cost, estimated_cost, name + 'neural')
    case_study(original_seera, gain_cost, id-1, ransac, actual_cost, estimated_cost, name + 'ransac')

    id, name = 120, 'highest loss, '
    case_study(original_seera, gain_cost, id-1, forest, actual_cost, estimated_cost, name + 'forest')
    case_study(original_seera, gain_cost, id-1, neural, actual_cost, estimated_cost, name + 'neural')
    case_study(original_seera, gain_cost, id-1, ransac, actual_cost, estimated_cost, name + 'ransac')
