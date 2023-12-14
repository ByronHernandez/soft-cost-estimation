'''
Initial script to load data from the database
'''

import numpy as np
import pandas as pd
import seaborn as sns
from .seera import SEERA
import matplotlib.pyplot as plt


def analyze_significance(seera):
    seera.print_column_names()
    seera.print_groups_indexes()
    seera.print_group_names()

def analyze_attributes(seera):
    complete = 0
    print('\n\nAttribute' + ' ' * 42 + " 'nan'     'miss'     'ok'       'total'\n" + "-" * 96)
    for key in seera.names:
        name, analysis = seera.analyze_missing_values_per_attribute(key)
        if analysis['ok'] == analysis['total']: complete += 1
        print(name + ' ' * (50 - len(name)), analysis)
    print('Number of attributes in all the projects:', complete)

def analyze_projects(seera):
    complete = 0
    print("\n\nPrj  'nan'     'miss'     'ok'      'total'\n" + "-" * 47)
    for i in range(len(seera.data)):
        name, analysis = seera.analyze_missing_values_per_project(i)
        if analysis['ok'] == analysis['total']: complete += 1
        print(name, analysis)
    print('Number of projects with all the attributes:', complete)

def initial_analysis(path, verbose=False):
    # Load data from the database
    seera = SEERA(path)
    if verbose:
        # Print the names of the columns
        analyze_significance(seera)
        # Print the number of missing values per attribute
        name, analysis = seera.analyze_missing_values_per_attribute('% project gain (loss)')
        print(name, analysis, '\n')
        name, analysis = seera.analyze_missing_values_per_attribute(12)
        print(name, analysis)
        # Print the number of missing values per project
        analyze_attributes(seera)
        analyze_projects(seera)
    return seera

def completeness_analysis(path, verbose=False):
    seera = SEERA(path)
    # Columns to delete [from own analysis]
    columns = [
        'ProjID',                     # not relevant
        'Organization id',            # not relevant
        # 'Estimated  duration',        # its an estimation from a formula
        'Estimated size',             # too many missing values: 107
        # 'Estimated effort',           # its an estimation from a formula
        'Economic instability impact',# formula including unknown values
        'Top management support',     # formula including unknown values
        'Requirment stability',       # formula including unknown values
        'Team contracts',             # too many missing values: 11
        'Team continuity ',           # formula including unknown values
        'Schedule quality',           # formula including unknown values 
        'Programming language used',  # very ambiguous
        'Technical stability',        # formula including unknown values
        'Outsourcing impact',         # too many missing values: 109
        'Degree of standards usage',  # too many missing values: 99
        'Process reengineering',      # formula including unknown values
               ]

    out_columns = [
        'Actual duration',            # not available a-priori
        # '% project gain (loss)',      # too many missing values: 56
        'Actual effort',              # not available a-priori
                  ]

    categorical_columns = [
        'Organization type',
        'Role in organization',
        'Customer organization type',
        'Development type',
        'Application domain',
        'Methodology',
        'DBMS used'
    ]

    for column in columns + out_columns + categorical_columns:
        seera.delete_column(column)
    
    if verbose:
        print('\n' + '=' * 28 + '\n=> After deleting columns <=\n' + '=' * 28 + '\n')
        analyze_significance(seera)
        analyze_attributes(seera)
        analyze_projects(seera)

    # Recover the output columns
    cost = seera.read_from_sheet("General Information", "Actual incurred costs ")
    seera.add_column('Actual cost', 'General Information', cost)
    estimated_cost = seera.read_from_sheet("General Information", "Contract price ")
    seera.add_column('Estimated cost', 'General Information', estimated_cost)
    
    if verbose:
        print('\n' + '=' * 25 + '\n=> After adding output <=\n' + '=' * 25 + '\n')
        analyze_significance(seera)
        analyze_attributes(seera)
        analyze_projects(seera)

    # Delete projects w/o output or with missing values
    rows_to_keep = list(range(len(seera.data)))
    rows_to_delete = [4] # projects with missing values. 4 is outlier

    for i in range(len(seera.data)):
        for column in seera.names:
            if seera.data.loc[i, column] in ['?', 'Not exist']: 
                rows_to_delete.append(i)
                seera.data.loc[i, column] = np.nan
            else: 
                seera.data.loc[i, column] = float(seera.data.loc[i, column])
    # in case the output(cost) is missing
    for i in range(len(cost)):
        if cost[i] in ['?', 'Not exist'] or i in rows_to_delete: rows_to_keep.remove(i)
    
    seera.data = seera.data.iloc[rows_to_keep]

    if verbose:
        print('\n' + '=' * 40 + '\n=> After deleting projects w/o output <=\n' + '=' * 40 + '\n')
        analyze_significance(seera)
        analyze_attributes(seera)
        analyze_projects(seera)

    return seera, rows_to_keep

def correlation_analysis(seera, rows_to_keep, output, font_size=6, figsize=(8, 8), show=False):
    # Compute the correlation matrix
    corr = seera.data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, square=True, linewidths=0.5, annot=False, cbar_kws={"shrink": 1},
                xticklabels=True, yticklabels=True, annot_kws={"size": font_size}, mask=mask, vmin=-0.8, vmax=0.8)
    # Set the font size for x and y tick labels
    ax.tick_params(axis='both', labelsize=font_size)
    # Save as png, eps, and pdf formats with adjusted layout
    # plt.tight_layout()
    plt.savefig('output/correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig('output/correlation.eps', dpi=300, format='eps', bbox_inches='tight')

    # Isolated analisys
    corr_var = corr[[output]]
    corr_var = corr_var.drop(output, axis=0)
    corr_var = corr_var.sort_values(by=output, ascending=False)
    
    # Using heatmaps
    f, ax = plt.subplots(figsize=figsize)
    # Convert the series into a DataFrame
    corr_var_df = corr_var.unstack().to_frame().T
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_var_df, cmap=cmap, square=True, linewidths=0.5, annot=False, cbar_kws={"shrink": 1},
                xticklabels=True, yticklabels=False, annot_kws={"size": font_size}, vmin=-0.6, vmax=0.6)
    # Set the font size for x and y tick labels
    ax.tick_params(axis='both', labelsize=font_size)
    ax.set_ylabel('Correlation with %s' % output), ax.set_xlabel('')
    # Save as png, eps, and pdf formats with adjusted layout
    plt.savefig('output/correlation_cost_heat.png', dpi=300, bbox_inches='tight')
    plt.savefig('output/correlation_cost_heat.eps', dpi=300, format='eps', bbox_inches='tight')
    plt.savefig('output/correlation_cost_heat.pdf', dpi=300, format='pdf', bbox_inches='tight')

    # Using barplots
    f, ax = plt.subplots(figsize=figsize)
    # Create a DataFrame for better visualization
    corr_var_df = pd.DataFrame(corr_var).reset_index()
    corr_var_df.columns = ['', 'Correlation with %s' % output]
    # Plot the correlations of the variable of interest against all others
    sns.barplot(y='', x='Correlation with %s' % output , data=corr_var_df, ax=ax)
    # Set the font size for x and y tick labels
    ax.tick_params(axis='both', labelsize=font_size)
    # Save as png, eps, and pdf formats with adjusted layout
    plt.savefig('output/correlation_cost_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig('output/correlation_cost_bars.eps', dpi=300, format='eps', bbox_inches='tight')
    plt.savefig('output/correlation_cost_bars.pdf', dpi=300, format='pdf', bbox_inches='tight')
    if show: plt.show()
