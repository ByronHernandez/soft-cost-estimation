'''
Initial script to load data from the database
'''

import os
from .seera import SEERA

def initial_analysis():
    os.system('clear')
    seera = SEERA('data/seera.xlsx')
    seera.print_column_names()
    seera.print_groups_indexes()
    seera.print_group_names()
    name, analysis = seera.analyze_missing_values_per_attribute('% project gain (loss)')
    print(name, analysis, '\n')
    name, analysis = seera.analyze_missing_values_per_attribute(12)
    print(name, analysis)

    print('\n\nAttribute' + ' ' * 42 + " 'nan'     'miss'     'ok'       'total'\n" + "-" * 96)
    for key in seera.names:
        name, analysis = seera.analyze_missing_values_per_attribute(key)
        print(name + ' ' * (50 - len(name)), analysis)

    print("\n\nPrj  'nan'     'miss'     'ok'      'total'\n" + "-" * 47)
    for i in range(len(seera.data)):
        name, analysis = seera.analyze_missing_values_per_project(i)
        print(name, analysis)

    columns = [
               'ProjID', 
               'Organization id', 
               'Estimated  duration', 
               'Estimated size', 
               'Estimated effort', 
               'Economic instability impact', 
               'Top management support', 
               'Requirment stability', 
               'Team continuity ', 
               'Schedule quality', 
               'Programming language used', 
               'Technical stability', 
               'Outsourcing impact', 
               'Degree of standards usage', 
               'Process reengineering',
               ]

    for column in columns:
        seera.delete_column(column)

    out_columns = [
                   'Actual duration', 
                   '% project gain (loss)', 
                   'Actual effort',
                  ]

    for column in out_columns:
        seera.delete_column(column)

    # seera.add_column('Actual zeros', 'General Information', [0] * len(seera.data)) # Works

    print('\n\n')
    seera.print_column_names()
    seera.print_groups_indexes()
    seera.print_group_names()

    print('\n\nAttribute' + ' ' * 42 + " 'nan'     'miss'     'ok'       'total'\n" + "-" * 96)
    for key in seera.names:
        name, analysis = seera.analyze_missing_values_per_attribute(key)
        print(name + ' ' * (50 - len(name)), analysis)

    complete = 0
    print("\n\nPrj  'nan'     'miss'     'ok'      'total'\n" + "-" * 47)
    for i in range(len(seera.data)):
        name, analysis = seera.analyze_missing_values_per_project(i)
        if analysis['ok'] == analysis['total']: complete += 1
        print(name, analysis)
    print('Number of projects with all the attributes after cleaning:', complete)
