'''
Initial script to load data from the database
'''

from .seera import SEERA

def analyze_significance(seera):
    seera.print_column_names()
    seera.print_groups_indexes()
    seera.print_group_names()

def analize_attributes(seera):
    complete = 0
    print('\n\nAttribute' + ' ' * 42 + " 'nan'     'miss'     'ok'       'total'\n" + "-" * 96)
    for key in seera.names:
        name, analysis = seera.analyze_missing_values_per_attribute(key)
        if analysis['ok'] == analysis['total']: complete += 1
        print(name + ' ' * (50 - len(name)), analysis)
    print('Number of attributes in all the projects:', complete)

def analize_projects(seera):
    complete = 0
    print("\n\nPrj  'nan'     'miss'     'ok'      'total'\n" + "-" * 47)
    for i in range(len(seera.data)):
        name, analysis = seera.analyze_missing_values_per_project(i)
        if analysis['ok'] == analysis['total']: complete += 1
        print(name, analysis)
    print('Number of projects with all the attributes:', complete)

def initial_analysis():
    # Load data from the database
    seera = SEERA('data/seera_raw.xlsx')
    # Print the names of the columns
    analyze_significance(seera)
    # Print the number of missing values per attribute
    name, analysis = seera.analyze_missing_values_per_attribute('% project gain (loss)')
    print(name, analysis, '\n')
    name, analysis = seera.analyze_missing_values_per_attribute(12)
    print(name, analysis)
    # Print the number of missing values per project
    analize_attributes(seera)
    analize_projects(seera)

def completeness_analysis():
    seera = SEERA('data/seera_raw.xlsx')
    # Columns to delete [from own analysis]
    columns = [
               'ProjID', 
               'Organization id', 
               'Estimated  duration', 
               'Estimated size', 
               'Estimated effort', 
               'Economic instability impact', 
               'Top management support', 
               'Requirment stability', 
               'Team contracts',       # too many missing values
               'Team continuity ', 
               'Schedule quality', 
               'Programming language used', 
               'Technical stability', 
               'Outsourcing impact', 
               'Degree of standards usage', 
               'Process reengineering',
               ]

    out_columns = [
                   'Actual duration', 
                   '% project gain (loss)', 
                   'Actual effort',
                  ]

    for column in columns + out_columns:
        seera.delete_column(column)

    print('\n' + '=' * 28 + '\n=> After deleting columns <=\n' + '=' * 28 + '\n')    
    # analyze_significance(seera)
    analize_attributes(seera)
    analize_projects(seera)

    # Recover the output columns
    cost = seera.read_from_sheet("General Information", "Actual incurred costs ")
    # seera.add_column('Actual cost', 'General Information', cost)
    
    print('\n' + '=' * 25 + '\n=> After adding output <=\n' + '=' * 25 + '\n')
    # analyze_significance(seera)
    analize_attributes(seera)
    analize_projects(seera)

    # Delete projects w/o output or with missing values
    rows_to_keep = list(range(len(seera.data)))
    rows_to_delete = [] # projects with missing values

    for i in range(len(seera.data)):
        for column in seera.names:
            if seera.data[column][i] in ['?', 'Not exist']: rows_to_delete.append(i)

    for i in range(len(cost)):
        if cost[i] in ['?', 'Not exist'] or i in rows_to_delete: rows_to_keep.remove(i)
    
    seera.data = seera.data.iloc[rows_to_keep]
    cost = cost.iloc[rows_to_keep]

    print('\n' + '=' * 40 + '\n=> After deleting projects w/o output <=\n' + '=' * 40 + '\n')
    analyze_significance(seera)
    analize_attributes(seera)
    analize_projects(seera)
