'''
Initial script to load data from the database
'''

import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # read the excel file
    f = pd.read_excel('data/seera.xlsx', header=0)
    groupindex, key = {}, None
    
    # collect the column groups
    for i, name in enumerate(f.columns):
        if name[:7] == 'Unnamed':
            groupindex[key].append(i)
        else:
            key = name
            groupindex[key] = [i]
    # print('Group                 Indexes\n')
    for key in groupindex:
        pass
        # print(key, ' ' * (20 - len(key)), groupindex[key])
    
    # collect the column names
    f = pd.read_excel('data/seera.xlsx', header=1)
    names = list(f.columns)
    # print('\n\nidx   Name\n')
    for i, name in enumerate(names):
        pass
        # print('%03d  ' % i, name)

    # collect groups and names
    groupnames = {}
    for key in groupindex:
        groupnames[key] = [names[i] for i in groupindex[key]]
    # print('\n\nGroup                 Names\n')
    for key in groupnames:
        for name in groupnames[key]:
            pass
            # print(key, ' ' * (20 - len(key)), name)
            print(str(key) + ',' + str(name))

    # collect the data
    data = f.values
    # print('\n\nData shape:', data.shape)
    # print(data)

    # print(f['% project gain (loss)'])
    nan, missed, num, total = 0, 0, 0, 0
    for i in range(len(f)):
        if f.values[i, names.index('% project gain (loss)')] in ['?', 'Not exist']:
            # print(i, f.values[i, names.index('% project gain (loss)')])
            missed += 1
        elif np.isnan(f.values[i, names.index('% project gain (loss)')]):
            nan += 1
        else:
            num += 1
        total += 1
    print('nan:', nan, 'missed:', missed, 'num:', num, 'total:', total)

    # print(f.values[:, names.index('% project gain (loss)')])

    col = f.values[:, names.index('% project gain (loss)')]
