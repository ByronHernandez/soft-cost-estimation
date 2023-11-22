import numpy as np
import pandas as pd

class SEERA():
    def __init__(self, path='data/seera.xlsx'):
        self.path = path
        self.data, self.names = self.load_data()
        self.group_index = self.collect_column_groups()
        self.group_names = self.collect_group_names()

    def load_data(self):
        data_frame = pd.read_excel(self.path, header=1)
        names = list(data_frame.columns)
        return data_frame, names

    def collect_column_groups(self): # idx
        group_index, key = {}, None
        data_frame = pd.read_excel(self.path, header=0)

        for i, name in enumerate(data_frame.columns):
            if name[:7] == 'Unnamed':
                group_index[key].append(i)
            else:
                key = name
                group_index[key] = [i]
        return group_index

    def collect_group_names(self):
        group_names = {}
        for key in self.group_index:
            group_names[key] = [self.names[i] for i in self.group_index[key]]
        return group_names

    def update_group_index(self):
        self.group_index = {}
        for i in range(len(self.names)):
            for key in self.group_names:
                if self.names[i] in self.group_names[key]:
                    if key in self.group_index: self.group_index[key].append(i)
                    else: self.group_index[key] = [i]

    def read_from_sheet(self, sheet_name, name):
        data = pd.read_excel(self.path, sheet_name=sheet_name, header=1)
        return data[name]

    def delete_column(self, column):
        column_idx, column_name = column, column
        if type(column) == str: 
            column_idx = self.names.index(column)
            colunm_name = column
        elif type(column) == int:
            column_idx = column
            column_name = self.names[column]
        else: raise ValueError('Invalid column name or index')
        self.data.drop(column_name, axis=1, inplace=True)
        self.names.pop(column_idx)
        # update group_index and group_names
        for key in self.group_names:
            if column_name in self.group_names[key]:
                self.group_names[key].remove(column_name)
        self.update_group_index()

    def delete_row(self, row):
        if type(row) == int: pass
        else: raise ValueError('Invalid row index')
        self.data.drop(self.data.index[row], inplace=True)

    def add_column(self, name, group_name, values):
        if group_name not in self.group_names: raise ValueError('Invalid group name')
        if len(values) != len(self.data): raise ValueError('Invalid values length')
        index = self.group_index[group_name][-1] + 1
        self.data.insert(index, name, values)
        self.names.insert(index, name)
        self.group_names[group_name].append(name)
        self.update_group_index()

    def analyze_missing_values_per_attribute(self, column):
        if type(column) == str: column = self.names.index(column)
        elif type(column) == int: pass
        else: raise ValueError('Invalid column name or index')

        nan, miss, ok, total = 0, 0, 0, 0
        for i in range(len(self.data)):
            value = self.data.values[i, column]
            if value in ['?', 'Not exist']:
                miss += 1
            elif np.isnan(value):
                nan += 1
            else:
                ok += 1
            total += 1
        return self.names[column], {'nan': nan, 'miss': miss, 'ok': ok, 'total': total}

    def analyze_missing_values_per_project(self, row):
        if type(row) == int: pass
        else: raise ValueError('Invalid row index')

        nan, miss, ok, total = 0, 0, 0, 0
        for i in range(len(self.data.values[row])):
            value = self.data.values[row, i]
            if value in ['?', 'Not exist']:
                miss += 1
            elif np.isnan(value):
                nan += 1
            else:
                ok += 1
            total += 1
        return '%03d' % row, {'nan': nan, 'miss': miss, 'ok': ok, 'total': total}


    def print_column_names(self):
        print('Idx   Name \n' + '-' * 21)
        for i, name in enumerate(self.names):
            print('%03d  ' % i, name)
        print('\n')

    def print_groups_indexes(self):
        print('Group' + ' ' * 17 + 'Indexes\n' + '-' * 64)
        for key in self.group_index:
            print(key, ' ' * (20 - len(key)), self.group_index[key])
        print('\n')

    def print_group_names(self):
        print('Group' + ' ' * 17 + 'Names\n' + '-' * 37)
        for key in self.group_names:
            for name in self.group_names[key]:
                print(key, ' ' * (20 - len(key)), name)
        print('\n')
