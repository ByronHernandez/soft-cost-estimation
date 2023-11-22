from lib.analysis import initial_analysis, completeness_analysis, correlation_analysis

if __name__ == '__main__':
    path = 'data/seera_raw.xlsx'
    original_seera = initial_analysis(path, verbose=True)
    clean_seera, rows_to_keep = completeness_analysis(path, verbose=False)
    correlation_analysis(clean_seera, rows_to_keep)
