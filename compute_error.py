import os
import pandas as pd

def load_and_compare(folder1, folder2, save_path):
    files1 = sorted(os.listdir(folder1))[:8]
    files2 = sorted(os.listdir(folder2))[:8]

    results = []

    for f1, f2 in zip(files1, files2):
        path1 = os.path.join(folder1, f1)
        path2 = os.path.join(folder2, f2)

        try:
            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)

            if 'Closeness Centrality' not in df1.columns or 'Closeness Centrality' not in df2.columns or \
               'Eigenvector Centrality' not in df1.columns or 'Eigenvector Centrality' not in df2.columns:
                print(f"Error: Columns missing in files {f1} or {f2}")
                continue

            closeness_error = abs(df1['Closeness Centrality'] - df2['Closeness Centrality']) / df2['Closeness Centrality']
            eigenvector_error = abs(df1['Eigenvector Centrality'] - df2['Eigenvector Centrality']) / df2['Eigenvector Centrality']

            avg_closeness_error = closeness_error.mean()
            avg_eigenvector_error = eigenvector_error.mean()

            results.append({
                'File': f1,
                'Avg Closeness Centrality Error': avg_closeness_error,
                'Avg Eigenvector Centrality Error': avg_eigenvector_error
            })

        except Exception as e:
            print(f"Failed to process files {f1} and {f2}: {e}")

    error_df = pd.DataFrame(results)
    error_df.to_csv(save_path, index=False)

# Usage
load_and_compare('Approximated_network_dataframes', 'network_dataframe', 'Error_file.csv')
