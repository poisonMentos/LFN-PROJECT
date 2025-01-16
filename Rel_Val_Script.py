import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None
    return df

def compute_relative_coefficient(df1, df2, column_name):
    try:
        relative_coefficient = 1 - ((df1[column_name].iloc[:8] - df2[column_name].iloc[:8]).abs() / df2[column_name].iloc[:8])
        return relative_coefficient
    except Exception as e:
        print(f"Error in computing relative coefficient for column {column_name}: {e}")
        return None

def add_relative_values_column(df, new_data, new_column_name):
    if new_column_name not in df.columns:
        df[new_column_name] = pd.NA  # initialize column if not exist
    df.loc[df.index[:8], new_column_name] = new_data.values
    return df

def process_column(df1, df2, column_name, new_column_name):
    relative_coeff = compute_relative_coefficient(df1, df2, column_name)
    if relative_coeff is not None:
        df1 = add_relative_values_column(df1, relative_coeff, new_column_name)
    return df1

def main():
    path1 = 'Approximated_network_dataframes\\approximate_times.csv'
    path2 = 'network_dataframe\\true_times.csv'
    output_path = 'Approximated_network_dataframes\\approximate_times_rel_values.csv'

    # Load the data
    df1 = load_data(path1)
    df2 = load_data(path2)

    if df1 is not None and df2 is not None:
        # Process both columns
        df1 = process_column(df1, df2, "Closeness Centrality Time (s)", "Rel_Val Close")
        df1 = process_column(df1, df2, "Eigenvector Centrality Time (s)", "Rel_Val Eig")

        # Save the updated DataFrame to the output path
        try:
            df1.to_csv(output_path, index=False)
            print(f"Updated DataFrame saved to {output_path}")
        except Exception as e:
            print(f"Failed to save the updated DataFrame: {e}")

if __name__ == "__main__":
    main()
