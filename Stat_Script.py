import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math 
from scipy.stats import norm
import sys
import os


def print_histogram_with_normal_and_ci(df, name, input_file_name, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the column to plot
    column = name
    data = df[column]

    # Calculate the number of bins based on the square root of the number of data points
    n = len(data)
    num_bins = math.ceil(math.sqrt(n))

    # Create the histogram
    plt.figure(figsize=(10, 6))
    frequencies, bins, patches = plt.hist(data, bins=num_bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)

    # Compute mean and standard deviation
    mean, std = data.mean(), data.std()

    # Generate x values for the normal distribution
    x = np.linspace(min(bins), max(bins), 500)
    pdf = norm.pdf(x, mean, std)

    # Set the color for the bell curve based on the column name
    bell_curve_color = 'blue' if name == 'Paradox Percentage' else 'orange'

    # Plot the normal distribution
    plt.plot(x, pdf, color=bell_curve_color, label=f'Normal Distribution (Mean={mean:.2f}, SD={std:.2f})', linewidth=2)

    # Compute the 95% confidence interval
    z = 1.96  # Z-score for 95% confidence
    ci_lower = mean - z * (std / math.sqrt(n))
    ci_upper = mean + z * (std / math.sqrt(n))

    # Shade the 95% confidence interval
    x_fill = np.linspace(ci_lower, ci_upper, 500)
    pdf_fill = norm.pdf(x_fill, mean, std)
    plt.fill_between(x_fill, pdf_fill, color='lightgreen', alpha=0.5, label='95% Confidence Interval of the mean value')

    # Add labels, title, and legend
    plt.xlabel(name, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'Histogram, Normal Distribution, and 95% CI for {column}', fontsize=16)
    plt.axvline(ci_lower, color='green', linestyle='dashed', linewidth=1.5, label=f'Lower CI: {ci_lower:.2f}')
    plt.axvline(ci_upper, color='red', linestyle='dashed', linewidth=1.5, label=f'Upper CI: {ci_upper:.2f}')
    plt.legend(fontsize=12)

    # Set x-axis ticks to the center of each bin
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.xticks(bin_centers, [f"[{bins[i]:.1f}, {bins[i + 1]:.1f})" for i in range(len(bins) - 1)], rotation=45, fontsize=10)

    # Adjust the plot layout
    plt.tight_layout()

    # Save the plot to the specified output folder
    output_path = os.path.join(output_folder, f"{input_file_name}_{name}_histogram.png")
    plt.savefig(output_path, dpi=300)
    print(f"Histogram with normal distribution and 95% CI saved to {output_path}")

    # Close the plot to free up memory
    plt.close()

def create_statistic_dataframe(df, input_file_name, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Compute the mean and standard deviation
    mean_values = numeric_df.mean()
    std_values = numeric_df.std()

    # Compute the 95% confidence intervals
    n = len(df)  # Number of data points
    z = 1.96  # Z-score for 95% confidence level

    # Calculate confidence intervals
    lower_bounds = mean_values - z * (std_values / np.sqrt(n))
    upper_bounds = mean_values + z * (std_values / np.sqrt(n))

    # Combine results into a single DataFrame
    summary = pd.DataFrame({
        'Mean': mean_values,
        'Standard Deviation': std_values,
        '95% CI Lower Bound': lower_bounds,
        '95% CI Upper Bound': upper_bounds
    })

    # Save the summary DataFrame to the output folder
    output_path = os.path.join(output_folder, f"{input_file_name}_stat_dataframe.csv")
    summary.to_csv(output_path, index_label='Column')
    print(f"Summary statistics saved to '{output_path}'")

def create_bell_curves_with_ci(df, input_file_name, output_folder): 
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Extract the columns to plot
    col1 = numeric_df['Paradox Percentage']
    col2 = numeric_df['Neighbor Paradox Percentage']

    # Compute the mean and standard deviation for both columns
    mean1, std1 = col1.mean(), col1.std()
    mean2, std2 = col2.mean(), col2.std()
    
    n1, n2 = len(col1), len(col2)  # Sample sizes

    # Compute 95% confidence intervals
    z = 1.96  # Z-score for 95% confidence level
    ci1_lower = mean1 - z * (std1 / np.sqrt(n1))
    ci1_upper = mean1 + z * (std1 / np.sqrt(n1))
    ci2_lower = mean2 - z * (std2 / np.sqrt(n2))
    ci2_upper = mean2 + z * (std2 / np.sqrt(n2))

    # Generate x values for the normal distributions
    x = np.linspace(0, 100, 500)  # Assuming both percentages range from 0 to 100

    # Compute the PDFs (Probability Density Functions)
    pdf1 = norm.pdf(x, mean1, std1)
    pdf2 = norm.pdf(x, mean2, std2)

    # Plot the PDFs
    plt.figure(figsize=(12, 6))
    plt.plot(x, pdf1, label=f'Paradox Percentage (Mean={mean1:.2f}, SD={std1:.2f})', color='blue')
    plt.plot(x, pdf2, label=f'Neighbor Paradox Percentage (Mean={mean2:.2f}, SD={std2:.2f})', color='orange')

    # Shade the 95% confidence intervals for both curves
    x_fill1 = np.linspace(ci1_lower, ci1_upper, 500)
    pdf_fill1 = norm.pdf(x_fill1, mean1, std1)
    plt.fill_between(x_fill1, pdf_fill1, color='blue', alpha=0.2, label=f'95% CI Paradox [%{ci1_lower:.2f}, %{ci1_upper:.2f}]')

    x_fill2 = np.linspace(ci2_lower, ci2_upper, 500)
    pdf_fill2 = norm.pdf(x_fill2, mean2, std2)
    plt.fill_between(x_fill2, pdf_fill2, color='orange', alpha=0.2, label=f'95% CI Neighbor [%{ci2_lower:.2f}, %{ci2_upper:.2f}]')

    # Add labels, title, and legend
    plt.xlabel('Percentage', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('Normal Distribution of Paradox Percentages with 95% Confidence Intervals', fontsize=16)
    plt.legend(fontsize=12)

    # Adjust the plot layout
    plt.tight_layout()

    # Save the plot to the specified output folder
    output_path = os.path.join(output_folder, f"{input_file_name}_bellcurves.png")
    plt.savefig(output_path, dpi=300)
    print(f"Bell curves with 95% CI saved to {output_path}")

    # Close the plot to free up memory
    plt.close()

def compare_distributions(input_folder1, input_folder2, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the first CSV file from each input folder
    file1 = os.listdir(input_folder1)[0]
    file2 = os.listdir(input_folder2)[0]
    df1 = pd.read_csv(os.path.join(input_folder1, file1))
    df2 = pd.read_csv(os.path.join(input_folder2, file2))

    # Specifically select the "Neighbor Paradox Percentage" column, and try to convert it to numeric
    col_name = "Paradox Percentage"
    try:
        data1 = pd.to_numeric(df1[col_name], errors='coerce').dropna()
        data2 = pd.to_numeric(df2[col_name], errors='coerce').dropna()
    except KeyError:
        print(f"The column '{col_name}' was not found in one or both of the datasets.")
        return
    except Exception as e:
        print(f"Error processing data in column: {col_name}. Error: {e}")
        return

    if data1.empty or data2.empty:
        print("One or both data columns are empty after attempting to convert to numeric. Check your data files.")
        return

    # Compute mean and standard deviation for both datasets
    mean1, std1 = data1.mean(), data1.std()
    mean2, std2 = data2.mean(), data2.std()

    # Compute 95% confidence intervals
    z = 1.96  # Z-score for 95% confidence
    n1, n2 = len(data1), len(data2)
    ci1_lower = mean1 - z * (std1 / math.sqrt(n1))
    ci1_upper = mean1 + z * (std1 / math.sqrt(n1))
    ci2_lower = mean2 - z * (std2 / math.sqrt(n2))
    ci2_upper = mean2 + z * (std2 / math.sqrt(n2))

    # Plotting both distributions and confidence intervals
    plt.figure(figsize=(12, 6))
    x = np.linspace(0, 100, 500)  # Fixed x-axis range from 0 to 100
    pdf1 = norm.pdf(x, mean1, std1)
    pdf2 = norm.pdf(x, mean2, std2)

    plt.plot(x, pdf1, label="Network Graphs "f'{col_name} (Mean={mean1:.2f}, SD={std1:.2f})', color='blue')
    plt.plot(x, pdf2, label="Random Graphs "f'{col_name} (Mean={mean2:.2f}, SD={std2:.2f})', color='orange')

    plt.fill_between(x, norm.pdf(x, mean1, std1), color='blue', alpha=0.2, label=f'{col_name} 95% CI [{ci1_lower:.2f}, {ci1_upper:.2f}]')
    plt.fill_between(x, norm.pdf(x, mean2, std2), color='orange', alpha=0.2, label=f'{col_name} 95% CI [{ci2_lower:.2f}, {ci2_upper:.2f}]')

    # Add dashed lines for CI bounds
    plt.axvline(ci1_lower, color='blue', linestyle='dashed', label=f'Lower CI {col_name} ({ci1_lower:.2f})')
    plt.axvline(ci1_upper, color='blue', linestyle='dashed', label=f'Upper CI {col_name} ({ci1_upper:.2f})')
    plt.axvline(ci2_lower, color='orange', linestyle='dashed', label=f'Lower CI {col_name} ({ci2_lower:.2f})')
    plt.axvline(ci2_upper, color='orange', linestyle='dashed', label=f'Upper CI {col_name} ({ci2_upper:.2f})')

    plt.xlabel('Percentage')
    plt.ylabel('Probability Density')
    plt.title('Comparison of Normal Distributions with 95% Confidence Intervals')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, 'comparison_plot_P.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f'Comparison plot saved to {output_path}')


def main(): 
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <path_to_mtx_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    mode = sys.argv[3]
    
    input_folder_1 = 'network_stat_data'
    input_folder_2 = 'Random_Graphs_stat_data'

    # Ensure the output folder exists, create if it does not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if mode == 'stat':
        for file in os.listdir(input_folder):

            full_file_path = os.path.join(input_folder, file)
            df = pd.read_csv(full_file_path) 

            names = ['Paradox Percentage', 'Neighbor Paradox Percentage']

            for name in names: 
                print_histogram_with_normal_and_ci(df, name, input_folder, output_folder)
            
            create_statistic_dataframe(df, input_folder, output_folder)

            create_bell_curves_with_ci(df, input_folder, output_folder)

    if mode == 'compare':
        compare_distributions(input_folder_1, input_folder_2, output_folder)
    

if __name__ == "__main__": 
    main()