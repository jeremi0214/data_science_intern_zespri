import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def get_all_parquet_files(directory):
    """
    Retrieve all parquet file paths from the specified directory.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".parquet")]


def calculate_statistics(parquet_file):
    """
    Extracts and calculates statistical metrics from a given parquet file containing spatial data.
    """
    # Read parquet file
    gdf = gpd.read_parquet(parquet_file)
    print(f"Reading {parquet_file}")
    results = []
    
    # Find the rectangle metrics
    rectangle_row = gdf[gdf['type'] == 'rectangle']
    rectangle = rectangle_row.iloc[0]['geometry']
    x_edges  = [coord[0] for coord in rectangle.exterior.coords]
    y_edges  = [coord[1] for coord in rectangle.exterior.coords]

    row_width = max(x_edges) * 2
    row_length = max(y_edges)
    area = row_width * row_length

    # Calculate the actual density
    points_row = gdf[gdf['type'] == 'point']
    actual_density = len(points_row) / area

    # Calculate row-centre area and density
    row_centre_width = 2 # assume the scanning width of one camera is 1 meter
    row_centre_area = row_centre_width * row_length

    # Extract x and y coordinates of points
    gdf["x"] = points_row.geometry.x
    gdf["y"] = points_row.geometry.y

    # Calculate points within row-centre and density
    min_x, max_x = -row_centre_width / 2, row_centre_width / 2

    points_in_row_centre= gdf[
        (gdf["x"] >= min_x) & (gdf["x"] <= max_x) &
        (gdf["y"] >= 0) & (gdf["y"] <= row_length)
    ]
    row_centre_density = len(points_in_row_centre) / row_centre_area

    # Extract kpin, block, row number from file name
    kpin, block, row = parquet_file.split("/")[-1].split(".")[0].split("_")

    results.append({
        "KPIN": kpin,
        "Block": block,
        "Row": row,
        "Row Width": row_width,
        "Row Length": row_length,
        "Row Area": area,
        "Whole Area Density": actual_density,
        "Row Centre Area": row_centre_area,
        "Middle Area Density": row_centre_density,
    })
    return results

def process_selected_files(all_files):
    """
    Process multiple parquet files and extract statistics for each file.
    """
    all_results = []
    for file in all_files:
        results = calculate_statistics(file)
        all_results.extend(results)
        print(f"Result extracted for {file}!")
    return pd.DataFrame(all_results)


def analyse_and_visualisation(df):
    """
    Performs statistical analysis and visualization on the extracted density data.
    """
    # Set global font size
    plt.rc('font', size=16)  # Default text size
    plt.rc('axes', titlesize=18)  # Axes title size
    plt.rc('axes', labelsize=16)  # Axes label size
    plt.rc('legend', fontsize=14)  # Legend font size
    plt.rc('xtick', labelsize=14)  # X-axis tick label size
    plt.rc('ytick', labelsize=14)  # Y-axis tick label size

    # Calculate summary of densities
    mean_whole_density = df['Whole Area Density'].mean()
    mean_middle_density = df['Middle Area Density'].mean()
    std_whole_density = df['Whole Area Density'].std()
    std_middle_density = df['Middle Area Density'].std()

    density_ratios = df['Middle Area Density'] / df['Whole Area Density']
    mean_density_ratio = np.mean(density_ratios)

    print("-"*10, "Density Summary", "-"*10)
    print(f"Mean Density (Whole Area): {mean_whole_density:.4f}")
    print(f"Mean Density (Middle Area): {mean_middle_density:.4f}")
    print(f"Standard Deviation (Whole Area): {std_whole_density:.4f}")
    print(f"Standard Deviation (Middle Area): {std_middle_density:.4f}")
    print(f"Mean Density Ratio (Middle/Whole): {mean_density_ratio:.4f}")

    # Conduct a one-sample t-test to test if the mean density ratio is greater than 1.1
    hypothesized_mean = 1.1
    std_density_ratio = np.std(density_ratios, ddof=1)
    n = len(density_ratios)

    # Calculate the t-statistic
    t_stat = (mean_density_ratio - hypothesized_mean) / (std_density_ratio / np.sqrt(n))

    # Calculate the one-tailed p-value (upper tail)
    p_value = stats.t.sf(t_stat, n-1)

    # Print results
    print("-"*10, "Hypothesis Test Summary", "-"*10)
    print(f"Mean Density Ratio: {mean_density_ratio:.4f}")
    print(f"Standard Deviation of Density Ratio: {std_density_ratio:.4f}")
    print(f"T-statistic for Hypothesis (Mean > 1.1): {t_stat:.4f}")
    print(f"P-value for One-Tailed Test: {p_value:.4f}")

    # Check if both p-value is significant at alpha = 0.05
    if p_value < 0.05:
        print("The density ratio is significantly greater than 1.1 (p < 0.05).")
    else:
        print("The density ratio is NOT significantly greater than 1.1 (p >= 0.05).")


    # Set up a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))

    # --- Plot 1: Density Ratio Distribution ---
    # Plot histogram of density ratios with KDE line
    sns.histplot(density_ratios, bins=30, color='purple', kde=True, stat="density", ax=axes[0], edgecolor='black', alpha=0.7)

    # Add vertical lines for hypothesized mean and actual mean
    axes[0].axvline(hypothesized_mean, color='red', linestyle='dashed', label=f'Hypothesized Mean ({hypothesized_mean})')
    axes[0].axvline(mean_density_ratio, color='blue', linestyle='solid', label=f'Mean Ratio ({mean_density_ratio:.4f})')

    # Labeling and formatting
    axes[0].set_xlabel('Density Ratio (Middle / Whole)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Density Ratio Distribution with Kernel Density Estimation', fontsize=18, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)
    # Highlight key numbers on the plot
    axes[0].text(hypothesized_mean, 1.6, f'{hypothesized_mean}', fontsize=16, color='red', fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
    axes[0].text(mean_density_ratio, 1.2, f'{mean_density_ratio:.4f}', fontsize=16, color='blue', fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
    
    # --- Plot 2: Density Distribution (Whole vs Middle) ---
    # Plot KDE for Whole Area Density
    sns.kdeplot(df['Whole Area Density'], color='blue', label='Whole Area Density', fill=True, alpha=0.4, ax=axes[1], linewidth=2)

    # Plot KDE for Middle Area Density
    sns.kdeplot(df['Middle Area Density'], color='green', label='Middle Area Density', fill=True, alpha=0.4, ax=axes[1], linewidth=2)

    # Add vertical lines for the mean of both densities
    axes[1].axvline(mean_whole_density, color='blue', linestyle='dashed', label=f'Mean Whole ({mean_whole_density:.4f})')
    axes[1].axvline(mean_middle_density, color='green', linestyle='dashed', label=f'Mean Middle ({mean_middle_density:.4f})')

    axes[1].set_xlabel('Density')
    axes[1].set_ylabel('Density Distribution (Normalised)', labelpad=15)
    axes[1].set_title('Distribution of Whole Area vs Middle Area Densities', fontsize=18, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    # Highlight key numbers on the plot
    axes[1].text(mean_whole_density + 0.1, 0.03, f'{mean_whole_density:.4f}', fontsize=16, color='blue', fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
    axes[1].text(mean_middle_density - 0.1, 0.01, f'{mean_middle_density:.4f}', fontsize=16, color='green', fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3'))

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, left=0.05)
    plt.show()


if __name__ == "__main__":
    # Directory containing parquet files
    input_directory = '' # your local directory where parquet files are
    
    # Retrieve all parquet files from the directory
    all_files = get_all_parquet_files(input_directory)
    df = process_selected_files(all_files)
    analyse_and_visualisation(df)

    # # Save dataframe into csv file
    # output_name = 'row_centre_only_sampling.csv'
    # output_dir = '' # save to your local directory
    # output_file = os.path.join(output_dir, output_name)
    # df.to_csv(output_file, index=False)
    # print(f"Data processing completed!")
