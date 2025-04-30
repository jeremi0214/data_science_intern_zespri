import os
import numpy as np
import pandas as pd
import geopandas as gpd
import random
from shapely.geometry import box
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing parquet files
input_directory = '' # your local directory where parquet files are

# Parse filenames and store metadata in a dictionary
metadata = []
for file in os.listdir(input_directory):
    if file.endswith(".parquet"):
        parts = file.split(".")[0].split("_")
        kpin = parts[0]  # Orchard name
        block = parts[1]  # Block name
        row = int(parts[2])  # Row number
        file_path = os.path.join(input_directory, file)
        metadata.append({"kpin": kpin, "block": block, "row": row, "file_path": file_path})

# Convert metadata to a pandas DataFrame
metadata_df = pd.DataFrame(metadata)

def get_parquet_paths(kpin=None, block=None):
    """
    Retrieve file paths for the specified kpin and/or block.
    """
    query = metadata_df.copy()
    if kpin:
        query = query[query['kpin'] == kpin]
    if block:
        query = query[query['block'] == block]
    return query['file_path'].tolist()


def create_random_quadrat(rectangle, width, height):
    """
    Generate a random quadrat of given size fully within a boundary polygon.
    """
    minx, miny, maxx, maxy = rectangle.bounds
    while True:
        random_x = random.uniform(minx, maxx - width)
        random_y = random.uniform(miny, maxy - height)
        random_rectangle = box(random_x, random_y, random_x + width, random_y + height)

        if rectangle.contains(random_rectangle):
            return random_rectangle
        

def calculate_rectangle_density(rectangle, points_gdf, width, height, num_quadrat):
    """
    Computes the density of points within randomly placed rectangle quadrats inside a given rectangle.
    Returns an array of densities computed over multiple quadrats.
    """
    sampled_rectangles = [create_random_quadrat(rectangle, width, height) for _ in range(num_quadrat)]
    
    # Convert quadrats to GeoDataFrame
    rectangles_gdf = gpd.GeoDataFrame(geometry=sampled_rectangles, crs="EPSG:2193")

    # Quadrat area in square meters
    rectangle_area = width * height
    
    # Calculate points within each quadrat and density
    points_per_quadrat = [len(points_gdf.sindex.query(quadrat, predicate='contains')) for quadrat in rectangles_gdf.geometry]
    densities = np.array(points_per_quadrat) / rectangle_area

    return densities

def perform_quadrat_sampling(parquet_file, dimentions, num_quadrats):
    """
    Loads a parquet file and performs rectangle quadrat sampling to estimate point density.
    Calculates bias, variance, and mean square error (MSE) for different quadrat configurations.
    """
    # Read parquet file
    gdf = gpd.read_parquet(parquet_file)

    rectangle_row = gdf[gdf['type'] == 'rectangle']
    rectangle = rectangle_row.iloc[0]['geometry']
    x_edges  = [coord[0] for coord in rectangle.exterior.coords]
    y_edges  = [coord[1] for coord in rectangle.exterior.coords]

    row_width = max(x_edges) * 2
    row_length = max(y_edges)
    area = row_width * row_length

    # Calculate the actual density
    points = gdf[gdf['type'] == 'point']
    actual_density = len(points) / area

    results = []
    # Loop through quadrat sizes and number of quadrats
    for width, height in dimentions:
        for num_quadrat in num_quadrats:
            # Check if quadrat size is too large for random sampling
            if width > row_width:
                print(f"Skipping file: Rectangle dimensions ({width}x{height}) are too large for the row.")
                return []
            densities = calculate_rectangle_density(
                rectangle, points, width, height, num_quadrat
            )
            avg_density = np.mean(densities)
            bias = np.abs(avg_density - actual_density)
            variance = np.var(densities)
            mse = bias ** 2 + variance

            results.append({
                "File": os.path.basename(parquet_file),
                "Width (m)": width,
                "Height (m)": height,
                "Number of Quadrats": num_quadrat,
                "Average Density (points/m²)": avg_density,
                "Actual Density": actual_density,
                "Bias": bias,
                "Variance": variance,
                "Mean Square Error": mse
            })
    return results


def process_selected_files(file_paths, quadrat_sizes, num_quadrats):
    """
    Iterates over multiple parquet files and performs quadrat sampling on each.
    Returns a DataFrame containing density estimation results.
    """
    all_results = []
    for parquet_file in file_paths:
        results = perform_quadrat_sampling(parquet_file, quadrat_sizes, num_quadrats)
        results = [result for result in results if result is not None] # Remove None values
        all_results.extend(results)
        print(f"Result extracted for {parquet_file}!")
    return pd.DataFrame(all_results)

# --- Step 6: Analyze and Plot Results ---
def plot_results(aggregated_df):
    """
    Plots bias vs. variance and mean square error (MSE) against quadrat size.
    Helps visualize the performance of different quadrat configurations.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    
    # Create a new column to represent quadrat dimensions
    aggregated_df["Quadrat Size (m)"] = aggregated_df.apply(
        lambda row: f"{row['Width (m)']}x{row['Height (m)']}", axis=1
    )

    # Plot each combination of quadrat size and number of quadrats
    sns.scatterplot(
        data=aggregated_df,
        x="Bias",
        y="Variance",
        hue="Quadrat Size (m)",
        style="Number of Quadrats",
        palette="viridis",
        s=100,
        ax=ax[0]
    )

    ax[0].set_title("Aggregated Bias vs Variance of Density")
    ax[0].set_xlabel("Bias of Density")
    ax[0].set_ylabel("Variance of Density")
    ax[0].grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax[0].legend(title="Quadrat Dimensions (m) and Number of Quadrats", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot the relationship between quadrat size and total error
    sns.lineplot(
        data=aggregated_df, 
        x='Quadrat Size (m)', 
        y='Mean Square Error',
        hue='Number of Quadrats',
        palette='tab10',
        marker='o',
        ax=ax[1]
    )
    ax[1].set_title("Mean Square Error vs Quadrat Dimensions")
    ax[1].set_xlabel("Quadrat Dimensions (m)")
    ax[1].set_ylabel("Mean Square Error")
    ax[1].grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax[1].legend(title="Number of Quadrats", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Query and process specific data
    selected_files = get_parquet_paths()
    dimentions = [(1, 4), (1, 9), (2, 4.5)]
    df = process_selected_files(selected_files, dimentions, [10, 50, 100, 200])

    # Find optimal configurations across all files
    aggregated_df = (
        df.groupby(["Width (m)", "Height (m)", "Number of Quadrats"])
        .agg({"Bias": "mean",
              "Variance": "mean",
              "Mean Square Error": "mean"})
        .reset_index()
    )
    print("Top configurations balancing MAE and Variance (Mean Square Error):")
    print(aggregated_df.sort_values("Mean Square Error"))

    # plot_results(aggregated_df)