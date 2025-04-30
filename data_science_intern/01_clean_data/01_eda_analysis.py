import os
import numpy as np
import pandas as pd
import geopandas as gpd
import random
from shapely.geometry import box


def get_all_parquet_files(directory):
    """
    Retrieve all parquet file paths from the specified directory.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".parquet")]


def calculate_statistics(parquet_file):
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
    row_centre_width = 1 # assume the scanning width of one camera is 1 meter
    row_centre_area = row_centre_width * row_length
    
    # Extrac x and y coordinates of points
    gdf["x"] = points_row.geometry.x
    gdf["y"] = points_row.geometry.y
    
    # Calculate mean and standard deviation of x and y coordinates
    coord_x_mean = np.mean(gdf["x"])
    coord_y_mean = np.mean(gdf["y"])
    coord_x_std = np.std(gdf["x"])
    coord_y_std = np.std(gdf["y"])

    # Calculate points within row-centre and density
    min_x, max_x = -row_centre_width/2, row_centre_width/2

    points_in_row_centre= gdf[
        (gdf["x"] >= min_x) & (gdf["x"] <= max_x) &
        (gdf["y"] >= 0) & (gdf["y"] <= row_length)
    ]
    row_centre_density = len(points_in_row_centre) / row_centre_area


    # Extract kpin, block, row number from file name
    kpin, block_name, row_num = parquet_file.split("/")[-1].split(".")[0].split("_")

    results.append({
        "KPIN": kpin,
        "Block Name": block_name,
        "Row Number": row_num,
        "Row Width": row_width,
        "Row Length": row_length,
        "Row Area": area,
        "Fruit Count (Row)": len(points_row),
        "Actual Density": actual_density,
        "Row Centre Area": row_centre_area,
        "Row Centre Density": row_centre_density,
        "Mean of X Coordinates": coord_x_mean,
        "Mean of Y Coordinates": coord_y_mean,
        "Standard Deviation of X Coordinates": coord_x_std,
        "Standard Deviation of Y Coordinates": coord_y_std
    })
    return results

def process_selected_files(all_files):
    all_results = []
    for file in all_files:
        results = calculate_statistics(file)
        results = [result for result in results if result is not None]
        all_results.extend(results)
        print(f"Result extracted for {file}!")
    return pd.DataFrame(all_results)



if __name__ == "__main__":
    # Directory containing parquet files
    input_directory = '' # your local directory where parquet files are
    
    # Retrieve all parquet files from the directory
    all_files = get_all_parquet_files(input_directory)
    df = process_selected_files(all_files)

    # Save dataframe into csv file
    output_name = 'eda_report.csv'
    df.to_csv(output_name, index=False)
    print(f"Data processing completed!")

