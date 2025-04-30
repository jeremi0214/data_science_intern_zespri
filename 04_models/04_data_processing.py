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

def create_random_quadrat(rectangle, quadrat_size):
    """
    Generate a random quadrat of given size fully within a boundary polygon.
    """
    minx, miny, maxx, maxy = rectangle.bounds
    while True:
        random_x = random.uniform(minx, maxx - quadrat_size)
        random_y = random.uniform(miny, maxy - quadrat_size)
        quadrat = box(random_x, random_y, random_x + quadrat_size, random_y + quadrat_size)

        if rectangle.contains(quadrat):
            return quadrat
        

def calculate_quadrat_density(rectangle, points_gdf, quadrat_size, num_of_quadrats):
    """
    Perform quadrat sampling to calculate statistics on point densities.
    """
    sampled_quadrats = [create_random_quadrat(rectangle, quadrat_size) for _ in range(num_of_quadrats)]
    
    # Convert quadrats to GeoDataFrame
    quadrats_gdf = gpd.GeoDataFrame(geometry=sampled_quadrats, crs="EPSG:2193")

    # Quadrat area in square meters
    quadrat_area = quadrat_size ** 2
    
    # Create lists for points within each quadrat and centriod of each quadrat
    points_per_quadrat = []
    distance_to_row_centre = []

    # Calculate points within each quadrat and density, each centroid and average distance to row centre
    for quadrat in quadrats_gdf.geometry:
        points_count = len(points_gdf.sindex.query(quadrat, predicate='contains'))
        points_per_quadrat.append(points_count)
        # Extract the centroid coordinates of the quadrat
        distance_to_row_centre.append(np.abs(quadrat.centroid.x))
        
    densities = np.array(points_per_quadrat) / quadrat_area
    avg_distance_to_row_centre = np.mean(distance_to_row_centre)

    return densities, avg_distance_to_row_centre

def calculate_statistics(parquet_file, quadrat_size, num_of_quadrats):
    """
    Calculate the geometry statistics of the row, the results from quadrat sampling
    and row-centre-only sampling.
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
    row_centre_width = 1 # assume the scanning width of one camera is 1 meter
    row_centre_area = row_centre_width * row_length
    
    # Extract x and y coordinates of points
    gdf["x"] = points_row.geometry.x
    gdf["y"] = points_row.geometry.y
    
    # Calculate the mean and std of x coordinate
    centroid_x = np.mean(gdf["x"])
    std_x = np.std(gdf["x"])

    # Calculate points within row-centre and density
    min_x, max_x = -1, 1

    points_in_row_centre= gdf[
        (gdf["x"] >= min_x) & (gdf["x"] <= max_x) &
        (gdf["y"] >= 0) & (gdf["y"] <= row_length)
    ]
    row_centre_density = len(points_in_row_centre) / row_centre_area

    # Check if quadrat size is too large for random sampling
    if quadrat_size > row_width:
        print(f"Skipping file: quadrat size is larger than row width.")
        return []
    
    # Calcualte the average quadrat density and the variance
    quadrat_densities, avg_distance_to_row_centre = calculate_quadrat_density(
        rectangle, points_row, quadrat_size, num_of_quadrats
    )
    avg_quadrat_density = np.mean(quadrat_densities)

    # Extract kpin, block, row number from file name
    kpin, block, row = parquet_file.split("/")[-1].split(".")[0].split("_")

    results.append({
        "KPIN": kpin,
        "Block": block,
        "Row": row,
        "Row Width": row_width,
        "Row Length": row_length,
        "Row Area": area,
        "Actual Density": actual_density,
        "Row Centre Area": row_centre_area,
        "Row Centre Density": row_centre_density,
        "Average Quadrat Density": avg_quadrat_density,
        "Mean of X Coordinates": centroid_x,
        "Standard Deviation of X Coordinates": std_x,
        "Average Distance to Row Centre": avg_distance_to_row_centre
    })
    return results

def process_selected_files(all_files, quadrat_size, num_of_quadrats):
    """
    Process multiple parquet files and compute density statistics.
    """
    all_results = []
    for file in all_files:
        results = calculate_statistics(file, quadrat_size, num_of_quadrats)
        results = [result for result in results if result is not None]
        all_results.extend(results)
        print(f"Result extracted for {file}!")
    return pd.DataFrame(all_results)


if __name__ == "__main__":
    # Directory containing parquet files
    input_directory = '' # your local directory where parquet files are
    
    # Retrieve all parquet files from the directory
    all_files = get_all_parquet_files(input_directory)
    df = process_selected_files(all_files, 3, 10) # optimal quadrat configs from tests
    print(len(df))

    # Save dataframe into csv file
    output_name = 'data_processed.csv'
    output_dir = '' # save to your local directory
    output_file = os.path.join(output_dir, output_name)
    df.to_csv(output_file, index=False)
    print(f"Data processing completed!")






