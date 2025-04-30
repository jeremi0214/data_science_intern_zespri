import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns


def add_info(parquet_file):
    stats = []
    # Get the kpin, block name and row number from the file name
    kpin, block_name, row_number = parquet_file.split('.')[0].split('_')

    # Read parquet file
    gdf = gpd.read_parquet(parquet_file)
    print(gdf)

    rectangle_row = gdf[gdf['type'] == 'rectangle']
    for idx, row in rectangle_row.iterrows():
        rectangle = row['geometry']
        x_edges = [coord[0] for coord in rectangle.exterior.coords]
        y_edges = [coord[1] for coord in rectangle.exterior.coords]

    row_width = max(x_edges) * 2
    row_length = max(y_edges)
    area = row_width * row_length

    points = gdf[gdf['type'] == 'point']
    density = len(points) / area
    
    x_coords = points['geometry'].x
    y_coords = points['geometry'].y
    z_coords = points['geometry'].z

    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    mean_z = np.mean(z_coords)

    q1_x = np.percentile(x_coords, 25)
    q3_x = np.percentile(x_coords, 75)
    iqr_x = q3_x - q1_x

    q1_y = np.percentile(y_coords, 25)
    q3_y = np.percentile(y_coords, 75)
    iqr_y = q3_y - q1_y

    q1_z = np.percentile(z_coords, 25)
    q3_z = np.percentile(z_coords, 75)
    iqr_z = q3_z - q1_z
    
    stats.append({
        "KPIN": kpin,
        "Block Name": block_name,
        "Row Number": row_number,
        "Row Width": row_width,
        "Row Length": row_length,
        "Row area (bounding rectangle)": area,
        "Fruit Density of Row": density,
        "Mean of X coordinate": mean_x,
        "Mean of Y coordinate": mean_y,
        "Mean of Z coordinate": mean_z,        
        "Interquartile Range of X Coordinate": iqr_x,
        "Interquartile Range of Y Coordinate": iqr_y,
        "Interquartile Range of Z Coordinate": iqr_z,
    })

    df = pd.DataFrame(stats)
    
    print(f"row width: {row_width}")
    print(f"row length: {row_length}")
    print(f"row area: {area}")
    print(f"fruit density: {density}")
    print(f"mean of x coordinate: {mean_x}")
    print(f"mean of y coordinate: {mean_y}")
    print(f"mean of z coordinate: {mean_z}")    
    print(f"interquartile range of x coordinate: {q1_x} to {q3_x}")
    print(f"interquartile range of y coordinate: {q1_y} to {q3_y}")
    print(f"interquartile range of z coordinate: {q1_z} to {q3_z}")

    fig, ax = plt.subplots(1, 3, figsize=(18,6))
    sns.histplot(
        x_coords,
        bins=30,
        kde=True,
        color='blue',
        alpha=0.7,
        ax=ax[0]
    )
    ax[0].set_title("Density of X-Coordinates", fontsize=16, fontweight='bold')
    ax[0].set_xlabel("X-Coordinate", fontsize=14)
    ax[0].set_ylabel("Density", fontsize=14)
    ax[0].legend()
    
    sns.histplot(
        y_coords,
        bins=30,
        kde=True,
        color='blue',
        alpha=0.7,
        ax=ax[1]
    )
    ax[1].set_title("Density of Y-Coordinates", fontsize=16, fontweight='bold')
    ax[1].set_xlabel("Y-Coordinate", fontsize=14)
    ax[1].set_ylabel("Density", fontsize=14)
    ax[1].legend()

    sns.histplot(
        z_coords,
        bins=30,
        kde=True,
        color='blue',
        alpha=0.7,
        ax=ax[2]
    )
    ax[2].set_title("Density of Z-Coordinates", fontsize=16, fontweight='bold')
    ax[2].set_xlabel("Z-Coordinate", fontsize=14)
    ax[2].set_ylabel("Density", fontsize=14)
    ax[2].legend()

    plt.tight_layout()
    plt.show()


add_info('') # choose a file from parquet files