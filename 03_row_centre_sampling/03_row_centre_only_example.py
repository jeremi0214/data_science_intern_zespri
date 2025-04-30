import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon 
import seaborn as sns
from scipy import stats


def plot_data(parquet_file):
    # Read parquet file
    gdf = gpd.read_parquet(parquet_file)
    
    # Find the rectangle metrics
    rectangle_row = gdf[gdf['type'] == 'rectangle']
    rectangle = rectangle_row.iloc[0]['geometry']
    x_edges  = [coord[0] for coord in rectangle.exterior.coords]
    y_edges  = [coord[1] for coord in rectangle.exterior.coords]

    row_width = max(x_edges) * 2
    row_length = max(y_edges)

    # Calculate the actual density
    points_row = gdf[gdf['type'] == 'point']

    # Calculate row-centre area and density
    row_centre_width = 2 # assume the scanning width of one camera is 1 meter

    # Extract x and y coordinates of points
    gdf["x"] = points_row.geometry.x
    gdf["y"] = points_row.geometry.y

    # Calculate points within row-centre and density
    min_x, max_x = -row_centre_width / 2, row_centre_width / 2
    row_centre_bounds = [(min_x, 0), (max_x, 0), (max_x, row_length), (min_x, row_length), (min_x, 0)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the rectangle
    x_rect, y_rect = zip(*rectangle.exterior.coords)
    ax.plot(y_rect, x_rect, label="Rectangle Boundary", color="blue", linewidth=2)

    # Plot all points
    ax.scatter(gdf["y"], gdf["x"], color="orange", label="Points", alpha=0.1)

    # Plot the row center area
    row_centre_polygon = Polygon(row_centre_bounds)
    x_row_centre, y_row_centre = row_centre_polygon.exterior.xy
    ax.plot(y_row_centre, x_row_centre, color="red", linestyle="--", linewidth=2, label="Row Centre Area")

    # Set plot limits
    ax.set_xlim(min(y_edges) - 3, max(y_edges) + 3)
    ax.set_ylim(min(x_edges) - 0.2, max(x_edges) + 0.2)

    # Set aspect ratio
    ax.set_aspect('2')
    
    # Labels and legend
    ax.set_title("An Example of Row-cenre-only sampling", fontsize=16)
    ax.set_xlabel("Row Longth", fontsize=12)
    ax.set_ylabel("Row Width", fontsize=12)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.6),  
        fancybox=True,
        shadow=True,
        ncol=3  
    )

    plt.show()

    
parquet_file = '' # choose a file from parquet files
plot_data(parquet_file)
