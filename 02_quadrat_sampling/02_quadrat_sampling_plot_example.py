import geopandas as gpd
import random
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# def get_all_parquet_files(directory):
#     """
#     Retrieve all parquet file paths from the specified directory.
#     """
#     return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".parquet")]

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
        

def perform_quadrat_sampling(boundary_polygon, quadrat_size, num_quadrats):
    """
    Perform quadrat sampling to calculate statistics on point densities, applying a rotation based on the tilt angle.
    """
    sampled_quadrats = [create_random_quadrat(boundary_polygon, quadrat_size) for _ in range(num_quadrats)]

    quadrats_gdf = gpd.GeoDataFrame(geometry=sampled_quadrats, crs="EPSG:2193")
    return quadrats_gdf


def get_rec_points(parquet_file):
    # Read parquet file
    gdf = gpd.read_parquet(parquet_file)
    
    # Find the rectangle metrics
    rectangle_row = gdf[gdf['type'] == 'rectangle']

    points_row = gdf[gdf['type'] == 'point']
    return rectangle_row, points_row

# Function to swap x and y coordinates
def swap_coordinates(geometry):
    if geometry.geom_type == 'Polygon':  # For rectangles
        return type(geometry)([(y, x) for x, y in geometry.exterior.coords])
    elif geometry.geom_type == 'Point':  # For points
        return type(geometry)(geometry.y, geometry.x)
    return geometry

if __name__ == "__main__":
    file = '' # choose a file from parquet files
    quadrat_size = 1
    num_of_quadrats = 50
    rectangle_gdf, points_gdf = get_rec_points(file)
    rectangle = rectangle_gdf.iloc[0]['geometry']
    x_edges  = [coord[0] for coord in rectangle.exterior.coords]
    y_edges  = [coord[1] for coord in rectangle.exterior.coords]

    row_width = max(x_edges) * 2
    row_length = max(y_edges)
    quadrats = perform_quadrat_sampling(rectangle, quadrat_size, num_of_quadrats)


    # Apply the transformation to all geometries
    rectangle_gdf['geometry'] = rectangle_gdf['geometry'].apply(swap_coordinates)
    points_gdf['geometry'] = points_gdf['geometry'].apply(swap_coordinates)
    quadrats['geometry'] = quadrats['geometry'].apply(swap_coordinates)

    # Define legend handles manually
    rectangle_patch = mpatches.Patch(facecolor='none', edgecolor='blue', linestyle='--', label='Row (Rectangle)')
    quadrat_patch = mpatches.Patch(facecolor='none', edgecolor='green', linestyle='--', label='Quadrats')
    points_patch = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=5, label='Point of Kiwifruits')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    rectangle_gdf.plot(ax=ax, color='lightblue', alpha=0.7, edgecolor='blue', label='_nolegend_')
    # gpd.GeoDataFrame(geometry=[bounding_rectangle]).plot(ax=ax, color='none', edgecolor='orange', linestyle='--', label='Bounding Rectangle')
    points_gdf.plot(ax=ax, color='orange', markersize=2, label='_nolegend_')
    quadrats.plot(ax=ax, color='none', edgecolor='darkgreen', linestyle='--', label='_nolegend_')
    
    # Dynamically adjust the x and y axis limits based on the new bounds
    new_x_edges = [coord[0] for coord in rectangle_gdf.iloc[0].geometry.exterior.coords]
    new_y_edges = [coord[1] for coord in rectangle_gdf.iloc[0].geometry.exterior.coords]

    ax.set_xlim([min(new_x_edges) - 2, max(new_x_edges) + 2])
    ax.set_ylim([min(new_y_edges) - 2, max(new_y_edges) + 2])

    ax.set_title("An Example of Quadrat Sampling Placement", fontsize=22)
    ax.set_xlabel("Row Length", fontsize=18)
    ax.set_ylabel("Row Width", fontsize=18)
    ax.set_aspect('2')
    ax.legend(
        handles=[rectangle_patch, quadrat_patch, points_patch], 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.3),
        fancybox=True,
        shadow=True,
        ncol=3,
        fontsize=16
    )
    
    plt.tight_layout()
    plt.show()
    

