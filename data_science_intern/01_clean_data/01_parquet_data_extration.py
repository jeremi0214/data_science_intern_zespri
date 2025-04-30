import os
import pandas as pd
import geopandas as gpd


def get_all_parquet_files(directory):
    """
    Retrieve all parquet file paths from the specified directory.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".parquet")]


def extract_data(parquet_file):
    """
    Extract relevant spatial data from a parquet file containing geometric features.
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

    # Extract kpin, block, row number from file name
    kpin, block, row = parquet_file.split("/")[-1].split(".")[0].split("_")

    # Mapping the results into columns
    results.append({
        "KPIN": kpin,
        "Block": block,
        "Row": row,
        "Row Width": row_width,
        "Row Length": row_length,
        "Row Area": area,
        "Number of Points": len(points_row),
        "Actual Density": actual_density
    })
    return results

def process_selected_files(all_files):
    """
    Process multiple parquet files and extract relevant spatial metrics.
    """
    all_results = []
    for file in all_files:
        results = extract_data(file)
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
    output_name = 'preprocessed_df.csv'
    output_dir = '' # save to your local directory
    output_file = os.path.join(output_dir, output_name)
    df.to_csv(output_file, index=False)
    print(f"Data extraction completed!")






