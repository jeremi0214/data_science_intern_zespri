import os
import zipfile
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
from shapely.geometry import Point, Polygon, box
from shapely.affinity import rotate, translate
import geopandas as gpd


# --- Step 1: Extract Data from KML ---
def extract_polygon_coordinates(polygons):
    """
    Extract coordinates of polygons from KML data.
    """
    if not polygons:
        raise ValueError("No polygons found in the KML file.")
    for poly in polygons:
        points = poly.text.strip().split()
        polygon_coords = []
        for point in points:
            coords = point.split(',')
            if len(coords) >= 3:
                lon, lat, elev = float(coords[0]), float(coords[1]), float(coords[2])
                polygon_coords.append((lon, lat, elev))
        return polygon_coords


def extract_point_coordinates(points):
    """
    Extract coordinates of points from KML data.
    """
    if not points:
        raise ValueError("No points found in the KML file.")
    point_coords = []
    for point in points:
        coords = point.text.strip().split(',')
        if len(coords) >= 2:
            lon, lat, elev= float(coords[0]), float(coords[1]), float(coords[2])
            point_coords.append((lon, lat, elev))
    return point_coords


# --- Step 2: Rotate and Translate the rectangle and points ---
def rotate_and_translate(bounding_rectangle, points_gdf):
    """
    Rotate and translate the bounding rectangle so that:
    - The long side lies on the x-axis.
    - The short sides are parallel to the y-axis.
    - The midpoint of the bottom long side is at the origin (0, 0).
    Rotate the points align with the rectangle.
    """
    # Get the corners of the bounding rectangle
    corners = list(bounding_rectangle.exterior.coords)[:-1]  # Exclude the closing point
    
    # Identify the long and short sides based on lengths
    edge1_length = Point(corners[0]).distance(Point(corners[1]))
    edge2_length = Point(corners[1]).distance(Point(corners[2]))
    
    if edge1_length >= edge2_length:
        dx, dy = corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]
    else:
        dx, dy = corners[2][0] - corners[1][0], corners[2][1] - corners[1][1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate the rectangle
    rotation_angle = 90 - angle
    original_centroid = bounding_rectangle.centroid
    rotated_rectangle = rotate(bounding_rectangle, rotation_angle, origin='centroid', use_radians=False)
    
    # Translate the rectangle
    rotated_centroid = rotated_rectangle.centroid
    translated_rectangle = translate(rotated_rectangle, xoff=-rotated_centroid.x, yoff=-rotated_centroid.y)

    # Find the minimum y-coordinate of the translated polygon
    min_y = min([point[1] for point in translated_rectangle.exterior.coords])

    # Translate the rectangle up so the bottom side is on the x-axis
    final_rectangle = translate(translated_rectangle, xoff=0, yoff=-min_y)    
    
    # Calculate the translation offsets
    translation_x = -rotated_centroid.x
    translation_y = -rotated_centroid.y - min_y

    # Rotate and translate the points
    rotated_points = points_gdf.geometry.apply(lambda point: rotate(point, angle=rotation_angle, origin=original_centroid, use_radians=False))
    translated_points = rotated_points.apply(lambda point: translate(point, xoff=translation_x, yoff=translation_y))
    
    # Convert points back to GeoDataFrame
    aligned_points_gdf = gpd.GeoDataFrame(geometry=translated_points, crs=points_gdf.crs)
    
    return final_rectangle, aligned_points_gdf


# --- Step 5: Process a Single KML File ---
def process_kml_file(kml_file):
    """
    Extract data from KML file for further calculation.
    """
    # Parse the KML file and extract data
    tree = ET.parse(kml_file)
    root = tree.getroot()
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

    polygon = root.findall('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces=namespaces)
    points = root.findall('.//kml:Point/kml:coordinates', namespaces=namespaces)

    # Create GeoDataFrames for boundary and points
    boundary_coords = extract_polygon_coordinates(polygon)
    fruit_coords = extract_point_coordinates(points)

    boundary_polygon = Polygon(boundary_coords)
    fruit_points = [Point(coord) for coord in fruit_coords]

    # Convert boundary and points to GeoDataFrame
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:4326").to_crs(epsg=2193)
    points_gdf = gpd.GeoDataFrame(geometry=fruit_points, crs="EPSG:4326").to_crs(epsg=2193)

    # Turn boundary into rectangle
    bounding_rectangle = boundary_gdf.geometry[0].minimum_rotated_rectangle

    # Rotate and translate the rectangle into x-y coordinate system
    final_rectangle, aligned_points_gdf = rotate_and_translate(bounding_rectangle, points_gdf)
    
    # Create GeoDataFrame for rectangle
    final_rectangle_gdf = gpd.GeoDataFrame({'geometry': [final_rectangle], 'type': ['rectangle']}, 
                                           crs=boundary_gdf.crs)

    # Add metadata to points
    aligned_points_gdf['type'] = 'point'

    # Combine rectangle and points
    combined_gdf = pd.concat([final_rectangle_gdf, aligned_points_gdf], ignore_index=True)
    return combined_gdf


# --- Step 6: Save Processed Data to Parquet ---
def save_to_parquet(output_path, combined_gdf, kml_filename):
    """
    Save the extracted rectangle and points to parquet files.
    """
    parquet_path = os.path.join(output_path, f"{os.path.splitext(kml_filename)[0]}.parquet")
    combined_gdf.to_parquet(parquet_path, index=False)


# --- Step 7: Process All KML Files in a Zip Archive ---
def process_zip_file(zip_file, output_path):
    """
    Extract data from all KML files in a zip archive and save as parquet files.
    """    
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    # Open the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get list of KML files
        kml_files = [f for f in zip_ref.namelist() if f.endswith('.kml')]
        
        if not kml_files:
            raise ValueError("No KML files found in the zip archive.")
        
        for kml_file in kml_files:
            # Extract the KML file to a temporary path
            with zip_ref.open(kml_file) as file:
                temp_path = os.path.join(output_path, "temp.kml")
                with open(temp_path, 'wb') as temp_file:
                    temp_file.write(file.read())

                try:
                    # Process the extracted KML file
                    combined_gdf = process_kml_file(temp_path)
                    save_to_parquet(output_path, combined_gdf, os.path.basename(kml_file))
                    print(f"{kml_file} has been processed!")
                except Exception as e:
                    print(f"Error processing {kml_file}: {e}")
                finally:
                    os.remove(temp_path)  # Clean up the temporary file


if __name__ == "__main__":
    zip_file = 'fruit_location_data.zip'
    output_path = '' # save to your local directory
    process_zip_file(zip_file, output_path)