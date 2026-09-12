"""
This module contains functions to convert navigation data to shapefiles. 
It provides utilities for reading navigation data, processing it, 
and exporting it in a format compatible with GIS applications. 
The module supports various navigation data formats and ensures 
that the output shapefiles maintain the integrity of the original data.
"""

def extract_nav_file_paths(nav_data_directory):
    """
    Extracts the file paths of navigation data files from the specified directory.

    Parameters:
    nav_data_directory (str): The directory containing navigation data files.

    Returns:
    list: A list of file paths to the navigation data files.
    """
    import os

    nav_file_paths = []
    for root, dirs, files in os.walk(nav_data_directory):
        for file in files:
            if file.endswith('.txt'):  # Assuming .txt is the extension for navigation data files
                nav_file_paths.append(os.path.join(root, file))
    
    return nav_file_paths


def extract_camera_locations(file_path: str) -> dict[int, tuple[float, float]]:
    """
    Extract camera longitude and latitude for each frame.

    Parameters
    ----------
    file_path : str
        Path to the metadata text file.

    Returns
    -------
    dict[int, tuple[float, float]]
        Dictionary in the form:
        {
            1: (-161.90969083, 59.72783492),
            2: (-161.90968848, 59.72783228),
            ...
        }
    """
    frame_locations = {}
    current_frame = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Blank line means we've reached the end of a frame
            if not line:
                if current_frame:
                    frame_id = int(current_frame["FRAME_ID"])
                    cam_lon = float(current_frame["CAM_LOC_LON"])
                    cam_lat = float(current_frame["CAM_LOC_LAT"])
                    frame_locations[frame_id] = (cam_lon, cam_lat)
                    current_frame = {}
                continue

            key, value = line.split("\t", 1)
            current_frame[key] = value

    # Handle the final frame if the file doesn't end with a blank line
    if current_frame:
        frame_id = int(current_frame["FRAME_ID"])
        cam_lon = float(current_frame["CAM_LOC_LON"])
        cam_lat = float(current_frame["CAM_LOC_LAT"])
        frame_locations[frame_id] = (cam_lon, cam_lat)

    return frame_locations




def camera_locations_to_shapefile(camera_locations, output_folder, shapefile_name):
    import arcpy
    import os
    """
    Convert a dictionary of camera locations into a point shapefile.

    Parameters
    ----------
    camera_locations : dict
        Dictionary of the form:
        {frame_id: (lon, lat)}

    output_folder : str
        Folder where the shapefile will be created.

    shapefile_name : str
        Name of the shapefile (e.g., "camera_locations.shp")
    """

    output_shp = os.path.join(output_folder, shapefile_name)

    # WGS84 coordinate system
    spatial_ref = arcpy.SpatialReference(4326)

    # Create the shapefile
    arcpy.management.CreateFeatureclass(
        out_path=output_folder,
        out_name=shapefile_name,
        geometry_type="POINT",
        spatial_reference=spatial_ref,
    )

    # Add Frame ID field
    arcpy.management.AddField(output_shp, "Frame_ID", "LONG")

    # Insert the points
    with arcpy.da.InsertCursor(output_shp, ["SHAPE@XY", "Frame_ID"]) as cursor:
        for frame_id, (lon, lat) in camera_locations.items():
            cursor.insertRow(((lon, lat), frame_id))

    print(f"Shapefile created: {output_shp}")

    return output_shp


def main():
    paths = extract_nav_file_paths("E:/Quinhagak_2026_HSI_Data/Floatplane_Data/HSI")
    print(paths[0])
    
    locations = []
    for path in paths:
        for location in extract_camera_locations(path):
            locations.append(location)
    camera_locations_to_shapefile(locations, "E:/Quinhagak_2026_HSI_Data/Floatplane_Data/HSI", "floatplane locs")

if __name__ == "__main_":
    main()
