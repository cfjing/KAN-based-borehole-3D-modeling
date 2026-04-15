# Data_utils.py
import pandas as pd
import numpy as np
from osgeo import gdal
from scipy.spatial import ConvexHull, KDTree
from shapely.geometry import Point, Polygon
from typing import Tuple, Dict, List, Optional

def read_dem(dem_path, boundary_polygon, max_points) -> Optional[Tuple[np.ndarray, KDTree]]:
    print("Loading DEM data...")
    try:
        dem_ds = gdal.Open(dem_path)
        if dem_ds is None: raise IOError(f"Cannot open DEM file: {dem_path}")
        gt, band = dem_ds.GetGeoTransform(), dem_ds.GetRasterBand(1)
        arr, nodata = band.ReadAsArray(), band.GetNoDataValue()
        dem_ds = None
        minx, miny, maxx, maxy = boundary_polygon.bounds
        cols, rows = arr.shape[1], arr.shape[0]
        step = max(1, int(np.sqrt(cols * rows / max_points)))
        points = []
        for r in range(0, rows, step):
            for c in range(0, cols, step):
                val = arr[r, c]
                if val != nodata:
                    x, y = gt[0] + (c + 0.5) * gt[1], gt[3] + (r + 0.5) * gt[5]
                    if boundary_polygon.contains(Point(x, y)):
                        points.append((x, y, val))
        if len(points) < 3: return None, None
        dem_data = np.array(points)
        dem_tree = KDTree(dem_data[:, :2])
        print(f"Loaded and sampled {len(dem_data)} DEM points within boundary.")
        return dem_data, dem_tree
    except Exception as e:
        print(f"Error loading DEM: {e}")
        return None, None

def create_boundary_polygon(xy_coords):
    print("Creating convex hull boundary...")
    hull = ConvexHull(xy_coords)
    boundary = Polygon(xy_coords[hull.vertices])
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    buffer_distance = 500  # Buffer Distance
    buffered_boundary = boundary.buffer(buffer_distance)
    print(f"Applied a {buffer_distance}-meter buffer to the convex hull.")
    return buffered_boundary


def calculate_z_coordinates(df: pd.DataFrame, dem_tree: KDTree, dem_data: np.ndarray, id_column: str = 'Name') -> pd.DataFrame:
    print("Calculating Z coordinates for borehole data...")
    if dem_tree is None or dem_data is None:
        raise ValueError("DEM data (tree or data array) is not available.")
    
    unique_boreholes = df.drop_duplicates(subset=[id_column])
    coords = unique_boreholes[['X', 'Y']].values
    
    _, indices = dem_tree.query(coords)
    dem_z_values = dem_data[indices, 2]
    
    id_to_dem_z = dict(zip(unique_boreholes[id_column], dem_z_values))
    df['DEM_Z'] = df[id_column].map(id_to_dem_z)
    df['Z'] = df['DEM_Z'] - df['Depth']
    
    print(f"Successfully calculated Z coordinates. Min Z: {df['Z'].min():.2f}, Max Z: {df['Z'].max():.2f}")
    return df

def process_borehole_intervals(df: pd.DataFrame, id_column: str = 'Name') -> Dict[str, List[Dict]]:
    """
    Processes borehole data into a dictionary of intervals for each borehole.
    Assumes 'Z' and 'DEM_Z' columns already exist.
    """
    print("Processing borehole data into plotting intervals...")
    borehole_intervals = {}
    for an_id, group in df.groupby(id_column):
        group = group.sort_values('Depth')
        intervals = []
        last_depth = 0.0

        if 'DEM_Z' not in group.columns:
            raise KeyError("'DEM_Z' column is missing. Please run `calculate_z_coordinates` first.")
            
        dem_z = group['DEM_Z'].iloc[0]
        
        for _, row in group.iterrows():
            # A valid interval is created only when the current depth is greater than the previous depth.
            if row['Depth'] > last_depth:
                interval_data = {
                    'top_z': dem_z - last_depth,
                    'bottom_z': dem_z - row['Depth'], 
                    'cls': row['Cls'],
                    'x': row['X'],
                    'y': row['Y']
                }
                intervals.append(interval_data)
                last_depth = row['Depth']
            # If row['Depth'] <= last_depth (including the first row, where the depth is 0), ignore it.
            elif row['Depth'] == last_depth and last_depth == 0.0:
                 # Special Case: The first line is Depth=0. Do nothing;
                    # let last_depth remain at 0 until the first line with Depth > 0 is encountered.
                 pass
            else:
                 # If duplicate depth values ​​are encountered (e.g., [10, 10, 20]), update last_depth
                    # to ensure the next interval starts from the correct position.
                 last_depth = row['Depth'] 

        borehole_intervals[an_id] = intervals
        
    print(f"Processed stratification information for {len(borehole_intervals)} boreholes.")
    return borehole_intervals