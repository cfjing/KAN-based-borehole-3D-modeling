# ModelPred.py
import numpy as np
import torch
import torch.nn.functional as F  
import pyvista as pv
from tqdm import tqdm
from matplotlib.path import Path
import pandas as pd
from scipy.spatial import KDTree 

import GetModel
import MakeDataset
import CONFIG
import PATH


@torch.no_grad()
def predict_grid_with_uncertainty(model, scalers, grid_points):

    model.eval()
    device = next(model.parameters()).device

    #  scaler
    pos_scaler = scalers['pos']
    fac_scaler = scalers['fac']
    num_pos_features = len(CONFIG.POS_AXES)

    all_preds = []
    all_uncertainties = []
    pbar = tqdm(range(0, len(grid_points), CONFIG.PRED_BATCH_SIZE), desc="Predicting with Uncertainty")

    num_classes = CONFIG.NUM_CLASSES
    max_entropy = np.log2(num_classes) if num_classes > 1 else 1.0

    for i in pbar:
        batch_points = grid_points[i: i + CONFIG.PRED_BATCH_SIZE]

        batch_pos = batch_points[:, :num_pos_features]
        batch_fac = batch_points[:, num_pos_features:]
        
        # 2. scaler
        batch_pos_scaled = pos_scaler.transform(batch_pos)
        batch_fac_scaled = fac_scaler.transform(batch_fac)
        

        pos_tensor = torch.tensor(batch_pos_scaled, dtype=torch.float32).to(device)
        fac_tensor = torch.tensor(batch_fac_scaled, dtype=torch.float32).to(device)

        logits = model(pos_tensor, fac_tensor)

        probs = F.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        epsilon = 1e-9
        entropy_val = -torch.sum(probs * torch.log2(probs + epsilon), dim=1)
        normalized_entropy = entropy_val / max_entropy
        
        all_preds.append(preds.cpu())
        all_uncertainties.append(normalized_entropy.cpu())

    return torch.cat(all_preds, dim=0).numpy(), torch.cat(all_uncertainties, dim=0).numpy()


def create_and_predict_3d_grid(borehole_data, dem_data, dem_tree, boundary_polygon):

    print("\n===== Starting 3D Grid Prediction (4-Nearest Neighbors Average Interpolation) =====")
    
    model = GetModel.get_model()
    scalers = MakeDataset.load_scalers()
    if model is None or scalers is None:
        print("Model or Scalers not loaded. Aborting prediction.")
        return None

    # --- Step 1: Define the full rectangular grid ---
    minx, miny, maxx, maxy = boundary_polygon.bounds
    minz, maxz = borehole_data['Z'].min(), borehole_data['Z'].max()

    print("Creating 3D grid points for the entire bounding box...")
    grid_res = CONFIG.GRID_RESOLUTION_3D
    x_coords = np.linspace(minx, maxx, grid_res)
    y_coords = np.linspace(miny, maxy, grid_res)
    z_coords = np.linspace(minz, maxz, grid_res)
    
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid_points_xyz = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # --- Step 2: Get factor values using 4-Nearest Neighbors Average ---
    print(f"Loading factors from fishnet file for interpolation: {PATH.FACTOR_DATA_CSV}")
    try:
        factor_df = pd.read_csv(PATH.FACTOR_DATA_CSV)
    except FileNotFoundError:
        print(f"FATAL ERROR: Fishnet factor file not found at {PATH.FACTOR_DATA_CSV}. Aborting.")
        return None

    # 2a. Build KDTree from the dense fishnet XY coordinates
    print("Building KDTree for fast factor lookup...")
    fishnet_xy = factor_df[['X', 'Y']].values
    fishnet_factors = factor_df[CONFIG.FAC_LSIT].values
    factor_kdtree = KDTree(fishnet_xy)

    xx_2d, yy_2d = np.meshgrid(x_coords, y_coords, indexing='ij') 
    xy_plane_points = np.c_[xx_2d.ravel(), yy_2d.ravel()]
    
    num_neighbors = 16
    print(f"Querying KDTree to find and average {num_neighbors} nearest factors for {len(xy_plane_points)} grid XY points...")

    _, nearest_indices = factor_kdtree.query(xy_plane_points, k=num_neighbors)

    neighbor_factors = fishnet_factors[nearest_indices]
    

    looked_up_factors_2d = np.mean(neighbor_factors, axis=1)

    num_z_levels = len(z_coords)
    looked_up_factors = np.repeat(looked_up_factors_2d, repeats=num_z_levels, axis=0)
    
    full_feature_grid_points = np.hstack([grid_points_xyz, looked_up_factors])
    full_feature_grid_points = full_feature_grid_points.astype(np.float32)
    print(f"Successfully created feature grid for {len(full_feature_grid_points)} points using 4-NN average.")

    # --- Step 3: Predict on the ENTIRE rectangular grid ---
    print("Converting Z to Depth and predicting for all points in the bounding box...")
    
    points_for_model = full_feature_grid_points.copy()
    
    all_points_xy = points_for_model[:, :2]
    _, dem_indices_all = dem_tree.query(all_points_xy)
    dem_z_all = dem_data[dem_indices_all, 2]
    
    depth = dem_z_all - points_for_model[:, 2]
    depth[depth < 0] = 0  
    points_for_model[:, 2] = depth

    predicted_labels_flat, uncertainty_flat = predict_grid_with_uncertainty(model, scalers, points_for_model)

    # --- Step 4: Clip the PREDICTED results ---
    print("Clipping the predicted model using the boundary polygon and DEM...")
    
    polygon_path = Path(boundary_polygon.exterior.coords)
    in_polygon_mask = polygon_path.contains_points(grid_points_xyz[:, :2])
    below_dem_mask = grid_points_xyz[:, 2] <= (dem_z_all + 50.0)
    valid_mask = in_polygon_mask & below_dem_mask
    
    predicted_labels_flat[~valid_mask] = -1
    uncertainty_flat[~valid_mask] = 0.0

    print(f"Clipping complete. {np.sum(valid_mask)} points remain within the boundary.")

    # --- Step 5: Create the PyVista grid object ---
    # (This part is unchanged)
    print("Creating PyVista grid object...")
    predicted_labels_3d = predicted_labels_flat.reshape((grid_res, grid_res, grid_res))
    uncertainty_3d = uncertainty_flat.reshape((grid_res, grid_res, grid_res))

    grid = pv.ImageData()
    dims = predicted_labels_3d.shape
    grid.dimensions = (dims[0] + 1, dims[1] + 1, dims[2] + 1)
    grid.spacing = ((maxx - minx) / dims[0], (maxy - miny) / dims[1], (maxz - minz) / dims[2])
    grid.origin = (minx, miny, minz)

    grid.cell_data['Geological_Class'] = predicted_labels_3d.flatten(order='F')
    grid.cell_data['Uncertainty'] = uncertainty_3d.flatten(order='F')

    grid.save(PATH.PRED_GRID_OUTPUT_PATH)
    print(f"Predicted and clipped 3D model saved to {PATH.PRED_GRID_OUTPUT_PATH}")

    return grid