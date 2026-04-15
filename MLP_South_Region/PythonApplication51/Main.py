# Main.py
import os
import pyvista as pv
import pandas as pd


import MakeDataset
import TrainModel
import DrawLog
import ModelPred
import Visualizer
import Data_utils
import MakeDataset 
import CONFIG
import PATH

def ConfigInput():
    flag_Train, flag_Draw = 'A', 'A'
    flag_Pred, flag_Vis = 'A', 'A'

    while not (flag_Train in ['Y', 'N']):
        flag_Train = input(r"Whether to train the model？(Y/N): ").strip().upper()
    while not (flag_Draw in ['Y', 'N']):
        flag_Draw = input(r"Output training progress charts？(Y/N): ").strip().upper()
    while not (flag_Pred in ['Y', 'N']):
        flag_Pred = input(r"Perform 3D Mesh Prediction？(Y/N): ").strip().upper()
    while not (flag_Vis in ['Y', 'N']):
        flag_Vis = input(r"Perform 3D Visualization？(Y/N): ").strip().upper()
    return flag_Train, flag_Draw, flag_Pred, flag_Vis

def main():

    flag_Train, flag_Draw, flag_Pred, flag_Vis = ConfigInput()

    if not os.path.exists(PATH.PRED_ROOT_DIR):
        os.makedirs(PATH.PRED_ROOT_DIR)

    borehole_data = pd.read_csv(PATH.CSV_DATA_PATH)
    xy_coords = borehole_data.drop_duplicates(subset=['Name'])[['X', 'Y']].values
    boundary = Data_utils.create_boundary_polygon(xy_coords)
   
    dem_data, dem_tree = Data_utils.read_dem(PATH.DEM_PATH, boundary, CONFIG.DEM_MAX_POINTS)

    processed_borehole_data = Data_utils.calculate_z_coordinates(borehole_data, dem_tree, dem_data, 'Name')
    
    if flag_Train == 'Y':
        train_loader, val_loader, test_loader, scalers = MakeDataset.make_dataset(processed_borehole_data)
        TrainModel.train(train_loader, val_loader, test_loader, scalers)

    if flag_Draw == 'Y':
        DrawLog.plt_drawing()

    predicted_grid = None
    if flag_Pred == 'Y':
        predicted_grid = ModelPred.create_and_predict_3d_grid(
            processed_borehole_data, dem_data, dem_tree, boundary
        )
    elif flag_Vis == 'Y' and os.path.exists(PATH.PRED_GRID_OUTPUT_PATH):
        print(f"Loading an existing prediction model:{PATH.PRED_GRID_OUTPUT_PATH}")
        predicted_grid = pv.read(PATH.PRED_GRID_OUTPUT_PATH)

    if flag_Vis == 'Y':
        import numpy as np
        print("\n===== Statistical Analysis of Model Uncertainty =====")
        
    
        uncertainty_values = predicted_grid.cell_data['Uncertainty']
        geological_classes = predicted_grid.cell_data['Geological_Class']
        
        valid_mask = (geological_classes != -1) & (uncertainty_values != -1.0)
        valid_uncertainty_values = uncertainty_values[valid_mask]
        
        total_valid_cells = len(valid_uncertainty_values)
        if total_valid_cells > 0:
            min_unc = np.min(valid_uncertainty_values)
            max_unc = np.max(valid_uncertainty_values)
            print(f"Effective Region Uncertainty Range: [ {min_unc:.4f}, {max_unc:.4f} ]")

            low_uncertainty_mask = (valid_uncertainty_values >= -0.1) & (valid_uncertainty_values <= 0.01)
            low_uncertainty_count = np.sum(low_uncertainty_mask)
            low_percentage = (low_uncertainty_count / total_valid_cells) * 100

            high_uncertainty_mask = (valid_uncertainty_values >= 0.3) & (valid_uncertainty_values <= 1.0)
            high_uncertainty_count = np.sum(high_uncertainty_mask)
            high_percentage = (high_uncertainty_count / total_valid_cells) * 100


            print(f"Total Number of Effective Mesh Elements in Model: {total_valid_cells}")
            print(f"Number of grid cells with uncertainty in the range [0.00, 0.01] (High Confidence):{low_uncertainty_count} ({low_percentage:.2f}%)")
            print(f"Number of grid cells with uncertainty in the range [0.30, 1.00] (low confidence): {high_uncertainty_count} ({high_percentage:.2f}%)")
        else:
            print("No valid model mesh found for statistics.")

        Visualizer.run_visualization_suite(
            predicted_grid=predicted_grid,
            processed_borehole_data=processed_borehole_data,
            dem_data=dem_data,
            boundary=boundary,
            dem_tree=dem_tree  
        )
        
    
    print("\n All requested operations have been completed.")

if __name__ == "__main__":
    main()