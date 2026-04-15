# Visualizer.py
import pyvista as pv
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import warnings
from scipy.interpolate import griddata

import Data_utils 
import CONFIG
import PATH

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def run_visualization_suite(predicted_grid, processed_borehole_data, dem_data, boundary, dem_tree):

    if predicted_grid is None:
        print("Error: No prediction grid available for visualization. Please run the prediction first.")
        return

    # 1. 
    print("\n===== Preparing Visualization Assets =====")
    unique_labels = sorted([c for c in np.unique(processed_borehole_data['Cls']) if c != -1])
    cmap_dict = {label: mcolors.to_rgba(CONFIG.VIS_COLOR_MAP.get(label, 'grey')) for label in unique_labels}

    # 2. 
    print("\n--> Step 1: Displaying 3D Geological Model with Smooth DEM Clip")
    _plot_3d_geology_model(predicted_grid, dem_data, cmap_dict, unique_labels)

    # 3.
    print("\n--> Step 2: Displaying 3D Uncertainty Model with Smooth DEM Clip")
    UNCERTAINTY_THRESHOLD = 1e-11
    _plot_3d_uncertainty_as_voxels(
        grid=predicted_grid,
        dem_data=dem_data, 
        unique_labels=unique_labels,
        cmap_dict=cmap_dict,
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        show_geology_context=True
    )

    # 4.
    print("\n--> Step 3: Displaying Multiple Geological Sections")
    _plot_multiple_geological_sections(
        predicted_grid=predicted_grid,
        processed_borehole_data=processed_borehole_data,
        dem_tree=dem_tree,
        dem_data=dem_data,
        cmap_dict=cmap_dict,
        unique_labels=unique_labels,
        boundary=boundary
    )



def _plot_3d_geology_model(grid, dem_data, cmap_dict, unique_labels):
    plotter = pv.Plotter(window_size=[1200, 900],off_screen=False)
    plotter.set_scale(zscale=5)
    plotter.remove_all_lights()

    light_position = (1, -1, 1) # X=0, Y=1, Z=1
    light = pv.Light(position=light_position, light_type='scene light', intensity=0.85)
    plotter.add_light(light)

    dem_surface = None
    if dem_data is not None and len(dem_data) > 2:
        dem_surface = pv.PolyData(dem_data).delaunay_2d()
        plotter.add_mesh(dem_surface, color='tan', opacity=0, style='surface', label='')
    else:
        warnings.warn("DEM data is not available for smooth clipping. Displaying full model.")
    
    for label in unique_labels:
        strata_volume = grid.threshold([label - 0.5, label + 0.5], scalars='Geological_Class')
        if strata_volume.n_cells > 0:
            clipped_strata = strata_volume
            if dem_surface:
                clipped_strata = strata_volume.clip_surface(dem_surface, invert=True)
            
            if clipped_strata.n_cells > 0:
                plotter.add_mesh(clipped_strata, color=cmap_dict[label], opacity=1, name=f'Class_{label}', label=f'类别 {label}',ambient=0.2)

    #plotter.add_legend(bcolor=None, border=True, size=(0.15, 0.25))
    
    plotter.add_title('', font_size=16)
    #plotter.add_axes(interactive=True)
    #plotter.show_grid(xtitle='X ', ytitle='Y ', ztitle='')
    #################################################################################
    plotter.view_isometric()

    focus_point = plotter.camera.focal_point

    light_direction = np.array([0.75, -1.7, 0.7])

    distance = grid.length * 2.0
    view_direction_normalized = light_direction / np.linalg.norm(light_direction)
    camera_pos = focus_point + (view_direction_normalized * distance)
    view_up = plotter.camera.view_up
    plotter.camera_position = [camera_pos, focus_point, view_up]
    #################################################################################
    #plotter.screenshot('Kan_Factor_N.png', transparent_background=True)
    plotter.show()


def _plot_3d_uncertainty_as_voxels(grid, dem_data, unique_labels, cmap_dict, uncertainty_threshold, show_geology_context=True):
    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.set_background('white')
    plotter.set_scale(zscale=5)

    dem_surface = None
    if dem_data is not None and len(dem_data) > 2:
        dem_surface = pv.PolyData(dem_data).delaunay_2d()
    else:
        warnings.warn("DEM data not available for smooth clipping in uncertainty plot.")

    high_uncertainty_voxels = grid.threshold(value=uncertainty_threshold, scalars='Uncertainty', invert=False)

    if high_uncertainty_voxels.n_cells > 0:
        clipped_uncertainty_voxels = high_uncertainty_voxels

        if dem_surface:
            clipped_uncertainty_voxels = high_uncertainty_voxels.clip_surface(dem_surface, invert=True)
        
        if clipped_uncertainty_voxels.n_cells > 0:
            print(f"找到 {clipped_uncertainty_voxels.n_cells} 个高不确定性体素 (DEM下方)。")
            plotter.add_mesh(
                clipped_uncertainty_voxels, scalars='Uncertainty', cmap='coolwarm',
                scalar_bar_args={'title': '归一化不确定性'}, opacity=1.0,
                clim=[uncertainty_threshold, 1.0]
            )
        else:
            print("所有高不确定性区域均位于DEM地表之上，裁剪后无显示内容。")
    else:
        warnings.warn(f"在阈值 {uncertainty_threshold:.2f} 下，没有找到高不确定性区域。")

    if show_geology_context:
        for label in unique_labels:
            strata_volume = grid.threshold([label - 0.5, label + 0.5], scalars='Geological_Class')
            if strata_volume.n_cells > 0:
                clipped_strata = strata_volume

                if dem_surface:
                    clipped_strata = strata_volume.clip_surface(dem_surface, invert=True)
                
                if clipped_strata.n_cells > 0:
                    plotter.add_mesh(clipped_strata, color=cmap_dict[label], opacity=0.1, style='surface')
        
##################################################################################
    plotter.view_isometric()
    focus_point = plotter.camera.focal_point

    light_direction = np.array([0.75, -1.7, 0.7])

    distance = grid.length * 2.0
    view_direction_normalized = light_direction / np.linalg.norm(light_direction)
    camera_pos = focus_point + (view_direction_normalized * distance)
    view_up = plotter.camera.view_up
    plotter.camera_position = [camera_pos, focus_point, view_up]
###############################################################################
    #plotter.add_mesh(grid.outline(), color='black')
    #plotter.add_title(f'高不确定性区域 (熵 > {uncertainty_threshold:.2f}, DEM裁剪)', font_size=16)
    #plotter.show_grid(xtitle='X ', ytitle='Y ', ztitle='高程 ')

    plotter.show()

def _dim_color(rgba_color, factor):

    if factor >= 1.0:
        return rgba_color

    rgb = np.array(rgba_color[:3]) * factor

    dimmed_rgb = np.clip(rgb, 0, 1)
    
    return (*dimmed_rgb, rgba_color[3])



def _plot_multiple_geological_sections(predicted_grid, processed_borehole_data, dem_tree, dem_data, cmap_dict, unique_labels, boundary):

    num_sections = len(CONFIG.SECTIONS)
    if num_sections == 0:
        print("CONFIG")
        return

    dim_factor = getattr(CONFIG, 'COLOR_DIM_FACTOR', 1.0)
    print(f"Applying color dimming factor: {dim_factor}")
    dimmed_cmap_dict = {label: _dim_color(color, dim_factor) for label, color in cmap_dict.items()}

    colors = [dimmed_cmap_dict.get(label, (0,0,0,0)) for label in unique_labels]

    custom_cmap = ListedColormap(colors + [(0,0,0,0)])
    label_map = {label: i for i, label in enumerate(unique_labels)}
    label_map[-1] = len(unique_labels)
    
    borehole_intervals = Data_utils.process_borehole_intervals(processed_borehole_data)

  
    minz, maxz = predicted_grid.bounds[4], predicted_grid.bounds[5] 

    TARGET_WIDTH_TO_HEIGHT_RATIO = 6.0 

    for i, section_info in enumerate(CONFIG.SECTIONS):
        print(f"\n--- Generating section: {section_info['name']} ---")
        
  
        figure_height_inches = 8 
        figure_width_inches = figure_height_inches * TARGET_WIDTH_TO_HEIGHT_RATIO 

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figure_width_inches, figure_height_inches), squeeze=False)
        ax = ax.flatten()[0] 
        
        _draw_single_section_on_ax(
            ax=ax,
            section_info=section_info,
            predicted_grid=predicted_grid,
            borehole_intervals=borehole_intervals,
            dem_tree=dem_tree,
            dem_data=dem_data,
            cmap_dict=dimmed_cmap_dict,
            custom_cmap=custom_cmap,
            label_map=label_map,
            boundary=boundary
        )

        ax.set_aspect('auto', adjustable='box') 

        fig.tight_layout()

        section_filename = f"{section_info['name']}.png"

        plt.savefig(section_filename, dpi=300, bbox_inches='tight')
        print(f"Section '{section_info['name']}' saved to {section_filename}")
        plt.close(fig) 

def _draw_single_section_on_ax(ax, section_info, predicted_grid, borehole_intervals, dem_tree, dem_data, cmap_dict, custom_cmap, label_map, boundary):

    minx, miny, maxx, maxy = boundary.bounds
    _, _, _, _, minz, maxz = predicted_grid.bounds
    
    section_start_xy = section_info['start']
    section_end_xy = section_info['end']

    valid_start = minx <= section_start_xy[0] <= maxx and miny <= section_start_xy[1] <= maxy
    valid_end = minx <= section_end_xy[0] <= maxx and miny <= section_end_xy[1] <= maxy
    if not (valid_start and valid_end):
        warnings.warn(f"剖面 '{section_info['name']}' 的点部分或全部超出数据范围，结果可能不准确。")

    start_point = [section_start_xy[0], section_start_xy[1], (minz + maxz) / 2]
    end_point = [section_end_xy[0], section_end_xy[1], (minz + maxz) / 2]

    direction_vec = np.array(end_point) - np.array(start_point)
    normal_vector = np.array([direction_vec[1], -direction_vec[0], 0])
    section_slice = predicted_grid.slice(normal=normal_vector, origin=start_point)

    section_length = np.linalg.norm(np.array(section_end_xy) - np.array(section_start_xy))
    dist_coords = np.linspace(0, section_length, CONFIG.SECTION_RESOLUTION_2D)
    elev_coords = np.linspace(minz, maxz, CONFIG.SECTION_RESOLUTION_2D)
    dist_grid, elev_grid = np.meshgrid(dist_coords, elev_coords)
    
    start_vec_xy = np.array(section_start_xy)
    direction_vec_xy_unit = (np.array(section_end_xy) - start_vec_xy) / section_length
    target_xy_coords = start_vec_xy + dist_grid.ravel()[:, np.newaxis] * direction_vec_xy_unit
    target_xyz_coords = np.hstack([target_xy_coords, elev_grid.ravel()[:, np.newaxis]])
    
    if len(section_slice.points) == 0:
        ax.text(0.5, 0.5, '切片操作未产生数据点', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        print(f"警告: 剖面 '{section_info['name']}' 的切片操作结果为空。")
        return

    source_points = section_slice.cell_centers().points
    source_values = section_slice.cell_data['Geological_Class']
    
    interpolated_labels = griddata(source_points, source_values, target_xyz_coords, method='nearest')
    class_grid = interpolated_labels.reshape(dist_grid.shape)

    color_indices = np.vectorize(lambda x: label_map.get(round(x), len(label_map) - 1))(class_grid)
    ax.imshow(color_indices, extent=[0, section_length, minz, maxz], origin='lower', aspect='auto', cmap=custom_cmap, interpolation='none', zorder=1)

    _plot_boreholes_on_section(ax, borehole_intervals, section_start_xy, section_length, cmap_dict, direction_vec_xy_unit)
    
    if dem_tree and dem_data is not None:
        section_xy_samples = start_vec_xy + dist_coords[:, np.newaxis] * direction_vec_xy_unit
        _, dem_indices = dem_tree.query(section_xy_samples)
        dem_z_values = dem_data[dem_indices, 2]
        
        ax.plot(dist_coords, dem_z_values, color='black', linewidth=1, zorder=5, label='DEM Surface')
        ax.fill_between(dist_coords, dem_z_values, maxz, color='white', alpha=1.0, zorder=4)

    ax.set_title('', fontsize=18)
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel('', fontsize=16)
    
    borehole_width_meters = section_length * 0.01
    ax.set_xlim(-borehole_width_meters, section_length + borehole_width_meters)
    ax.set_ylim(minz, maxz)

    #ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.set_yticks(np.arange(900, 1500 + 1, 200))
    ax.set_xticks(np.arange(0, section_length + 1, 2000))
    ax.tick_params(axis='x', labelsize=30) 
    ax.tick_params(axis='y', labelsize=30) 

# --- _plot_boreholes_on_section  _create_cylindrical_gradient  ---
def _plot_boreholes_on_section(ax, borehole_intervals, section_start_xy, section_length, cmap_dict, direction_vec_xy_unit):

    start_vec = np.array(section_start_xy)
    dir_vec_proj = direction_vec_xy_unit 
    normal_vec_proj = np.array([-dir_vec_proj[1], dir_vec_proj[0]])
    
    projected_count = 0
    borehole_width_meters = section_length * 0.01
    filtered_borehole_logs = {an_id: intervals for an_id, intervals in borehole_intervals.items() if "SAMP" not in str(an_id)}

    for an_id, intervals in filtered_borehole_logs.items():
        if not intervals: continue
        
        borehole_pos = np.array([intervals[0]['x'], intervals[0]['y']])
        rel_vec = borehole_pos - start_vec
        distance_to_line = abs(np.dot(rel_vec, normal_vec_proj))
        
        if distance_to_line <= CONFIG.BOREHOLE_PROJECTION_THRESHOLD:
            projected_dist = np.dot(rel_vec, dir_vec_proj)
            if 0 <= projected_dist <= section_length:
                projected_count += 1
                for interval in intervals:
                    base_color = cmap_dict.get(interval['cls'], 'gray')
                    gradient_image = _create_cylindrical_gradient(base_color)
                    extent = [
                        projected_dist - borehole_width_meters / 2,
                        projected_dist + borehole_width_meters / 2,
                        interval['bottom_z'],
                        interval['top_z']
                    ]
                    ax.imshow(gradient_image, extent=extent, aspect='auto', origin='lower', zorder=6)

    print(f"The {projected_count}  borehole column is projected onto the cross-section.")

def _create_cylindrical_gradient(color, steps=20, dim_factor=0.7):
    rgba_bright = mcolors.to_rgba(color)
    rgb_dark = tuple(c * dim_factor for c in rgba_bright[:3])
    rgba_dark = (*rgb_dark, rgba_bright[3])
    profile = np.sin(np.linspace(0, np.pi, steps))
    gradient = np.array([(1 - p) * np.array(rgba_dark) + p * np.array(rgba_bright) for p in profile])
    return gradient.reshape(1, steps, 4)