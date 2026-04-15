# K3D：KAN-based-borehole-3D-modeling
neural network model with KAN for sparse borehole 3D interpolation modeling

This software code is the official implementation for the article: **"A KAN-Based Dual-Branch Model for 3D Geological Modeling from Sparse Borehole Data Incorporating Geo-Environmental Constraints"**.Contains the workflows of the KAN model and the corresponding MLP model in the southern and northern rigions.

## Description

This repository provides a complete computational framework for 3D geological modeling using a **Dual-Branch Kolmogorov-Arnold Network (KAN)**. The model is specifically engineered to address the challenges of **sparse borehole data** by integrating macro-environmental features as soft constraints within a decoupled neural architecture. 

## Repository Structure

The project is modularized to ensure clarity and reproducibility:

* **`Main.py`**: The central entry point with an interactive command-line interface.
* **`MakeDataset.py` & `Data_utils.py`**: Handles data cleaning, coordinate transformation, and tensor dataset preparation.
* **`TrainModel.py`**: Contains the core logic for training the Dual-Branch KAN and validating model performance.
* **`ModelPred.py`**: Generates 3D grid predictions and calculates spatial uncertainty for the geological bodies.
* **`Visualizer.py`**: A high-performance 3D visualization suite powered by `PyVista` for rendering boreholes, surfaces, and predicted volumes.
* **`CONFIG.py` & `PATH.py`**: Global settings for model hyperparameters and directory management.

## Installation

* Download the repository as a zip file

* Unzip the folder to a desired location on your machine

* Create a Python environment to run this code.

* Ensure you have **Python 3.8 or higher** installed. You can install all necessary dependencies using the following command:

```bash
pip install torch pyvista pandas numpy scikit-learn matplotlib
```

## Usage

To execute the modeling workflow, run the main script from your terminal:

```bash
python Main.py
```

The program will guide you through an interactive menu:

1.  **Train the model? (Y/N)**: Start the KAN training process.
2.  **Output training progress charts? (Y/N)**: Generate loss/accuracy curves.
3.  **Perform 3D Mesh Prediction? (Y/N)**: Calculate the geological grid and uncertainty.
4.  **Perform 3D Visualization? (Y/N)**: Launch the interactive 3D viewer.

* Upon the completion of the training phase, the console prints various statistical performance metrics. Simultaneously, a window automatically displays the ROCand PR curves for a diagnostic evaluation of the model.

* After closing the diagnostic plot window, the system launches the 3D visualization suite. This displays the 3D geological model and the 3D uncertainty model using PyVista. Detailed statistical results regarding the model's uncertainty and grid confidence levels are printed to the console during this stage.

* Upon closing the 3D visualization window, the software automatically generates and saves the cross-section plots to the project’s root directory for further geological interpretation and documentation.

## Data Availability

The repository includes a **preprocessed dataset** in the `data/` directory to demonstrate the model's functionality:

* `borehole.csv`: A desensitized subset of borehole records.
* `dem.tif`: Pre-processed Digital Elevation Model data used for topographic constraints.

*Note: The complete original dataset is available upon request from the authors.*
