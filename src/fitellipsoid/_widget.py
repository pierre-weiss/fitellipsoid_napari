#%% Library import
import napari
from napari.utils.notifications import show_info
from magicgui.widgets import Container, PushButton, FileEdit, FloatSpinBox, Label
from qtpy.QtWidgets import QFileDialog
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from .math_func import *

#%% Saving the results
def save_objects_to_pickle(objects, filepath="./tmp.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump(objects, f)

def load_objects_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_ellipsoids_to_excel(ellipsoid_list, filename="ellipsoids.xlsx"):    
    # Create a list to store the data for all ellipsoids
    data = []
    
    ind = 0
    for i, ellipsoid in enumerate(ellipsoid_list):
        # Check if eigvecs is a valid 3x3 matrix
        eigvecs = ellipsoid.eigvecs
        if len(eigvecs) != 0 :
            ind = ind + 1
            # Extract relevant information from each ellipsoid
            eigvecs = ellipsoid.eigvecs  # Assuming eigvecs is a list of 3D vectors
            center = ellipsoid.center    # Assuming center is a 3D vector
            axes_length = ellipsoid.axes_length  # Assuming axes_length is a list/array of 3 values
            
            # For each eigenvector, store each component separately
            eigvec1_x, eigvec1_y, eigvec1_z = eigvecs[0]  # If eigvecs[0] is a 3D vector
            eigvec2_x, eigvec2_y, eigvec2_z = eigvecs[1]  # If eigvecs[1] is a 3D vector
            eigvec3_x, eigvec3_y, eigvec3_z = eigvecs[2]  # If eigvecs[2] is a 3D vector
            
            # Create a row for the current ellipsoid
            row = {
                'Ellipsoid Index': i,
                'EigVec1 X': eigvec1_x,
                'EigVec1 Y': eigvec1_y,
                'EigVec1 Z': eigvec1_z,
                'EigVec2 X': eigvec2_x,
                'EigVec2 Y': eigvec2_y,
                'EigVec2 Z': eigvec2_z,
                'EigVec3 X': eigvec3_x,
                'EigVec3 Y': eigvec3_y,
                'EigVec3 Z': eigvec3_z,
                'Center X': center[0],
                'Center Y': center[1],
                'Center Z': center[2],
                'Axis Length 1': axes_length[0],
                'Axis Length 2': axes_length[1],
                'Axis Length 3': axes_length[2]
            }
            
            # Append the row to the data list
            data.append(row)
            print(f"Saved ellipsoid {ind}")
    
    # Convert the data list into a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to an Excel file
    filename = Path(filename)
    if filename.suffix != '.csv':
        print("Warning: Filename doesn't have '.csv' extension. Adding it.")
        # Create a new path with the correct extension
        filename = filename.with_suffix('.csv')

    df.to_csv(filename, index=False)
    print(f"Ellipsoids saved to {filename}")

def display_ellipsoid_mesh(viewer, samples, color=[0, 0, 1, 0.7], name='Ellipsoid Mesh'):
    vertices, faces = create_ellipsoid_mesh(samples)
    
    # Add surface mesh to viewer
    mesh = viewer.add_surface(
        (vertices, faces),
        colormap='blues',
        opacity=0.7,
        name=name,
        shading='flat'
    )
    return mesh

#%% 
class FitEllipsoidWidget(Container):
    def __init__(self, napari_viewer):
        print("fitellipsoid._widget loaded")

        self.default_point_size = 2
        self.points_layer_list = []
        self.ellipsoid_list = []
        self.point_layer_list = []

        self.viewer = napari_viewer
        # self.image = tifffile.imread("./data/img_test.tif")   
        # self.viewer.add_image(self.image,  blending='translucent_no_depth', name="Image")
        points_layer = self.viewer.add_points([], size = self.default_point_size, face_color='red', ndim=3,  blending='translucent_no_depth')
        self.points_layer_list.append(points_layer)
        self.ellipsoid_list.append(Ellipsoid())
        # Attach both left and right click events
        points_layer.mouse_drag_callbacks.clear()  # Remove old bindings
        points_layer.mouse_drag_callbacks.append(self._on_mouse_press)
        
        # Create your widgets
        self.fit_button = PushButton(label="Fit Ellipsoid")
        self.save_button = PushButton(label="Save Results")

        self.x_input = FloatSpinBox(label="X Scale", value=1.0, min=0.0001, max=100.0)
        self.y_input = FloatSpinBox(label="Y Scale", value=1.0, min=0.0001, max=100.0)
        self.z_input = FloatSpinBox(label="Z Scale", value=1.0, min=0.0001, max=100.0)
        
        # Set up callbacks
        self.fit_button.clicked.connect(self._fit_ellipsoid)
        self.save_button.clicked.connect(self._save_dialog)
        self.x_input.changed.connect(self._update_scale)
        self.y_input.changed.connect(self._update_scale)
        self.z_input.changed.connect(self._update_scale)
        
        # Create container with all widgets
        super().__init__(
            widgets=[
                Label(value="Fit your ellopsoid"),
                self.fit_button,
                Label(value="Save Results"),
                self.save_button,
                Label(value="Scale Settings"),
                self.x_input, 
                self.y_input,
                self.z_input
            ],
            layout="vertical"
        )

    def _save_dialog(self):
        # Get the filename from the save_path widget
        filename, _ = QFileDialog.getSaveFileName(
            parent=None,
            caption="Save Ellipsoids",
            filter="CSV files (*.csv)"
        )
        if filename:
            # Call the save function when the user presses the button
            if self.ellipsoid_list:                    
                save_ellipsoids_to_excel(self.ellipsoid_list, filename)
                print(f"Ellipsoids saved to {filename}")
        else:
            print("No filename provided")   

    def _on_mouse_press(self, layer, event):
            layer_number = self.points_layer_list.index(layer)
            ellipsoid = self.ellipsoid_list[layer_number]

            if event.button == 1:  # Left-click adds a point
                position = self.viewer.cursor.position  # Get cursor position
                # Adds the points to fit to ellipsoid
                if position is None: 
                    return
                
                ellipsoid = self.ellipsoid_list[layer_number]
                ellipsoid.fitting_points.append(position)

                # Update points display
                layer.data = np.array(ellipsoid.fitting_points)
                print(f"Ellipsoid {layer_number} contains {len(ellipsoid.fitting_points)} points. Added ({position[0]},{position[1]})")
                save_objects_to_pickle(self.ellipsoid_list)

            elif event.button == 2:  # Right-click removes the last point if any
                # Right-click 
                if ellipsoid.fitting_points: 
                    # Remove the last point
                    ellipsoid.fitting_points.pop()  
                    # Update points display
                    layer.data = np.array(ellipsoid.fitting_points)  
                    print(f"Ellipsoid {layer_number} contains {len(ellipsoid.fitting_points)} points")

            elif event.button == 3:  # Middle-click
                print("No function implemented on center click")

    def _fit_ellipsoid(self):
        active_layer = self.viewer.layers.selection.active
        if active_layer in self.points_layer_list:
            active_layer_index = self.points_layer_list.index(active_layer)

        ellipsoid = self.ellipsoid_list[active_layer_index]
        if len(ellipsoid.fitting_points) >= 10:  # Need at least 10 points to fit an ellipsoid
            # We fit the clicked points, sample the estimated ellipsoid and add a fitted ellipsoid layer
            ellipsoid.fit_ellipsoid()
            ellipsoid.sample_ellipsoid()
            self.viewer.add_points(ellipsoid.samples, size = self.default_point_size, face_color='blue', blending='translucent_no_depth', name=f'FitEllipsoid {active_layer_index}')
            #display_ellipsoid_mesh(self.viewer, ellipsoid.samples, color=[0, 0, 1, 0.7], name=f'FitEllipsoid {active_layer_index}') # For mesh visualization
            print(f"Fitted Ellipsoid {active_layer_index} now displayed.")

            # We can add a new ellipsoid layer
            new_point_layer = self.viewer.add_points([], size = self.default_point_size, face_color='red', ndim=3,  blending='translucent_no_depth')
            self.points_layer_list.append(new_point_layer)
            self.viewer.layers.selection.active = new_point_layer

            self.ellipsoid_list.append(Ellipsoid())
            new_point_layer.mouse_drag_callbacks.clear()  # Remove old bindings
            new_point_layer.mouse_drag_callbacks.append(self._on_mouse_press)
            self._update_scale()
            print(f"Added a new point layer for your new ellipsoid.")

        else:
            print("Not enough points to fit an ellipsoid.")

    def _update_scale(self, event=None):
        # Get values from inputs
        x_scale = self.x_input.value
        y_scale = self.y_input.value
        z_scale = self.z_input.value

        # Napari uses ZYX order for scale
        new_scale = (z_scale, y_scale, x_scale)

        # Apply scale to all layers
        for layer in self.viewer.layers:
            try:
                layer.scale = new_scale
            except Exception as e:
                print(f"Could not set scale for {layer.name}: {e}")

        print(f"Updated scale to: {new_scale}")
