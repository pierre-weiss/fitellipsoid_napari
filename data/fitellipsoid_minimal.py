#%% Library import
import napari
from magicgui import magicgui, magic_factory
import tifffile

import numpy as np
import scipy.linalg as linalg
from scipy.linalg import eig, inv, eigh
from scipy.spatial import ConvexHull

import pandas as pd
from pathlib import Path
import pickle

#%% The heart of the method
def ellipsoid3d_fitting_dr_svd(x, nit=1000):
    # An ellipsoid is parameterized as:
    # a1 x^2 + a2 y^2 + a3 z^2 + a4 xy + a5 xz + a6 yz + a7 x + a8 y + a9 z + a10 = 0
    # Vector q = (a11,a22,a33,sqrt(2)a12,sqrt(2)a13,sqrt(2)a23,b1,b2,b3,c)
    n = x.shape[1]
    xx = x
    
    # First find the SVD of x and change coordinates
    t = np.mean(x, axis=1)
    xb = x - t[:, None]
    
    S, U = linalg.eigh(xb @ xb.T)
    sp = np.maximum(S, 1e-15 * np.ones(3))
    P = np.diag(sp ** (-0.5)) @ U.T
    x = P @ xb
    
    D = np.zeros((10, n))
    D[0, :] = x[0, :] ** 2
    D[1, :] = x[1, :] ** 2
    D[2, :] = x[2, :] ** 2
    D[3, :] = np.sqrt(2) * x[0, :] * x[1, :]
    D[4, :] = np.sqrt(2) * x[0, :] * x[2, :]
    D[5, :] = np.sqrt(2) * x[1, :] * x[2, :]
    D[6, :] = x[0, :]
    D[7, :] = x[1, :]
    D[8, :] = x[2, :]
    D[9, :] = 1
    K = D @ D.T
    
    # The objective is now to solve min <q,Kq>, Tr(Q)=1, Q>=0
    c1 = np.mean(x[0, :])
    c2 = np.mean(x[1, :])
    c3 = np.mean(x[2, :])
    r2 = np.var(x[0, :]) + np.var(x[1, :]) + np.var(x[2, :])
    u = np.array([1/3, 1/3, 1/3, 0, 0, 0, -2*c1/3, -2*c2/3, -2*c3/3, (c1**2 + c2**2 + c3**2 - r2)/3])
    
    # Douglas-Rachford (Lions-Mercier) iterative algorithm
    gamma = 10  # Parameter in ]0,+infty[
    M = gamma * K + np.eye(K.shape[0])
    
    def proxf1(q):
        return linalg.solve(M, q)
    
    def proxf2(q):
        return project_on_B(q)
    
    p = u
    CF = np.zeros(nit + 1)
    
    for k in range(nit):
        q = proxf2(p)
        CF[k] = 0.5 * q.T @ K @ q
        p = p + 1.0 * (proxf1(2. * q - p) - q)
    
    q = proxf2(q)
    print(q)
    CF[nit] = 0.5 * q.T @ K @ q
    
    A2 = np.array([
        [q[0], q[3]/np.sqrt(2), q[4]/np.sqrt(2)],
        [q[3]/np.sqrt(2), q[1], q[5]/np.sqrt(2)],
        [q[4]/np.sqrt(2), q[5]/np.sqrt(2), q[2]]
    ])
    b2 = np.array([q[6], q[7], q[8]])
    c2 = q[9]
        
    # Go back to the initial basis
    A = P.T @ A2 @ P
    b = -2 * A @ t + P.T @ b2
    c = t.T @ A @ t - b2.T @ P @ t + c2
    
    q = np.array([
        A[0, 0], A[1, 1], A[2, 2], 
        np.sqrt(2) * A[1, 0], np.sqrt(2) * A[2, 0], np.sqrt(2) * A[2, 1],
        b[0], b[1], b[2], c
    ])
    
    # Normalization to stay on the simplex
    q = q / (A[0, 0] + A[1, 1] + A[2, 2])
    
    return q, CF, A, b, c

def project_on_B(q0):
    Q0 = np.array([
        [q0[0], q0[3]/np.sqrt(2), q0[4]/np.sqrt(2)],
        [q0[3]/np.sqrt(2), q0[1], q0[5]/np.sqrt(2)],
        [q0[4]/np.sqrt(2), q0[5]/np.sqrt(2), q0[2]]
    ])
    
    S0, U = linalg.eigh(Q0)
    s = projsplx(S0)
    S = np.diag(s)
    Q = U @ S @ U.T
    
    q = np.zeros_like(q0)
    q[0] = Q[0, 0]
    q[1] = Q[1, 1]
    q[2] = Q[2, 2]
    q[3] = np.sqrt(2) * Q[1, 0]
    q[4] = np.sqrt(2) * Q[2, 0]
    q[5] = np.sqrt(2) * Q[2, 1]
    q[6:] = q0[6:]
    
    return q

def projsplx(v):
    # Project onto the simplex
    n = len(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

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
    if filename.suffix != '.csv':
        print("Warning: Filename doesn't have '.csv' extension. Adding it.")
        # Create a new path with the correct extension
        filename = filename.with_suffix('.csv')

    df.to_csv(filename, index=False)
    print(f"Ellipsoids saved to {filename}")

#%% Display purposes
def create_ellipsoid_mesh(points):    
    # Create convex hull
    hull = ConvexHull(points)
    
    # Extract vertices and faces for mesh
    vertices = hull.points
    faces = hull.simplices
    
    return vertices, faces

def fibonacci_sphere(num_points):
    """
    Generate points uniformly distributed on a unit sphere using the Fibonacci spiral method.
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        Array of shape (3, num_points) containing the x, y, z coordinates
    """
    points = np.zeros((3, num_points))
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    for i in range(num_points):
        # Spread points evenly between -1 and 1 on z-axis
        z = 1 - (2 * i) / (num_points - 1)
        
        # Compute radius at z
        radius = np.sqrt(1 - z**2)
        
        # Golden angle increment
        theta = 2 * np.pi * i / phi
        
        # Convert to Cartesian coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        points[0, i] = x
        points[1, i] = y
        points[2, i] = z
        
    return points

class Ellipsoid:
    def __init__(self):
        self.nsamples = 100
        self.A = []
        self.b = []
        self.c = []
        
        self.eigvecs = []
        self.axes_length = []
        self.center = []
        self.samples = []

        self.fitting_points = []

    def get_ellipsoid_from_Abc(self):
        # Compute the center of the ellipsoid
        x0 = -0.5 * np.linalg.inv(self.A) @ self.b

        # Transform to centered form: (x - x0)^T A (x - x0) = r^2
        r2 = x0.T @ self.A @ x0 + self.b.T @ x0 + self.c

        # Eigen-decomposition of A to get axes
        eigvals, eigvecs = np.linalg.eigh(self.A)
        axes = np.sqrt(-r2 / eigvals)  # semi-axis lengths

        self.eigvecs = eigvecs
        self.center = x0
        self.axes_length = axes
        return 
    
    def fit_ellipsoid(self):
        q, CF, A, b, c = ellipsoid3d_fitting_dr_svd(np.array(self.fitting_points).T)
        self.A = A
        self.b = b 
        self.c = c 
    
    def sample_ellipsoid(self, num_points=500):
        """Sample points on the ellipsoid defined by <Ax, x> + <b,x> + c = 0."""
        
        # First convert the algebraic representation
        self.get_ellipsoid_from_Abc()
        # Sampling the sphere...
        sphere = fibonacci_sphere(num_points)
        # Scaling and rotating it
        ellipsoid_samples = (self.eigvecs @ np.diag(self.axes_length) @ sphere) + self.center[:, None]

        self.samples = ellipsoid_samples.T
        return

def main():
    viewer = napari.Viewer()

    # Load your image using tifffile
    image = tifffile.imread("./img_test.tif")
    
    # Add the image to the Napari viewer
    viewer.add_image(image,  blending='translucent_no_depth', name="Image")

    points_layer_list = []
    ellipsoid_list = []

    #%% Displaying an ellipsoid
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

    #%% Saving 
    # Create the magicgui widget with the save button
    @magicgui(call_button="Save results", filename={"widget_type": "FileEdit", "mode": "w", "filter": "*.csv"})

    def save_dialog(filename: str, viewer):
        if filename:
            # Call the save function when the user presses the button
            if ellipsoid_list:                    
                save_ellipsoids_to_excel(ellipsoid_list, filename)
                print(f"Ellipsoids saved to {filename}")
        else:
            print("No filename provided")

    widget = save_dialog  # We don't call it here, just reference it
    viewer.window.add_dock_widget(widget, area='right')
    widget.viewer.bind(viewer)
    viewer.show()

    #%% Scaling procedure + button
    @magic_factory(
        x_scale={"label": "X Scale", "widget_type": "FloatSpinBox", "value": 1.0, "min": 0.0001, "max": 100.0},
        y_scale={"label": "Y Scale", "widget_type": "FloatSpinBox", "value": 1.0, "min": 0.0001, "max": 100.0},
        z_scale={"label": "Z Scale", "widget_type": "FloatSpinBox", "value": 1.0, "min": 0.0001, "max": 100.0},
    )
    def scale_widget(x_scale: float, y_scale: float, z_scale: float):
        pass  # This function won't be called automatically

    scale_controls = scale_widget()

    def update_scale():
        # Get the current values from the widget
        x_scale = scale_controls.x_scale.value
        y_scale = scale_controls.y_scale.value
        z_scale = scale_controls.z_scale.value
        
        # Note that napari uses (z, y, x) order for scale
        new_scale = (z_scale, y_scale, x_scale)
        
        for layer in viewer.layers:
            layer.scale = new_scale
        
        print(f"Updated scale to: {new_scale}")

    # Connect signals to update function
    scale_controls.x_scale.changed.connect(update_scale)
    scale_controls.y_scale.changed.connect(update_scale)
    scale_controls.z_scale.changed.connect(update_scale)

    # Add widget to the viewer
    viewer.window.add_dock_widget(scale_controls)

    #%% Button clicks
    def on_mouse_press(layer, event):
            layer_number = points_layer_list.index(layer)
            ellipsoid = ellipsoid_list[layer_number]

            if event.button == 1:  # Left-click adds a point
                position = viewer.cursor.position  # Get cursor position
                # Adds the points to fit to ellipsoid
                if position is None: 
                    return
                
                ellipsoid = ellipsoid_list[layer_number]
                ellipsoid.fitting_points.append(position)

                # Update points display
                layer.data = np.array(ellipsoid.fitting_points)
                print(f"Ellipsoid {layer_number} contains {len(ellipsoid.fitting_points)} points. Added ({position[0]},{position[1]})")
                save_objects_to_pickle(ellipsoid_list)

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

    #%% Fit Ellipsoid procedure + button
    @magicgui(call_button="Fit Ellipsoid")
    def fit_ellipsoid_button():
        active_layer = viewer.layers.selection.active
        if active_layer in points_layer_list:
            active_layer_index = points_layer_list.index(active_layer)

        ellipsoid = ellipsoid_list[active_layer_index]
        if len(ellipsoid.fitting_points) >= 10:  # Need at least 10 points to fit an ellipsoid
            # We fit the clicked points, sample the estimated ellipsoid and add a fitted ellipsoid layer
            ellipsoid.fit_ellipsoid()
            ellipsoid.sample_ellipsoid()
            viewer.add_points(ellipsoid.samples, size=2, face_color='blue', blending='translucent_no_depth', name=f'FitEllipsoid {active_layer_index}')
            #display_ellipsoid_mesh(viewer, ellipsoid.samples, color=[0, 0, 1, 0.7], name=f'FitEllipsoid {active_layer_index}') # For mesh visualization
            print(f"Fitted Ellipsoid {active_layer_index} now displayed.")

            # We can add a new ellipsoid layer
            new_point_layer = viewer.add_points([], size=1, face_color='red', ndim=3,  blending='translucent_no_depth')
            points_layer_list.append(new_point_layer)
            viewer.layers.selection.active = new_point_layer

            ellipsoid_list.append(Ellipsoid())
            new_point_layer.mouse_drag_callbacks.clear()  # Remove old bindings
            new_point_layer.mouse_drag_callbacks.append(on_mouse_press)
            update_scale()
            print(f"Added a new point layer for your new ellipsoid.")

        else:
            print("Not enough points to fit an ellipsoid.")

    viewer.window.add_dock_widget(fit_ellipsoid_button)

    #%% Putting new image at last place to see the points
    def on_new_layer(event):
        layer = event.value
        if isinstance(layer, napari.layers.Image):
            print(f"New image added: {layer.name}")
            viewer.layers.move(layer, 0)  # Push image to the bottom
    
    viewer.layers.events.inserted.connect(on_new_layer)

    #%% The first layer is instantiated
    points_layer = viewer.add_points([], size=1, face_color='red', ndim=3,  blending='translucent_no_depth')
    points_layer_list.append(points_layer)
    ellipsoid_list.append(Ellipsoid())
    # Attach both left and right click events
    points_layer.mouse_drag_callbacks.clear()  # Remove old bindings
    points_layer.mouse_drag_callbacks.append(on_mouse_press)

    #%% Launching the plugin
    napari.run()

if __name__ == "__main__":
    main()
