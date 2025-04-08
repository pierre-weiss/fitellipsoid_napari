import numpy as np
import scipy.linalg as linalg
from scipy.linalg import eig, inv, eigh
from scipy.spatial import ConvexHull

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


#%% Display purposes
def create_ellipsoid_mesh(points):    
    # Create convex hull
    hull = ConvexHull(points)
    
    # Extract vertices and faces for mesh
    vertices = hull.points
    faces = hull.simplices
    
    return vertices, faces