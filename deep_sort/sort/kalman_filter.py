# vim: expandtab:ts=4:sw=4
import numpy as np
# import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

def cho_factor_np(A):
    """
    Computes the Cholesky factor of a symmetric positive-definite matrix A.
    Returns the lower triangular matrix L such that A = L * L.T
    
    Parameters:
    A (np.ndarray): Symmetric positive-definite matrix of shape (n, n)

    Returns:
    L (np.ndarray): Lower triangular matrix of shape (n, n)
    """

    # initialize L
    L = np.zeros_like(A)

    for i in range(A.shape[0]):
        for j in range(i + 1):
            sum_k = np.dot(L[i, :j], L[j, :j])
            
            if i == j:  # Diagonal elements
                L[i, j] = np.sqrt(A[i, i] - sum_k)
            else:       # Off-diagonal elements
                L[i, j] = (A[i, j] - sum_k) / L[j, j]
    
    return L

def forward(L, b):
    """
    Forward substitution to solves Ly = b.
    
    Parameters:
    L (np.ndarray): Lower triangular matrix of shape (n, n)
    b (np.ndarray): Right-hand side vector or matrix of shape (n,) or (n, k)

    Returns:
    y (np.ndarray): vector or matrix of shape (n,) or (n, k)
    """

    y = np.zeros_like(b)

    for i in range(L.shape[0]):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    return y

def backward(LT, y):
    """
    Backward substitution to solve L^Tx = y.
    
    Parameters:
    LT (np.ndarray): Upper triangular matrix (transpose of lower triangular matrix L) of shape (n, n)
    y (np.ndarray): Right-hand side vector or matrix of shape (n,) or (n, k)

    Returns:
    x (np.ndarray): vector or matrix of shape (n,) or (n, k)
    """

    x = np.zeros_like(y)

    for i in range(LT.shape[0]-1, -1, -1):   #as the LT is upper triangular so the values will obtained from boutmn
        x[i] = (y[i] - np.dot(LT[i, i+1:], x[i+1:])) / LT[i, i]
    
    return x

def cho_solve_np(L_lower, b):
    """
    Solves the linear equations Ax = b using the Cholesky factorization.
    
    Parameters:
    L_lower (tuple): Lower triangular Cholesky factor
    b (np.ndarray): Right-hand side vector or matrix of shape (n,) or (n, k)

    Returns:
    x (np.ndarray): vector or matrix of shape (n,) or (n, k)
    """
    L = L_lower
    
    # Forward substitution to solve for Ly = b
    y = forward(L, b)
    
    # Backward substitution to solve L^Tx = y
    x = backward(L.T, y)
    
    return x

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 60
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[0],   # the center point x
            2 * self._std_weight_position * measurement[1],   # the center point y
            1 * measurement[2],                               # the ratio of width/height
            2 * self._std_weight_position * measurement[3],   # the height
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            1 * mean[2],
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[0],
            self._std_weight_velocity * mean[1],
            0.1 * mean[2],
            self._std_weight_velocity * mean[3]]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            0.1 * mean[2],
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor_np = cho_factor_np(projected_cov)
        kalman_gain = cho_solve_np(chol_factor_np, np.dot(covariance, self._update_mat.T).T).T
        
        # chol_factor, lower = scipy.linalg.cho_factor(
        #     projected_cov, lower=True, check_finite=False)
        # kalman_gain = scipy.linalg.cho_solve(
        #     (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
        #     check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = forward(cholesky_factor, d.T)
        # z = scipy.linalg.solve_triangular(
        #     cholesky_factor, d.T, lower=True, check_finite=False,
        #     overwrite_b=True)
        # zz = scipy.linalg.cho_solve(
        #     (cholesky_factor, True), d.T)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
