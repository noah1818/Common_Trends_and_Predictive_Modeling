import numpy as np
from numpy.typing import NDArray

from typing import Tuple

from scipy.sparse.linalg import eigsh

class GroupLassoProxRankSelection:
    """
    Group Lasso with proximal gradient updates for rank selection in
    high-dimensional VECMs.

    Implements column-wise soft-thresholding (proximal operator) for
    coefficient matrices. Supports both vanilla proximal gradient descent
    and monotone FISTA (accelerated gradient with safeguard).
    
    Attributes
    ----------
    const_term : NDArray[np.floating] or None
        Precomputed gradient constant term (-2 Z Y^T).
    ZZt : NDArray[np.floating] or None
        Precomputed Gram matrix Z Z^T.
    """
    def __init__(self) -> None:
        """
        Initialize an empty GroupLassoProxRankSelection object.
        """
        self.const_term: NDArray[np.floating] | None = None
        self.ZZt: NDArray[np.floating] | None = None

    def compute_weights(self, weights: NDArray[np.floating], gamma: float) -> NDArray[np.floating]:
        """
        Compute adaptive weights for group lasso.

        Parameters
        ----------
        weights : NDArray[np.floating]
            Base weights, typically norms of groups or columns.
        gamma : float
            Power exponent applied to weights.

        Returns
        -------
        NDArray[np.floating]
            Adaptive group weights (1 / weights**gamma).

        Raises
        ------
        ValueError
            If input weights contain non-positive values.
        """
        # Sanity check
        if np.any(weights <= 0):
            raise ValueError("All input weights must be strictly positive.")
        return 1.0 / (weights ** gamma)
    
    def clip_weights(self, weights: NDArray[np.floating], min_val: float = 1e-3, max_val: float = 1e3) -> NDArray[np.floating]:
        """
        Clip weights to a specified range.

        Parameters
        ----------
        weights : NDArray[np.floating]
            Input weights.
        min_val : float, optional
            Minimum allowed value (default: 1e-3).
        max_val : float, optional
            Maximum allowed value (default: 1e3).

        Returns
        -------
        NDArray[np.floating]
            Clipped weights.
        """
        return np.clip(weights, min_val, max_val)
    
    def scale_weights(self, weights: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Normalize group weights so that they sum to 1.

        Parameters
        ----------
        weights : NDArray[np.floating]
            Input weight array.

        Returns
        -------
        NDArray[np.floating]
            Normalized weights.

        Raises
        ------
        ValueError
            If the sum of weights is zero.
        """
        total = np.sum(weights)
        if total == 0.0:
            raise ValueError("Cannot scale weights: sum is zero.")
        return weights / total
    
    def construct_Y_Z_matrices(self, Q_hat: NDArray[np.floating], Y_tilde: NDArray[np.floating], dY_tilde: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Construct the dependent (Y) and regressor (Z) matrices for rank selection.

        Parameters
        ----------
        Q_hat : NDArray[np.floating]
            Orthonormal matrix from QR decomposition.
        Y_tilde : NDArray[np.floating]
            Residualized lagged levels.
        dY_tilde : NDArray[np.floating]
            Residualized differences.

        Returns
        -------
        NDArray[np.floating]
            Dependent matrix.
        NDArray[np.floating]
            Regressor matrix (Q_hat.T @ Y_tilde).
        
        Raises
        ------
        ValueError
            If dimensions of inputs are inconsistent.
        """
        # Sanity checks
        if Q_hat.ndim != 2 or Y_tilde.ndim != 2 or dY_tilde.ndim != 2:
            raise ValueError("All inputs must be 2D arrays.")
        if Q_hat.shape[0] != Y_tilde.shape[0] or Y_tilde.shape != dY_tilde.shape:
            raise ValueError("Shapes of Q_hat, Y_tilde, and dY_tilde must align.")
        return dY_tilde, Q_hat.T @ Y_tilde
    
    def prepare_const_term(self, Y: NDArray[np.floating], Z: NDArray[np.floating]) -> None:
        """
        Precompute the constant term of the gradient.

        Parameters
        ----------
        Y : NDArray[np.floating]
            Dependent matrix.
        Z : NDArray[np.floating]
            Regressor matrix.
        
        Raises
        ------
        ValueError
            If dimensions are inconsistent.
        """
        if Y.ndim != 2 or Z.ndim != 2 or Y.shape[1] != Z.shape[1]:
            raise ValueError("Y and Z must be 2D with the same number of columns.")
        self.const_term = -2.0 * (Z @ Y.T)

    def prepare_ZZt(self, Z: NDArray[np.floating]) -> None:
        """
        Precompute the Gram matrix Z Z^T.

        Parameters
        ----------
        Z : NDArray[np.floating]
            Regressor matrix.
        
        Raises
        ------
        ValueError
            If Z is not 2D.
        """
        if Z.ndim != 2:
            raise ValueError("Z must be 2D.")
        self.ZZt = Z @ Z.T
    
    def get_eta(self, eps: float = 1e-6, offset: float = 0.9) -> float:
        """
        Compute a safe step size eta from largest eigenvalue of ZZ^T.

        Parameters
        ----------
        eps : float, optional
            Minimum eigenvalue threshold (default: 1e-6).
        offset : float, optional
            Scaling factor to ensure eta is slightly less than 1/lambda_max (default: 0.9).

        Returns
        -------
        float
            Step size eta, set to 0.9 / lambda_max or 1.0 if lambda_max < eps.

        Raises
        ------
        ValueError
            If ZZt has not been prepared.
        """
        # Sanity check
        if self.ZZt is None:
            raise ValueError("ZZt not initialized. Call prepare_ZZt(...) first.")
        
        lam_max, _ = eigsh(self.ZZt, k=1, which="LM") # largest eigenvalue
        if lam_max[0] < eps:
            return 1.0
        return (1.0 / lam_max[0]) * offset # slightly less than 1/largest eigenvalue
    
    def grad_linear_part(self, R: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the linear part of the gradient.

        Parameters
        ----------
        R : NDArray[np.floating]
            Current estimate of matrix R.

        Returns
        -------
        NDArray[np.floating]
            Gradient linear component (2 Z Z^T R).

        Raises
        ------
        ValueError
            If ZZt has not been initialized.
        """
        if self.ZZt is None:
            raise ValueError("ZZt not initialized. Call prepare_ZZt(...) first.")
        return 2.0 * (self.ZZt @ R)
    
    def gradient(self, R: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the full gradient at R.

        Parameters
        ----------
        R : NDArray[np.floating]
            Current estimate of matrix R.

        Returns
        -------
        NDArray[np.floating]
            Full gradient (const_term + 2 Z Z^T R).
        
        Raises
        ------
        ValueError
            If const_term or ZZt have not been initialized.
        """
        if self.const_term is None or self.ZZt is None:
            raise ValueError("Call prepare_const_term(...) and prepare_ZZt(...) first.")
        return self.const_term + self.grad_linear_part(R)
    
    '''
    Instead of looping over groups, we can do it in a vectorized manner.
    This should be much faster for larger p, K, k.
    But one can also use the here excluded non-vectorized version for smaller problems, or debugging.

    def prox_group_block(self, Bj, eta, wj, tau, eps=1e-8): # small eps to avoid division by zero
        norm_Bj = np.linalg.norm(Bj, "fro")
        if norm_Bj <= eps:
            return np.zeros_like(Bj)     # was returning Bj (wrong)
        scaling_factor = max(0, 1 - (eta * tau * wj) / norm_Bj)
        return scaling_factor * Bj

    def update_step(self, B, eta, weights, tau, groups):
        Y = B - eta * self.gradient(B)
        B_new = np.zeros_like(B)
        for j, group in enumerate(groups):
            block = Y[:, group[0]:group[1]]
            B_new[:, group[0]:group[1]] = self.prox_group_block(block, eta, weights[j], tau)
        
        return B_new
    '''

    def update_step_vectorized(self, R: NDArray[np.floating], eta: float, weights: NDArray[np.floating], tau: float, eps: float = 1e-8) -> NDArray[np.floating]:
        """
        Perform one proximal gradient update step (vectorized).

        Parameters
        ----------
        R : NDArray[np.floating]
            Current estimate of matrix R.
        eta : float
            Step size.
        weights : NDArray[np.floating]
            Group weights.
        tau : float
            Regularization strength.
        eps : float, optional
            Small constant to avoid division by zero.

        Returns
        -------
        NDArray[np.floating]
            Updated estimate of R after applying group-lasso proximal operator.
        
        Raises
        ------
        ValueError
            If shapes of R and weights are inconsistent.
        """
        if R.ndim != 2 or weights.shape[0] != R.shape[1]:
            raise ValueError("weights must have same length as number of columns in R.")

        Y = R - eta * self.gradient(R)

        # column norms
        norms = np.linalg.norm(Y, axis=0) + eps
        scaling = np.maximum(0.0, 1.0 - (eta * tau * weights) / norms)

        return Y * scaling[None, :]  # scale each column
    
    def objective(self, R: NDArray[np.floating], Y: NDArray[np.floating], Z: NDArray[np.floating], tau: float, weights: NDArray[np.floating]) -> float:
        """
        Compute the group lasso objective function.

        Parameters
        ----------
        R : NDArray[np.floating]
            Current estimate of matrix R.
        Y : NDArray[np.floating]
            Dependent matrix.
        Z : NDArray[np.floating]
            Regressor matrix.
        tau : float
            Regularization strength.
        weights : NDArray[np.floating]
            Group weights.

        Returns
        -------
        float
            Value of the objective (loss + penalty).

        Notes
        -----
        This uses the objective function in 3.27 of our paper.
        """
        residual = Y - R.T @ Z
        loss = np.linalg.norm(residual, "fro")**2
        norms = np.linalg.norm(R, axis=0)
        penalty = tau * np.dot(weights, norms)
        return loss + penalty

    def check_stopping_criteria(self, history: list[float], R_new: NDArray[np.floating], R: NDArray[np.floating], tol: float) -> bool:
        """
        Check convergence using parameter change and objective decrease.

        Parameters
        ----------
        history : list of float
            Sequence of objective values per iteration.
        R_new : NDArray[np.floating]
            Current iterate.
        R : NDArray[np.floating]
            Previous iterate.
        tol : float
            Tolerance for both parameter change and relative objective change.

        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        diff = np.linalg.norm(R_new - R, "fro")
        if len(history) > 1:
            obj_diff = abs(history[-1] - history[-2]) / max(1.0, abs(history[-2]))
        else:
            obj_diff = np.inf

        return diff < tol or obj_diff < tol

    def fit(self, R_init: NDArray[np.floating], Y: NDArray[np.floating], Z: NDArray[np.floating], tau: float, weights: NDArray[np.floating], epochs: int = 100, tol: float = 1e-8, verbose: bool = False) -> Tuple[NDArray[np.floating], list[float]]:
        """
        Fit group lasso for rank selection via proximal gradient descent.

        Parameters
        ----------
        R_init : NDArray[np.floating]
            Initialization for coefficient matrix.
        Y : NDArray[np.floating]
            Response matrix.
        Z : NDArray[np.floating]
            Regressor matrix.
        tau : float
            Regularization parameter.
        weights : NDArray[np.floating]
            Column weights.
        epochs : int, optional
            Maximum number of iterations (default: 100).
        tol : float, optional
            Convergence tolerance (default: 1e-8).
        verbose : bool, optional
            If True, print objective values per iteration.

        Returns
        -------
        Tuple[NDArray[np.floating], List[float]]
            Final coefficient estimate and objective history.

        Raises
        ------
        ValueError
            If input shapes are inconsistent.
        """
        if R_init.ndim != 2 or Y.ndim != 2 or Z.ndim != 2:
            raise ValueError("R_init, Y, and Z must all be 2D arrays.")
        if weights.shape[0] != R_init.shape[1]:
            raise ValueError("weights must match the number of columns in R_init.")
        if Y.shape[1] != Z.shape[1]:
            raise ValueError("Y and Z must have the same number of columns.")

        # Precompute constant terms
        self.prepare_const_term(Y, Z)
        self.prepare_ZZt(Z)
        eta = self.get_eta()

        R = R_init.copy()
        history: list[float] = []
        
        for epoch in range(epochs):
            R_new = self.update_step_vectorized(R, eta, weights, tau)
            obj_value = self.objective(R_new, Y, Z, tau, weights)
            history.append(obj_value)
            
            if verbose:
                print(f"Epoch {epoch+1}, Objective: {obj_value:.6f}")
            
            if self.check_stopping_criteria(history, R_new, R, tol=tol):
                if verbose:
                    print(f"Converged at epoch {epoch+1}")
                break

            R = R_new
        
        return R, history

    # FISTA implementation
    def compute_FISTA_tk(self, tk: float) -> float:
        """
        Compute FISTA acceleration parameter t_{k+1}.

        Parameters
        ----------
        tk : float
            Previous momentum parameter.

        Returns
        -------
        float
            Updated momentum parameter.
        """
        return (1 + np.sqrt(1 + 4 * tk**2)) / 2
    
    def fit_FISTA(self, R_init: NDArray[np.floating], Y: NDArray[np.floating], Z: NDArray[np.floating], tau: float, weights: NDArray[np.floating], epochs: int = 100, tol: float = 1e-4, verbose: bool = False) -> Tuple[NDArray[np.floating], list[float]]:
        """
        Fit group lasso for rank selection via monotone FISTA.

        Uses accelerated proximal gradient descent with a monotonicity safeguard:
        if the accelerated step increases the objective, the algorithm falls
        back to a standard proximal gradient step.

        Parameters
        ----------
        R_init : NDArray[np.floating]
            Initialization for coefficient matrix.
        Y : NDArray[np.floating]
            Response matrix.
        Z : NDArray[np.floating]
            Regressor matrix.
        tau : float
            Regularization parameter.
        weights : NDArray[np.floating]
            Column weights.
        epochs : int, optional
            Maximum number of iterations (default: 100).
        tol : float, optional
            Convergence tolerance (default: 1e-4).
        verbose : bool, optional
            If True, print objective values per iteration.

        Returns
        -------
        Tuple[NDArray[np.floating], List[float]]
            Final coefficient estimate and objective history.

        Raises
        ------
        ValueError
            If input shapes are inconsistent.
        """
        if R_init.ndim != 2 or Y.ndim != 2 or Z.ndim != 2:
            raise ValueError("R_init, Y, and Z must all be 2D arrays.")
        if weights.shape[0] != R_init.shape[1]:
            raise ValueError("weights must match the number of columns in R_init.")
        if Y.shape[1] != Z.shape[1]:
            raise ValueError("Y and Z must have the same number of columns.")

        # Precompute constants
        self.prepare_const_term(Y, Z)
        self.prepare_ZZt(Z)
        eta = self.get_eta()

        R = R_init.copy()
        history: list[float] = []

        tk = 1.0
        U = R_init.copy()

        for epoch in range(epochs):
            R_new = self.update_step_vectorized(U, eta, weights, tau)
            tk_new = self.compute_FISTA_tk(tk)
            U = R_new + ((tk - 1) / tk_new) * (R_new - R)

            obj_value = self.objective(R_new, Y, Z, tau, weights)
            history.append(obj_value)

            if verbose:
                print(f"Epoch {epoch+1}, Objective: {obj_value:.6f}")

            if self.check_stopping_criteria(history, R_new, R, tol=tol):
                if verbose:
                    print(f"Converged at epoch {epoch+1}")
                break

            if len(history) > 1 and history[-1] > history[-2]:
                U = R_new.copy()
                tk = 1.0
            else:
                tk = tk_new
                R = R_new

        return R, history
