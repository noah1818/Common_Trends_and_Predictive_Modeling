import numpy as np
from numpy.typing import NDArray

from typing import Tuple

from scipy.sparse.linalg import eigsh

class GroupLassoProxLagSelection:
    """
    Group Lasso with proximal gradient updates for lag selection in high-dimensional VECMs.

    Implements block-wise soft-thresholding (proximal operator) for grouped
    coefficient matrices. Supports both vanilla proximal gradient descent and monotonic
    FISTA (accelerated gradient) updates.

    Attributes
    ----------
    const_term : ndarray or None
        Precomputed term -2 Y Z^T for the gradient.
    ZZt : ndarray or None
        Precomputed Gram matrix Z Z^T for efficiency.
    """
    def __init__(self) -> None:
        """
        Initialize an empty GroupLassoProxLagSelection object.
        """
        self.const_term: NDArray[np.floating] | None = None  # -2 Y Z^T
        self.ZZt: NDArray[np.floating] | None = None         # Z Z^T (Gram)

    def compute_weights(self, Gamma_blocks: NDArray[np.floating], p: int, k: int, gamma: float, norm_type: int | str = np.inf, eps: float = 1e-6) -> NDArray[np.floating]:
        """
        Compute adaptive group weights for each lag block.

        Parameters
        ----------
        Gamma_blocks : NDArray[np.floating]
            Stacked coefficient matrix [Gamma_1 | ... | Gamma_p].
        p : int
            Number of lag groups.
        k : int
            Block size (columns per group).
        gamma : float
            Exponent for adaptive weights; weights[j] = (||block_j|| + eps)^(-gamma).
        norm_type : {int, str}, optional
            Norm type used to measure block magnitude (default: np.inf).
        eps : float, optional
            Small constant to avoid division by zero (default: 1e-6).

        Returns
        -------
        NDArray[np.floating]
            Adaptive weights for each block.

        Raises
        ------
        ValueError
            If the second dimension of Gamma_blocks is not equal to p * k.
        """
        # Sanity check
        if Gamma_blocks.ndim != 2 or Gamma_blocks.shape[1] != p * k:
            raise ValueError("Gamma_blocks must have shape (k, p*k).")

        weights = np.zeros(p)
        for j in range(p):
            norm_val = np.linalg.norm(Gamma_blocks[:, j*k:(j+1)*k].ravel(order="F"), norm_type)
            weights[j] = (norm_val + eps) ** (-gamma)
        return weights
    
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
        Normalize weights to sum to 1.

        Parameters
        ----------
        weights : NDArray[np.floating]
            Input weights.

        Returns
        -------
        NDArray[np.floating]
            Normalized weights summing to 1.

        Raises
        ------
        ValueError
            If the sum of weights is zero.
        """
        total = np.sum(weights)
        if total == 0.0:
            raise ValueError("Cannot scale weights: sum is zero.")
        return weights / total

    def helper_loss(self, p: int, dY_check: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Build lagged design matrices (Y, Z) for group-lasso regression.

        Parameters
        ----------
        p : int
            Number of lags.
        dY_check : NDArray[np.floating]
            Residualized differenced data used for lag selection.

        Returns
        -------
        Tuple[NDArray[np.floating], NDArray[np.floating]]
            Y : (K, T - p), target data.
            Z : (K*p, T - p), stacked lagged regressors.

        Raises
        ------
        ValueError
            If T <= p.
        """
        # Sanity check
        if dY_check.ndim != 2:
            raise ValueError("dY_check must be 2D (K, T).")
        T = dY_check.shape[1]
        if T <= p:
            raise ValueError("Need more observations than lags (T > p).")

        lags = []
        for j in range(1, p+1):
            lags.append(dY_check[:, p-j:T-j])
        Z = np.vstack(lags) # (K*p, T-p)
        Y = dY_check[:, p:T]
        return Y, Z
    
    def prepare_const_term(self, Y: NDArray[np.floating], Z: NDArray[np.floating]) -> None:
        """
        Precompute constant gradient term.

        Parameters
        ----------
        Y : NDArray[np.floating]
            Target matrix.
        Z : NDArray[np.floating]
            Regressor matrix.

        Returns
        -------
        None
        """
        # Sanity check
        if Y.ndim != 2 or Z.ndim != 2 or Y.shape[1] != Z.shape[1]:
            raise ValueError("Y and Z must be 2D with the same number of columns.")
        self.const_term = -2.0 * (Y @ Z.T)  # -2 Y Z^T

    def prepare_ZZt(self, Z: NDArray[np.floating]) -> None:
        """
        Precompute Gram matrix ZZ^T.

        Parameters
        ----------
        Z : NDArray[np.floating]
            Regressor matrix.

        Returns
        -------
        None
        """
        if Z.ndim != 2:
            raise ValueError("Z must be 2D.")
        self.ZZt = Z @ Z.T  # Z Z^T (Gram)
    
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
        if lam_max < eps:
            return 1.0
        return (1.0 / lam_max[0]) * offset # slightly less than 1/largest eigenvalue
    
    def compute_groups(self, p: int, k: int) -> list[Tuple[int, int]]:
        """
        Build index ranges for each group. These are the groups for the blocks of Gamma.

        Parameters
        ----------
        p : int
            Number of groups.
        k : int
            Block size (columns per group).

        Returns
        -------
        List[Tuple[int, int]]
            List of (start, end) indices for each block.
        """
        if p <= 0 or k <= 0:
            raise ValueError("p and k must be positive integers.")
        return [(j * k, (j + 1) * k) for j in range(p)]
    
    def grad_linear_part(self, Gamma: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the linear part of the gradient, 2 * Gamma * ZZ^T.

        Parameters
        ----------
        Gamma : NDArray[np.floating]
            Current coefficient matrix.

        Returns
        -------
        NDArray[np.floating]
            Linear gradient term.

        Raises
        ------
        ValueError
            If ZZt has not been prepared.
        """
        if self.ZZt is None:
            raise ValueError("ZZt not initialized. Call prepare_ZZt first.")
        return 2.0 * (Gamma @ self.ZZt) # 2 Gamma Z Z^T
    
    def gradient(self, Gamma: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the full gradient of the smooth loss term.

        Parameters
        ----------
        Gamma : NDArray[np.floating]
            Current coefficient matrix.

        Returns
        -------
        NDArray[np.floating]
            Full gradient = const_term + 2 * Gamma * ZZ^T.

        Raises
        ------
        ValueError
            If const_term or ZZt have not been prepared.
        """
        if self.const_term is None or self.ZZt is None:
            raise ValueError("Call prepare_const_term and prepare_ZZt before gradient.")
        return self.const_term + self.grad_linear_part(Gamma)
    
    '''
    Instead of looping over groups, we can do it in a vectorized manner.
    This should be much faster for larger p, K, k.
    But one can also use the here excluded non-vectorized version for smaller problems, or debugging.

    def prox_group_block(self, Gammaj, eta, wj, tau, eps=1e-8): # small eps to avoid division by zero
        norm_Gammaj = np.linalg.norm(Gammaj, "fro")
        if norm_Gammaj <= eps:
            return np.zeros_like(Gammaj)     # was returning Gammaj (wrong)
        scaling_factor = max(0, 1 - (eta * tau * wj) / norm_Gammaj)
        return scaling_factor * Gammaj

    def update_step(self, Gamma, eta, weights, tau, groups):
        Y = Gamma - eta * self.gradient(Gamma)
        Gamma_new = np.zeros_like(Gamma)
        for j, group in enumerate(groups):
            block = Y[:, group[0]:group[1]]
            Gamma_new[:, group[0]:group[1]] = self.prox_group_block(block, eta, weights[j], tau)
        
        return Gamma_new
    '''

    def update_step_vectorized(self, Gamma: NDArray[np.floating], eta: float, weights: NDArray[np.floating], tau: float, groups: list[Tuple[int, int]], eps: float = 1e-8) -> NDArray[np.floating]:
        """
        Perform a single vectorized proximal gradient update (group soft-thresholding).

        Parameters
        ----------
        Gamma : NDArray[np.floating]
            Current coefficient estimate.
        eta : float
            Step size.
        weights : NDArray[np.floating]
            Group weights.
        tau : float
            Regularization parameter.
        groups : list of tuple(int, int)
            Group index ranges (start, end) for each block.
        eps : float, optional
            Small value to avoid division by zero in block norms (default: 1e-8).

        Returns
        -------
        NDArray[np.floating]
            Updated coefficient estimate with block-wise shrinkage applied.

        Raises
        ------
        ValueError
            If groups are empty or inconsistent with Gamma's shape.
        """
        if len(groups) == 0:
            raise ValueError("groups list is empty.")
        p = len(groups)
        k = groups[0][1] - groups[0][0]
        if Gamma.ndim != 2 or Gamma.shape[1] != p * k:
            raise ValueError("Gamma must have shape (k, p*k).")
        if weights.shape[0] != p:
            raise ValueError("weights must have length p.")

        Y = Gamma - eta * self.gradient(Gamma)

        # reshape into (p, K, k)
        K_dim = Y.shape[0]
        Y_blocks = Y.reshape(K_dim, p, k).transpose(1, 0, 2)

        # block norms and scaling factors for group soft-thresholding
        norms = np.linalg.norm(Y_blocks, axis=(1, 2)) + eps
        scaling = np.maximum(0.0, 1.0 - (eta * tau * weights) / norms)

        # apply scaling in place
        Y_blocks *= scaling[:, None, None]

        return Y_blocks.transpose(1, 0, 2).reshape(Gamma.shape)
    
    def objective(self, Gamma: NDArray[np.floating], Y: NDArray[np.floating], Z: NDArray[np.floating], tau: float, weights: NDArray[np.floating], groups: list[Tuple[int, int]]) -> np.floating:
        """
        Compute the group-lasso objective value.

        Parameters
        ----------
        Gamma : NDArray[np.floating]
            Coefficient matrix.
        Y : NDArray[np.floating]
            Target matrix.
        Z : NDArray[np.floating]
            Regressor matrix.
        tau : float
            Regularization parameter.
        weights : NDArray[np.floating]
            Group weights.
        groups : list of tuple(int, int)
            Group index ranges.

        Returns
        -------
        np.floating
            Objective value.
        
        Notes
        -----
        This uses the objective function in 3.32 of our paper.
        """
        residual = Y - Gamma @ Z
        loss = np.linalg.norm(residual, "fro") ** 2

        p = len(groups)
        k = groups[0][1] - groups[0][0]
        Gamma_blocks = Gamma.reshape(-1, p, k).transpose(1, 0, 2)
        block_norms = np.linalg.norm(Gamma_blocks, axis=(1, 2))
        penalty = tau * np.dot(weights, block_norms)

        return loss + penalty
    
    def check_stopping_criteria(self, history: list[float], Gamma_new: NDArray[np.floating], Gamma: NDArray[np.floating], tol: float) -> bool:
        """
        Check convergence using parameter change and objective decrease.

        Parameters
        ----------
        history : list of float
            Sequence of objective values per iteration.
        Gamma_new : NDArray[np.floating]
            Current iterate.
        Gamma : NDArray[np.floating]
            Previous iterate.
        tol : float
            Tolerance for both parameter change and relative objective change.

        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        diff = np.linalg.norm(Gamma_new - Gamma, "fro")
        if len(history) > 1:
            obj_diff = abs(history[-1] - history[-2]) / max(1.0, abs(history[-2]))
        else:
            obj_diff = np.inf
        
        return diff < tol or obj_diff < tol

    def fit(self, Gamma_init: NDArray[np.floating], Y: NDArray[np.floating], Z: NDArray[np.floating], p: int, k: int, tau: float, weights: NDArray[np.floating], epochs: int = 100, tol: float = 1e-4, verbose: bool = False) -> Tuple[NDArray[np.floating], list[float]]:
        """
        Fit group lasso via proximal gradient descent.

        Parameters
        ----------
        Gamma_init : NDArray[np.floating]
            Initialization for coefficients.
        Y : NDArray[np.floating]
            Target matrix.
        Z : NDArray[np.floating]
            Regressor matrix.
        p : int
            Number of lag groups.
        k : int
            Block size (columns per group).
        tau : float
            Regularization parameter.
        weights : NDArray[np.floating]
            Group weights.
        epochs : int, optional
            Maximum iterations (default: 100).
        tol : float, optional
            Convergence tolerance (default: 1e-4).
        verbose : bool, optional
            If True, print objective values (default: False).

        Returns
        -------
        Tuple[NDArray[np.floating], List[float]]
            Final coefficient estimate and objective history.
        """
        # precompute gradient terms
        self.prepare_const_term(Y, Z)
        self.prepare_ZZt(Z)
        eta = self.get_eta()
        
        Gamma = Gamma_init.copy()
        groups = self.compute_groups(p, k)
        history: list[float] = []
        
        for epoch in range(epochs):
            Gamma_new = self.update_step_vectorized(Gamma, eta, weights, tau, groups)
            obj_value = self.objective(Gamma_new, Y, Z, tau, weights, groups)
            history.append(obj_value)
            
            if verbose:
                print(f"Epoch {epoch+1}, Objective: {obj_value:.6f}")
            
            if self.check_stopping_criteria(history, Gamma_new, Gamma, tol=tol):
                if verbose:
                    print(f"Converged at epoch {epoch+1}")
                break

            Gamma = Gamma_new
        
        return Gamma, history

    # FISTA implementation
    def compute_FISTA_tk(self, tk: float) -> float:
        """
        Compute next FISTA momentum parameter.

        Parameters
        ----------
        tk : float
            Current momentum parameter.

        Returns
        -------
        float
            Updated momentum parameter t_{k+1} = (1 + sqrt(1 + 4 tk^2)) / 2.
        """
        return (1 + np.sqrt(1 + 4 * tk**2)) / 2
    
    def fit_FISTA(self, Gamma_init: NDArray[np.floating], Y: NDArray[np.floating], Z: NDArray[np.floating], p: int, k: int, tau: float, weights: NDArray[np.floating], epochs: int = 100, tol: float = 1e-4, verbose: bool = False) -> Tuple[NDArray[np.floating], list[float]]:
        """
        Fit group lasso via **monotone** FISTA (accelerated proximal gradient with monotonicity safeguard).

        The update enforces a non-increasing objective by falling back to a
        non-accelerated proximal step whenever the accelerated step increases
        the objective.

        Parameters
        ----------
        Gamma_init : NDArray[np.floating]
            Initialization for coefficients.
        Y : NDArray[np.floating]
            Target matrix.
        Z : NDArray[np.floating]
            Regressor matrix.
        p : int
            Number of lag groups.
        k : int
            Block size (columns per group).
        tau : float
            Regularization parameter.
        weights : NDArray[np.floating]
            Group weights.
        epochs : int, optional
            Maximum iterations (default: 100).
        tol : float, optional
            Convergence tolerance (default: 1e-4).
        verbose : bool, optional
            If True, print objective values (default: False).

        Returns
        -------
        Tuple[NDArray[np.floating], List[float]]
            Final coefficient estimate and objective history.
        """
        # precompute gradient terms
        self.prepare_const_term(Y, Z)
        self.prepare_ZZt(Z)
        eta = self.get_eta()

        groups = self.compute_groups(p, k)

        Gamma = Gamma_init.copy()   # base iterate
        U = Gamma_init.copy()       # accelerated point
        tk = 1.0

        history: list[float] = []
        prev_obj = np.inf
        
        for epoch in range(epochs):
            # accelerated prox step at U
            Gamma_acc = self.update_step_vectorized(U, eta, weights, tau, groups)
            obj_acc = float(self.objective(Gamma_acc, Y, Z, tau, weights, groups))

            # monotone safeguard: if objective increased, fall back to base step
            if obj_acc > prev_obj:
                # reset momentum and take plain prox step at Gamma (monotone FISTA)
                Gamma_new = self.update_step_vectorized(Gamma, eta, weights, tau, groups)
                obj_value = float(self.objective(Gamma_new, Y, Z, tau, weights, groups))
                U = Gamma_new.copy()
                tk = 1.0  # restart momentum
            else:
                # accept accelerated step
                Gamma_new = Gamma_acc
                obj_value = obj_acc
                tk_new = self.compute_FISTA_tk(tk)
                U = Gamma_new + ((tk - 1.0) / tk_new) * (Gamma_new - Gamma)
                tk = tk_new

            history.append(obj_value)
            if verbose:
                print(f"Epoch {epoch + 1}, Objective: {obj_value:.6f}")

            # stopping check
            if self.check_stopping_criteria(history, Gamma_new, Gamma, tol=tol):
                if verbose:
                    print(f"Converged at epoch {epoch + 1}")
                Gamma = Gamma_new
                break

            # advance base iterate and objective
            Gamma = Gamma_new
            prev_obj = obj_value

        return Gamma, history
    
