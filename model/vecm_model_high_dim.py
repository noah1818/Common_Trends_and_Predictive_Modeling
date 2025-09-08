from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import det
from scipy.linalg import qr

from vecm_model_standard import VECMModel

class VECMModelHD(VECMModel):
    """
    High-dimensional extension of the VECM estimation procedure.

    Implements the Frisch-Waugh residualization for rank and lag selection, 
    QR with pivoting, etc as in Section 3.4 of our Paper. 
    The Model inherits from the 'standard' VECMModel class.

    Group lasso based rank and lagselection is implemented in a separate class
    (see model/group_lasso_prox_rank.py, model/group_lasso_prox_lag.py).
    """
    def __init__(self):
        """Initialize the high-dimensional VECM model."""
        super().__init__()

    def frisch_waugh_rank_matrices(self) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute Frisch-Waugh residualized matrices for rank estimation.

        Returns
        -------
        NDArray[np.floating] of shape (K, T_eff)
            Residualized differenced data.
        NDArray[np.floating] of shape (K, T_eff)
            Residualized lagged levels.
        NDArray[np.floating] of shape (K, K)
            Pre-estimator of the long-run impact matrix.

        Raises
        ------
        ValueError
            If required VECM matrices (_delta_X, _delta_Y, _Y_minus_1) are not built.

        Notes
        -----
        This follows Section 3.4.1 of our paper.
        """
        # Sanity checks
        if any(x is None for x in (
            getattr(self, "_delta_X", None),
            getattr(self, "_delta_Y", None),
            getattr(self, "_Y_minus_1", None),
        )):
            raise ValueError("VECM matrices not built. Call build_vecm_matrices(...) first.")
                
        # Aliases for readability
        dY = self._delta_Y
        dX = self._delta_X
        Y1 = self._Y_minus_1
        ''' 
        The following is computational more efficient then using 
        OdX, *_ = lstsq(dX.T, dY.T, rcond=None)  # coefficients
        dY_tilde = (dY.T - dX.T @ OdX).T
        OdX, *_ = lstsq(dX.T, Y1.T, rcond=None)
        Y1_tilde = (Y1.T - dX.T @ OdX).T
        Pi_tilde = (dY_tilde @ Y1.T) @ invert_matrix(Y1_tilde @ Y1_tilde.T)
        '''
        # Project out the effect of Delta X using QR decomposition (Frisch–Waugh residualization).
        Q, _ = np.linalg.qr(dX.T, mode="reduced")
        dY_tilde = dY - (dY @ Q) @ Q.T
        Y1_tilde = Y1 - (Y1 @ Q) @ Q.T

        # Pre estimate of Pi
        Sxy = dY_tilde @ Y1_tilde.T
        Sxx = Y1_tilde @ Y1_tilde.T
        Pi_tilde = Sxy @ np.linalg.solve(Sxx, np.eye(Sxx.shape[0]))

        return dY_tilde, Y1_tilde, Pi_tilde

    def frisch_waugh_lag_matrices(self) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute Frisch-Waugh residualized matrices for lag selection.

        Returns
        -------
        NDArray[np.floating] of shape (K, T_eff-1)
            Residualized differenced data, aligned for lag regression.
        NDArray[np.floating] of shape (K*P, T_eff-1)
            Residualized differenced regressors, aligned for lag regression.

        Raises
        ------
        ValueError
            If required VECM matrices (_delta_X, _delta_Y, _Y_minus_1) are not built.

        Notes
        -----
        This follows Section 3.4.2 of our paper.
        """
        # Sanity checks
        if any(x is None for x in (
            getattr(self, "_delta_X", None),
            getattr(self, "_delta_Y", None),
            getattr(self, "_Y_minus_1", None),
        )):
            raise ValueError("VECM matrices not built. Call build_vecm_matrices(...) first.")
        
        # Aliases for readability
        dY = self._delta_Y
        dX = self._delta_X
        Y1 = self._Y_minus_1
        
        '''
        The following is computational much more efficient then using 
        'Ox, *_ = lstsq(Y1.T, dX.T, rcond=None)  # coefficients
        X_check = (dX.T - Y1.T @ Ox).T
        Oy, *_ = lstsq(Y1.T, dY.T, rcond=None)
        Y_check = (dY.T - Y1.T @ Oy).T
        '''
       
        # QR residualization of dY and dX on Y1
        Q, _ = np.linalg.qr(Y1.T, mode="reduced") 
        Y_check = dY - (dY @ Q) @ Q.T
        X_check = dX - (dX @ Q) @ Q.T

        # Align lags: drop first column of Y_check, last column of X_check
        dY_check = Y_check[:, 1:]
        dX_check = X_check[:, :-1]

        return dY_check, dX_check
    
    def compute_mu_tilde(self, R_tilde: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the mu_tilde vector from the upper-triangular QR factor.

        Parameters
        ----------
        R_tilde : NDArray[np.floating] of shape (m, m)
            Upper-triangular matrix from QR decomposition.

        Returns
        -------
        NDArray[np.floating] of shape (m,)
            Sequence of row-wise norms of R_tilde.

        Raises
        ------
        ValueError
            If R_tilde is not a square matrix.

        Notes
        -----
        Used in Eq. (27) of our paper.
        """
        # Sanity check
        if R_tilde.ndim != 2 or R_tilde.shape[0] != R_tilde.shape[1]:
            raise ValueError("R_tilde must be a square upper-triangular matrix.")

        m = R_tilde.shape[0]
        mu_tilde = np.empty(m)

        # Each mu_tilde[k] is the norm of the k-th row, starting from diagonal
        for k in range(m):
            mu_tilde[k] = np.linalg.norm(R_tilde[k, k:])
        return mu_tilde
    
    def qr_decomp(self, A: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.intp]]:
        """
        Perform QR decomposition with column pivoting.

        Parameters
        ----------
        A : NDArray[np.floating] of shape (m, n)
            Input matrix.

        Returns
        -------
        NDArray[np.floating]
            Orthonormal Q matrix.
        NDArray[np.floating]
            Upper-triangular R matrix.
        NDArray[np.intp]
            Permutation indices from pivoting.

        Raises
        ------
        ValueError
            If A is not a 2D numpy array.

        Notes
        -----
        Used in Eq. (23) of our paper.
        """
        # Sanity check
        if not isinstance(A, np.ndarray) or A.ndim != 2:
            raise ValueError("A must be a 2D NumPy array.")
        
        Q_hat, R_tilde, Perm = qr(A, pivoting=True , mode='economic')
        return Q_hat, R_tilde, Perm

    def residual_covariance(self, A: NDArray[np.floating], B: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute residual covariance matrix from VECM regression.

        Parameters
        ----------
        A : NDArray[np.floating]
            Response matrix.
        B : NDArray[np.floating]
            Fitted values.

        Returns
        -------
        NDArray[np.floating] of shape (K, K)
            Estimated residual covariance matrix.

        Raises
        ------
        ValueError
            If effective sample size T_eff is not set, or if A and B have mismatched shapes.

        Notes
        -----
        Used in Section 3.4 of our paper.
        """
        # Sanity checks
        if not isinstance(self.T_eff, int) or self.T_eff <= 0:
            raise ValueError("Effective sample size T_eff must be set.")

        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape.")

        T_eff = self.T_eff
        Sigma = 1 / T_eff * (A - B) @ (A - B).T
        return Sigma
    
    def rank_pre_estimate(self, mu_tilde: NDArray[np.floating], epsilon: float = 1e-5) -> int:
        """
        Pre-estimate the cointegration rank from mu_tilde.

        Parameters
        ----------
        mu_tilde : NDArray[np.floating]
            Sequence of row-wise norms of R_tilde.
        epsilon : float, optional
            Threshold below which values are considered zero.

        Returns
        -------
        int
            Estimated rank.

        Raises
        ------
        ValueError
            If mu_tilde is not a 1D array.
        """
        if mu_tilde.ndim != 1:
            raise ValueError("mu_tilde must be a 1D array.")
        r_hat_qr_decomp = sum(mu_j > epsilon for mu_j in mu_tilde)
        return r_hat_qr_decomp
    
    def get_beta(self, Q_hat: NDArray[np.floating], r: int) -> NDArray[np.floating]:
        """
        Extract beta from QR decomposition.

        Parameters
        ----------
        Q_hat : NDArray[np.floating]
            Orthonormal Q matrix from QR decomposition.
        r : int
            Estimated cointegration rank.

        Returns
        -------
        NDArray[np.floating] of shape (K, r)
            Cointegration matrix estimate.

        Raises
        ------
        ValueError
            If r is not between 1 and Q_hat.shape[1]-1.
        """
        # Sanity check
        if r <= 0 or r >= Q_hat.shape[1]:
            raise ValueError(f"Rank r must be between 1 and {Q_hat.shape[1]-1}")
        
        return Q_hat[:, : r]
    
    def get_alpha(self, P_mat: NDArray[np.floating], R_hat: NDArray[np.floating], r: int) -> NDArray[np.floating]:
        """
        Extract alpha from QR decomposition and permutation using R_hat from the group lasso solution.

        Parameters
        ----------
        P_mat : NDArray[np.floating]
            Permutation matrix from pivoting.
        R_hat : NDArray[np.floating]
            Upper-triangular R matrix from QR decomposition.
        r : int
            Estimated cointegration rank.

        Returns
        -------
        NDArray[np.floating] of shape (K, r)
            Loading matrix estimate.

        Raises
        ------
        ValueError
            If r is not between 1 and R_hat.shape[1]-1,
            if P_mat is not square or if r is not between 1 and R_hat.shape[1]-1.
        """
        # Sanity checks
        if r <= 0 or r >= R_hat.shape[1]:
            raise ValueError(f"Rank r must be between 1 and {R_hat.shape[1]-1}")
        
        if P_mat.shape[0] != P_mat.shape[1]:
            raise ValueError("P_mat must be a square permutation matrix.")
        
        return P_mat @ R_hat.T[:, : r]
    
    def bic_criterion(self, Sigma_w: NDArray[np.floating], A: NDArray[np.floating]) -> float:
        """
        Compute Bayesian Information Criterion (BIC)-type score.

        Parameters
        ----------
        Sigma_w : NDArray[np.floating]
            Residual covariance matrix.
        A : NDArray[np.floating]
            Parameter matrix.

        Returns
        -------
        float
            BIC score.

        Raises
        ------
        ValueError
            If Sigma_w is not square.
        LinAlgError
            If det(Sigma_w) cannot be computed reliably.
        """
        # Sanity check
        if Sigma_w.ndim != 2 or Sigma_w.shape[0] != Sigma_w.shape[1]:
            raise ValueError("Sigma_w must be square.")
        
        T_eff = self.T_eff
        logdet = np.log(det(Sigma_w))
        nonzeros = np.count_nonzero(A) # Number of active coefficients
        penalty = (np.log(T_eff) / T_eff) * nonzeros
        return logdet + penalty       

    def select_lag_from_blocks(self, Gamma_hat: NDArray[np.floating], P: int, eps: float = 1e-5) -> Tuple[NDArray[np.floating], int, NDArray[np.floating]]:
        """
        Select lag order from block Frobenius norms.

        Parameters
        ----------
        Gamma_hat : NDArray[np.floating] of shape (K, K*P)
            Stacked short-run coefficient blocks.
        P : int
            Maximum lag order.
        eps : float, optional
            Threshold for activity.

        Returns
        -------
        NDArray[np.floating]
            Frobenius norms of lag blocks.
        int
            Estimated lag order.
        NDArray[np.floating]
            Concatenation of active lag blocks.
        
        Raises
        ------
        ValueError
            If Gamma_hat has incompatible shape.
        """
        # Sanity check
        K = self.K
        if Gamma_hat.shape[1] != K * P:
            raise ValueError(f"Gamma_hat must have shape (K, K*P), got {Gamma_hat.shape}")

        norms = np.array([
            np.linalg.norm(Gamma_hat[:, j * K : (j + 1) * K], "fro")
            for j in range(P)
        ])
        active = np.where(norms > eps)[0]    
        p_hat = 0 if active.size == 0 else active.max() + 1

        # Concatenate active blocks or return empty matrix if none
        B_active = (
            np.hstack([Gamma_hat[:, j * K : (j + 1) * K] for j in active])
            if active.size
            else np.zeros((Gamma_hat.shape[0], 0))
        )

        return norms, p_hat, B_active

    def least_squares_estimate_Gamma(self, dX_check: NDArray[np.floating], dY_check: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Estimate short-run coefficient matrix Gamma via least squares.

        Parameters
        ----------
        dX_check : NDArray[np.floating]
            Residualized lag regressors.
        dY_check : NDArray[np.floating]
            Residualized differenced data.

        Returns
        -------
        NDArray[np.floating]
            Estimated short-run coefficient matrix.

        Raises
        ------
        LinAlgError
            If the Gram matrix is singular.
        ValueError
            If input dimensions are incompatible.
        
        Notes
        -----
        Used in Eq. (30) of our paper.
        """
        T_eff = self.T_eff
        cross = (1 / T_eff) * dY_check @ dX_check.T
        gram  = (1 / T_eff) * dX_check @ dX_check.T

        # Sanity checks
        if np.linalg.matrix_rank(gram) < gram.shape[0]:
            raise np.linalg.LinAlgError("Gram matrix in least squares is singular.")
        
        if dX_check.shape[1] != dY_check.shape[1]:
            raise ValueError("dX_check and dY_check must have the same number of columns.")
        
        Gamma_hat = cross @ np.linalg.inv(gram)
        return Gamma_hat
    
    def ridge_estimator_Gamma(self, dX_check: NDArray[np.floating], dY_check: NDArray[np.floating], tau_ridge: float, M: int) -> NDArray[np.floating]:
        """
        Estimate short-run coefficient matrix Gamma via ridge regression.

        Parameters
        ----------
        dX_check : NDArray[np.floating]
            Residualized lag regressors.
        dY_check : NDArray[np.floating]
            Residualized differenced data.
        tau_ridge : float
            Ridge penalty parameter.
        M : int
            Maximum lag order.

        Returns
        -------
        NDArray[np.floating]
            Ridge-regularized estimate of Gamma.

        Raises
        ------
        ValueError
            If input dimensions are incompatible.
        LinAlgError
            If ridge penalty is zero and Gram matrix is singular.

        Notes
        -----
        Used in Eq. (31) of our paper.
        """
        # Sanity check
        if dX_check.shape[1] != dY_check.shape[1]:
            raise ValueError("dX_check and dY_check must have the same number of columns.")
        
        K, T_eff = self.K, self.T_eff
        I_KM = np.eye(K*M)

        cross = (1 / T_eff) * dY_check @ dX_check.T
        gram  = (1 / T_eff) * dX_check @ dX_check.T
        Gamma_tilde = cross @ np.linalg.inv(gram + (tau_ridge / T_eff) * I_KM)
        return Gamma_tilde


