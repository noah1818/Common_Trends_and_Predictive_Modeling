import os
from typing import Tuple, Optional

import numpy as np
from numpy.linalg import inv as invert_matrix
from numpy.linalg import lstsq
from numpy.linalg import matrix_rank
from numpy.typing import NDArray  # For precise type hints of NumPy arrays, e.g. NDArray[np.floating]
import pandas as pd
from scipy.linalg import null_space
from scipy.linalg import qr
from scipy.linalg import sqrtm
from statsmodels.tsa.stattools import adfuller

class VECMModel:
    def __init__(self) -> None:
        """
        Initialize a VECM helper object with placeholders for matrices and parameters. 
        Attributes are set to None until constructed by the according builder methods.
        """
        # Core VECM data matrices
        # First differences Delta Y
        self._delta_Y: NDArray[np.floating] | None = None
        # Lagged differences Delta X
        self._delta_X: NDArray[np.floating] | None = None
        # Lagged levels Y_{t-1}
        self._Y_minus_1: NDArray[np.floating] | None = None
        # Regressor matrix for VAR
        self._Z: NDArray[np.floating] | None = None

        # Model parameters
        self._P: int | None = None   # Lag order P (> 0)
        self._r: int | None = None   # Cointegration rank (> 0)
        # The effective sample, size, T_eff = T_raw - P, T_raw is the number of columns of Y_minus_1
        self.T_eff: int | None = None
        self.K: int | None = None    # Number of endogenous variables (K > 1)

        # Johansen residual covariance (moment) matrices
        self._S_00: NDArray[np.floating] | None = None
        self._S_01: NDArray[np.floating] | None = None
        self._S_10: NDArray[np.floating] | None = None
        self._S_11: NDArray[np.floating] | None = None

    def construct_Y(self, dfs: list[pd.Series]) -> NDArray[np.floating]:
        """
        Concatenate a list of Series column-wise into a numpy array, this stacks y_t^(b) (or y_t^(a)) like seen in the paper and also sets the 
        number of varaibles in our VECM.

        Parameters
        ----------
        dfs : list of pandas.Series
            A list of Series to concatenate. All Series must have
            the same length (number of time points) for concatenation.

        Returns
        -------
        NDArray[np.floating] of shape (K, T)
            Matrix of stacked time series.

        Raises
        ------
        ValueError
            If the input list is empty or if the Series have different lengths.

        Notes
        -----
        This follows Eq. 3.2 of the thesis.
        """
        # Sanity checks
        if not dfs:
            raise ValueError("Input list of Series is empty.")

        length = dfs[0].shape[0]
        if any(s.shape[0] != length for s in dfs):
            raise ValueError(
                "All Series must have the same length for concatenation.")

        # Set K to the number of variables
        self.K = len(dfs)

        return pd.concat(dfs, axis=1).T.values

    def train_test_split_Y(self, Y: NDArray[np.floating], train_pct: float) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Split a matrix Y into train and test sets.

        Parameters
        ----------
        Y : NDArray[np.floating] of shape (K, T)
            Time series matrix.
        train_pct : float
            Fraction of observations for training. Must be in (0, 1).

        Returns
        -------
        Tuple[NDArray[np.floating], NDArray[np.floating]]
            Training set, test set.

        Raises
        ------
        ValueError
            If 'train_pct' is not between 0 and 1.
        """
        # Sanity check
        if not (0 < train_pct < 1):
            raise ValueError(
                f"train_pct must be a float in (0, 1); got {train_pct}.")

        Y_shape: int = Y.shape[1]
        train_size: int = int(Y_shape * train_pct)

        Y_train_data: pd.Series = Y[:, :train_size]
        Y_test_data: pd.Series = Y[:, train_size:]

        return Y_train_data, Y_test_data

    def adf_p_value(self, time_series: NDArray[np.floating]) -> float:
        """
        Perform the Augmented Dickey-Fuller (ADF) test for stationarity.

        This function runs the ADF test on a given time time_series and returns 
        the p-value of the test. The p-value indicates the likelihood of 
        observing the data if the null hypothesis is true.

        - Null hypothesis (H0): The time time_series has a unit root (is non-stationary).
        - Alternative hypothesis (H1): The time time_series is stationary.

        Parameters
        ----------
        time_series : NDArray[np.floating] of shape (T,)
            1D time series array.

        Returns
        -------
        float
            ADF test p-value.

        Raises
        ------
        ValueError
            If the input is not a 1D numpy array, or if fewer than 10 valid
            (non-NaN) observations are available.

        Notes
        -----
        This method returns MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
        """
        if not isinstance(time_series, np.ndarray):
            raise ValueError("Input must be a numpy array.")

        if time_series.ndim != 1:
            raise ValueError("Input array must be one-dimensional.")

        # Drop NaN values
        time_series = time_series[~np.isnan(time_series)]

        if len(time_series) < 10:
            raise ValueError("Array must contain at least 10 non-null values.")

        result = adfuller(time_series)
        return result[1]

    def build_vecm_matrices(self, Y: NDArray[np.floating]) -> None:
        """
        Construct (and store) the matrix-form variables for VECM estimation as in Luetkepohl, specifically this prepares the dependent and regressor matrices for the VECM.

        Parameters
        ----------
        Y : NDArray[np.floating] of shape (K, T)
            Endogenous variable matrix.

        Returns
        -------
        None
            This method stores the matrices as instance attributes.

        Raises
        ------
        ValueError
            If the Lag order is not a positive int.

        Notes
        -----
        This follows Eq. 3.2 of the thesis.
        """
        # Sanity checks
        if not isinstance(self._P, int) or self._P <= 0:
            raise ValueError(
                f"Lag order must be set to a positive int before building matrices; got {self._P}.")

        # First differences: shape (K, T-1)
        dY = np.diff(Y, axis=1)

        # Set the effective sample size here
        self.T_eff = dY.shape[1]

        # Alias for readability
        T_eff = self.T_eff
        P = self._P

        # Check for dimension missmatch
        if T_eff <= P:
            raise ValueError(f"Need T > P; got T={T_eff}, P={P}.")

        # Dependent variable: Delta Y_t for t = P,...,T (K, T-P)
        delta_Y = dY[:, P:]

        # Levels lagged once: Y_{t-1}, aligned with Y_t (K, T-P)
        Y_minus_1 = Y[:, P:-1]

        # Short-run dynamics regressors: Delta Y_{t-1},...,Delta Y_{t-P}
        lags = []
        for i in range(1, P+1):
            lag_i = dY[:, P-i:-i]   # each (K, T-P)
            lags.append(lag_i)
        delta_X = np.vstack(lags) if lags else np.empty((0, T_eff-P))

        # Store
        self._delta_Y = delta_Y      # (K, T-P)
        self._delta_X = delta_X      # (K*P, T-P)
        self._Y_minus_1 = Y_minus_1  # (K, T-P)

    def get_vecm_matrices(self) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Retrieve the previously constructed matrix-form variables for VECM estimation.

        This method returns the dependent and regressor matrices for the VECM,
        assuming they were built earlier using 'build_vecm_matrices'.

        Returns
        -------
        Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
            1. Matrix of current first differences Delta Y_t.
            Shape: (K, T - P).

            2. Matrix of stacked lagged differences [Delta Y_{t-1}, ..., Delta Y_{t-P}].
            Shape: (K x P, T - P).

            3. Matrix of lagged levels Y_{t-1}.
            Shape: (K, T - P).

        Raises
        ------
        ValueError
            If the matrices have not been built yet (i.e., 'build_vecm_matrices' was not called before this method).
        """
        # Sanity check
        if any(x is None for x in (self._delta_X, self._delta_Y, self._Y_minus_1)):
            raise ValueError(
                "VECM matrices not built yet. Call build_vecm_matrices first.")

        return self._delta_Y, self._delta_X, self._Y_minus_1

    def build_residual_covariances(self) -> None:
        """
        Compute the residual covariance (moment) matrices S_ij and the Johansen
        matrix S_tilde used in the Johansen cointegration rank test.

        This method requires that the VECM matrices have already been built
        using 'build_vecm_matrices(...)', and that the cointegration rank 'r'
        and sample size 'T' have been initialized.

        Steps
        -----
        1. Regress Delta Y on Delta X (short-run regressors) to obtain residuals R0.
        2. Regress Y_{t-1} on Delta X to obtain residuals R1.
        3. Form the covariance blocks:
            S_00 = (R0 R0^T) / T
            S_01 = (R0 R1^T) / T
            S_10 = (R1 R0^T) / T
            S_11 = (R1 R1^T) / T
        4. Construct the Johansen (canonical correlation) matrix:
            S_tilde = S_11^(-1/2) S_10 S_00^(-1) S_01 S_11^(-1/2)

        Raises
        ------
        ValueError
            If the VECM matrices (_delta_X, _delta_Y, _Y_minus_1) are not built,
            or if sample size attributes are not initialized.

        Notes
        -----
        This follows Eq. 3.4 of the thesis.
        """
        # Sanity checks
        if any(x is None for x in (
            getattr(self, "_delta_X", None),
            getattr(self, "_delta_Y", None),
            getattr(self, "_Y_minus_1", None),
        )):
            raise ValueError(
                "VECM matrices not built. Call build_vecm_matrices(...) first.")

        if not isinstance(self.T_eff, int) or self.T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {self.T_eff}.")

        # Aliases for readability
        dY = self._delta_Y  # (K, T_eff)
        dX = self._delta_X  # (K * P, T_eff)
        Y1 = self._Y_minus_1  # (K, T_eff)
        T_eff = self.T_eff

        # Residuals of Delta Y on Delta X
        # This is computational much more efficient then using
        # I_T - (Delta X).T (Delta X (Delta X).T)^-1 Delta X on Delta Y and Y_-1
        B0, *_ = np.linalg.lstsq(dX.T, dY.T, rcond=None)  # coefficients
        R0 = dY - (B0.T @ dX)

        # Residuals of Y_{t-1} on Delta X
        B1, *_ = np.linalg.lstsq(dX.T, Y1.T, rcond=None)
        R1 = Y1 - (B1.T @ dX)

        # compute residual covariance blocks S_ij for i,j = 0,1
        self._S_00 = (R0 @ R0.T) / T_eff  # (K, K)
        self._S_01 = (R0 @ R1.T) / T_eff  # (K, K)
        self._S_10 = (R1 @ R0.T) / T_eff  # (K, K)
        self._S_11 = (R1 @ R1.T) / T_eff  # (K, K)

        S_11_half_inv = invert_matrix(sqrtm(self._S_11))
        S_00_inv = invert_matrix(self._S_00)
        self._S_tilde = S_11_half_inv @ self._S_10 @ S_00_inv @ self._S_01 @ S_11_half_inv

    def get_johansen_matrix(self) -> NDArray[np.floating]:
        """
        Retrieve the Johansen canonical correlation matrix S_tilde.

        The matrix is defined as:
            S_tilde = S_11^(-1/2) S_10 S_00^(-1) S_01 S_11^(-1/2),
        and is constructed in 'build_residual_covariances'.

        Returns
        -------
        NDArray[np.floating] of shape (K, K)
            The Johansen canonical correlation matrix S_tilde.

        Raises
        ------
        ValueError
            If the matrix has not been built yet. Call 'build_residual_covariances'
            before accessing this method.

        Notes
        -----
        This matrix is the basis for the Johansen eigenvalue problem used
        to determine the cointegration rank in VECM estimation.
        """
        if getattr(self, "_S_tilde", None) is None:
            raise ValueError(
                "S_tilde not built yet. Call build_residual_covariances first."
            )
        return self._S_tilde

    def sort_eigenvectors(self, A: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute the eigen-decomposition of a square matrix, and return both the
        eigenvalues and eigenvectors sorted in descending order by the real part
        of the eigenvalues.

        Parameters
        ----------
        A : NDArray[np.floating]
            A 2D square array of shape (n, n).

        Returns
        -------
        NDArray[np.floating]
            1D array of eigenvalues of length n, sorted by descending real part.
        NDArray[np.floating]
            2D array of eigenvectors of shape (n, n). The i-th column corresponds
            to the eigenvector associated with eigvals_sorted[i].

        Raises
        ------
        ValueError
            If A is not a square 2D NumPy array.

        Notes
        -----
        - This method uses 'np.linalg.eig' under the hood.
        - Eigenvectors are not unique: any nonzero scaling of an eigenvector is
        also a valid eigenvector. This function only ensures a consistent
        ordering based on eigenvalues.
        """
        if not isinstance(A, np.ndarray):
            raise ValueError("Input must be a NumPy ndarray.")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input must be a square matrix (n x n).")

        eigvals, eigvecs = np.linalg.eig(A)
        order = np.argsort(eigvals.real)[::-1]  # sort by descending real part
        eigvals_sorted = eigvals[order]
        eigvecs_sorted = eigvecs[:, order]
        return eigvals_sorted, eigvecs_sorted

    def vecm_variable_estimation(self, eigvecs_sorted: Optional[NDArray[np.floating]] = None) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Estimate the VECM parameters (alpha, beta, Gamma, Sigma_u) via the Johansen procedure.

        This routine assumes the VECM design matrices have already been built and stored
        on the instance (see 'build_vecm_matrices'). Using the standard notation,
        it computes the residual covariance blocks S_ij, constructs the Johansen matrix
        S_11^(-1/2) S_10 S_00^(-1) S_01 S_11^(-1/2), and extracts the
        r leading eigenvectors to form beta. It then derives alpha, Gamma, and the residual
        covariance Sigma_u. We assume that S_00 and S_11 are invertible, e.g. R_0 and R_1 have full rank.

        Parameters
        ----------
        eigvecs_sorted : Optional[NDArray[np.floating]]
            Pre-computed eigenvectors of the Johansen matrix, sorted by descending
            real part of their eigenvalues. If None (default), they are computed internally.


        Returns
        -------
        Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
            (alpha, beta, Gamma, Sigma_u).

        Raises
        ------
        ValueError
            If the VECM matrices ('delta_X', 'delta_Y', 'Y_minus_1') are not set,
            if the residual matrices have not been built,
            if the cointegration rank 'self._r' is missing/invalid,
            if the sample size 'self.T_eff' is missing/invalid.

        Notes
        -----
        This follows Section 3.3.1 of the thesis.
        """
        # sanity checks
        if any(x is None for x in (getattr(self, "_delta_X", None), getattr(self, "_delta_Y", None), getattr(self, "_Y_minus_1", None))):
            raise ValueError(
                "VECM matrices not built. Call build_vecm_matrices(...) first.")

        # sanity checks
        if any(x is None for x in (getattr(self, "_S_00", None), getattr(self, "_S_01", None), getattr(self, "_S_10", None), getattr(self, "_S_11", None), getattr(self, "_S_tilde", None))):
            raise ValueError(
                "Residual covariance matrices not built. Call build_residual_covariances(...) first.")

        if not hasattr(self, "_r") or not isinstance(self._r, int) or self._r <= 0:
            raise ValueError(
                f"Cointegration rank 'self._r' must be a positive integer or has not been initialized yet; got {self._r}.")

        if not isinstance(self.T_eff, int) or self.T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {self.T_eff}.")

        # Aliases for readability
        dY = self._delta_Y  # (K, T_eff)
        dX = self._delta_X  # (K * P, T_eff)
        Y1 = self._Y_minus_1  # (K, T_eff)
        T_eff = self.T_eff
        r = self._r
        S_01 = self._S_01
        S_11 = self._S_11
        S_tilde = self._S_tilde

        S_11_half_inv = invert_matrix(sqrtm(S_11))

        if eigvecs_sorted is None:
            _, eigvecs_sorted = self.sort_eigenvectors(S_tilde)

        # alpha and beta
        beta = (eigvecs_sorted[:, :r].T @ S_11_half_inv).T  # (K, r)
        alpha = S_01 @ beta @ invert_matrix(beta.T @ S_11 @ beta)  # (K, r)

        # Compute Gamma
        resid_lr = dY - alpha @ (beta.T @ Y1)  # (K, T_eff)
        Gamma = resid_lr @ dX.T @ invert_matrix(dX @ dX.T)  # (K, K x P)

        # Compute Sigma_u
        u = dY - (alpha @ beta.T @ Y1) - (Gamma @ dX)  # (K, T_eff)
        Sigma_u = (u @ u.T) / T_eff  # (K, K)

        return alpha, beta, Gamma, Sigma_u

    def compute_null_space_basis(self, A: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute an orthonormal basis for the null space of a matrix.

        This method calls 'scipy.linalg.null_space' to obtain an orthogonal
        (orthonormal) matrix (or vector) Q such that A @ Q = 0. The columns of Q span the
        null space of A.

        Parameters
        ----------
        A : NDArray[np.floating]
            Input matrix (or vector) of shape (m, n).

        Returns
        -------
        numpy.ndarray
            Orthonormal basis for the null space of A.
            Shape is (n, k), where k is the dimension of the null space.

        Notes
        -----
        - If A has full column rank, the null space is empty and Q has shape (n, 0).
        - The columns of Q are orthonormal (Q.T @ Q = I).
        """
        return null_space(A)

    def compute_P_T_decomposition(self, alpha: NDArray[np.floating], beta: NDArray[np.floating], alpha_perp: NDArray[np.floating], beta_perp: NDArray[np.floating], X: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute the Gonzalo Granger P-T decomposition.

        Given the VECM long-run (cointegration) matrices 'alpha' and 'beta', and their
        orthogonal complements 'alpha_perp' and 'beta_perp', this function decomposes the
        system 'X' into:
        - a *permanent* component driven by common stochastic trends, and
        - a *transitory* component that is cointegration-stationary.

        The standard closed-form expressions are:
            A1 = alpha_perp @ (beta_perp.T @ alpha_perp)^{-1}
            A2 = beta       @ (alpha.T     @ beta)      ^{-1}

            persistent  = A1 @ beta_perp.T @ X
            transitory  = A2 @ alpha.T     @ X

        Parameters
        ----------
        alpha : NDArray[np.floating]
            Adjustment (loading) matrix alpha with shape (K, r).
        beta : NDArray[np.floating]
            Cointegration matrix beta with shape (K, r).
        alpha_perp : NDArray[np.floating]
            Orthogonal complement of alpha, shape (K, K - r), with alpha_perp.T @ alpha = 0.
        beta_perp : NDArray[np.floating]
            Orthogonal complement of beta, shape (K, K - r), with beta_perp.T  @ beta  = 0.
        X : NDArray[np.floating]
            Data matrix to decompose, shape (K, T).

        Returns
        -------
        NDArray[np.floating]
            Permanent component (common-trend driven), shape (K, T).
        NDArray[np.floating]
            Transitory/cointegration-stationary component, shape (K, T).

        Raises
        ------
        ValueError
            If dimensions of inputs are inconsistent.

        Notes
        -----
        This follows Eq. 4.42 of the thesis.
        """
        # Sanity checks
        K, r = alpha.shape
        if beta.shape != (K, r):
            raise ValueError(
                f"beta must have shape {(K, r)}, got {beta.shape} instead.")

        if alpha_perp.shape[0] != K:
            raise ValueError(
                f"alpha_perp must have shape {K, r}, got {alpha_perp.shape} instead.")

        if beta_perp.shape[0] != K:
            raise ValueError(
                f"beta_perp must have shape {K, r}, got {beta_perp.shape} instead.")

        if X.shape[0] != K:
            raise ValueError(
                f"X must have {K} rows, got {X.shape} rows instead.")

        if alpha_perp.shape != (K, K - r):
            raise ValueError(
                f"alpha_perp must have shape (K, K-r), got {alpha_perp.shape} instead.")

        if beta_perp.shape != (K, K - r):
            raise ValueError(
                f"beta_perp must have shape (K, K-r), got {beta_perp.shape} instead.")

        # Build A1, A2
        # (K-r) x (K-r)
        A1 = alpha_perp @ invert_matrix(beta_perp.T @ alpha_perp)
        A2 = beta @ invert_matrix(alpha.T @ beta)  # r x r

        # Build components
        persistent_component = A1 @ beta_perp.T @ X  # (K x T)
        transitory_component = A2 @ alpha.T @ X  # (K x T)

        return persistent_component, transitory_component

    def compute_granger_representation_matrix_XI(self, alpha_perp: NDArray[np.floating], beta_perp: NDArray[np.floating], Gamma: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the Granger representation matrix Xi as in Luetkepohl (2005).

        The Granger representation (used in the common-trend decomposition) is
            Xi = beta_per [alpha_perp.T ( I_K - Sigma_{i=1}^{P} Gamma_i ) beta_perp]^{-1} alpha_perp.T,

        where Gamma is the block-concatenation of the short-run coefficient matrices
        (each Gamma_i in R^{KxK}) and P is the lag order of the VECM.

        Parameters
        ----------
        alpha_perp : NDArray[np.floating]
            Orthonormal complement of alpha, shape (K, K - r), with alpha_perp.T @ alpha = 0.
        beta_perp : NDArray[np.floating]
            Orthonormal complement of beta, shape (K, K - r), with beta_perp.T @ beta = 0.
        Gamma : NDArray[np.floating]
            Horizontal concatenation of the short-run coefficient blocks:
            shape (K, K x P). The i-th block Gamma_i is 'Gamma[:, i*K:(i+1)*K]'
            for i = 0,..., P.

        Returns
        -------
        NDArray[np.floating]
            Granger representation matrix Xi, shape (K, K).

        Raises
        ------
        AttributeError
            If lag order self._P is not set.
        ValueError
            If input dimensions are inconsistent or the inner matrix is singular.

        Notes
        -----
        This follows Eq. 4.35 of the thesis.
        """
        # Sanity checks
        if not hasattr(self, "_P") or not isinstance(self._P, int) or self._P <= 0:
            raise AttributeError(
                "Lag order self._P must be set to a positive integer.")

        K = self.K
        if beta_perp.shape[0] != K:
            raise ValueError(
                f"beta_perp must have {K} rows, got {beta_perp.shape} instead.")

        if Gamma.shape[0] != K or Gamma.shape[1] != K * (self._P):
            raise ValueError(
                f"Gamma must have shape ({K}, {K*(self._P)}), got {Gamma.shape} instead.")

        K_minus_r = alpha_perp.shape[1]
        if beta_perp.shape[1] != K_minus_r:
            raise ValueError(
                f"alpha_perp and beta_perp must have shape (K, K-r), got {alpha_perp.shape} and {beta_perp.shape} instead.")

        # Sum of Gamma-blocks
        Gamma_blocks = [Gamma[:, i*K:(i+1)*K] for i in range(self._P)]
        Gamma_sum = np.sum(Gamma_blocks, axis=0)
        I_K = np.eye(K)

        # Inner (K-r)×(K-r) matrix
        W = alpha_perp.T @ (I_K - Gamma_sum) @ beta_perp

        if matrix_rank(W) < K_minus_r:
            raise ValueError(
                "Matrix alpha_perp.T (I_K - Gamma_sum) beta_perp is singular.")

        Xi = beta_perp @ invert_matrix(W) @ alpha_perp.T  # (K x K)

        return Xi

    def build_var_matrix(self, Y: NDArray[np.floating], d: int) -> None:
        """
        Construct the regressor (design) matrix Z of shape (1 + K*d, T - d), for a VAR(lags) model.

        Each column of Z corresponds to the stacked regressors for one observation:
        - An intercept (1.0).
        - The lagged values of Y from t-1 down to t-lags.

        Parameters
        ----------
        Y : NDArray[np.floating]
            Endogenous variable(s) matrix, shape (K, T).
        d : int
            Number of lags to include in the VAR.

        Raises
        ------
        ValueError
            If Y has wrong shape,
            if d is negative,
            if T <= d.

        Notes
        -----
        This follows Eq. 3.7 of the thesis.
        """
        # Sanity checks
        if Y.ndim != 2:
            raise ValueError("Y must be 2D with shape (K, T).")

        K, T = Y.shape
        if d < 0:
            raise ValueError("d must be nonnegative.")
        if T <= d:
            raise ValueError(f"Need T > d; got T={T}, d={d}.")

        T_eff = T - d  # usable observations

        # Collect lag blocks: each is (K, T_eff)
        lag_blocks = [Y[:, d - i: T - i] for i in range(1, d + 1)]

        # Intercept row: shape (1, T_eff)
        intercept = np.ones((1, T_eff), dtype=Y.dtype)

        # Stack vertically: (1 + K*d, T_eff)
        Z = np.vstack([intercept] + lag_blocks)

        self._Z = Z

    def get_var_matrix(self) -> NDArray[np.floating]:
        """
        Retrieve the previously constructed regressor matrix Z for VAR estimation.

        This returns the design matrix Z built earlier using
        'build_var_regressor_matrix'. Z contains an intercept row (ones) and,
        below it, the stacked lagged values of the endogenous variables.

        Returns
        -------
        NDArray[np.floating]
            The VAR regressor matrix Z with shape (K * lags + 1, T - lags),
            where K is the number of variables and T the sample length used to
            build Z. Row 0 is the intercept; the remaining rows are the lag
            blocks [Y_{t-1}, ..., Y_{t-lags}] stacked by variable.

        Raises
        ------
        ValueError
            If Z has not been built yet. Call 'build_var_regressor_matrix(...)'
            before accessing it.
        """
        # Sanity check
        if getattr(self, "_Z", None) is None:
            raise ValueError(
                "VAR matrix not built yet. Call build_var_regressor_matrix first.")

        return self._Z

    @property
    def lag(self) -> int:
        """
        Sets the lag order of the VECM.

        Returns
        -------
        int : The lag order of the VECM model.
        """
        return self._P

    @lag.setter
    def lag(self, lag: int) -> None:
        """
        Set the lag order of the VECM model.

        Parameters
        ----------
        lag : int
            The lag order (must be a positive integer).

        Raises
        ------
        ValueError
            If 'value' is not a positive integer.
        """
        # Sanity check
        if not isinstance(lag, int) or lag <= 0:
            raise ValueError("lag must be a positive integer.")

        self._P = lag

    def compute_residual_covariance(self, Y: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the estimated residual covariance matrix Sigma_u^tilde for a VAR using
        the projection residuals Y = B Z + U, where
            B^hat = Y Z^T (ZZ^T)^-1.

        Parameters
        ----------
        Y : NDArray[np.floating]
            Matrix of stacked dependent variables used in the VAR regression.

        Returns
        -------
        NDArray[np.floating]
            Estimated residual covariance matrix Sigma_u^tilde of shape (K, K), computed as
            (1 / T) * (U @ U.T).

        Raises
        ------
        ValueError
            If the sample size T is not a positive integer or is None,
            if the VAR matrix has not been built yet,
            if Y and Z are incompatable.

        Notes
        -----
        This follows Section 3.3.2 of the thesis.
        """
        # Sanity checks
        if not isinstance(self.T_eff, int) or self.T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {self.T_eff}.")

        # Alias for readability
        T_eff = self.T_eff

        Z = getattr(self, "_Z", None)
        if Z is None:
            raise ValueError(
                "VAR matrix not built yet. Call build_var_regressor_matrix first.")

        Ty = Y.shape[1]
        Tz = Z.shape[1]
        if Tz != Ty:
            raise ValueError(
                f"Incompatible shapes: Y has T={T_eff} columns but Z has T={Tz}.")

        ZZ_inv = invert_matrix(Z @ Z.T)
        Y_BZ = (Y - Y @ Z.T @ ZZ_inv @ Z)
        return (1 / T_eff) * Y_BZ @ Y_BZ.T

    def compute_HQC(self, sigma_u_hat: NDArray[np.floating], m: int, T_eff: int) -> np.floating:
        """
        Computes the Hannan-Quinn Information Criterion (HQC) for lag selection.

        The HQC is defined as:
            HQ(m) = log(det(Sigma_u_hat)) + (2 * m * K^2 * log(log(T))) / T

        where:
            - Sigma_u_hat is the estimated residual covariance matrix,
            - m is the lag order,
            - K is the number of endogenous variables,
            - T is the effective sample size.

        Parameters
        ----------
        sigma_u_hat : NDArray[np.floating]
            Estimated residual covariance matrix of shape (K, K).
            Must be symmetric positive semi-definite.
        m : int
            Candidate lag order (must be >= 1).
        T_eff : int
            The effective sample size

        Returns
        -------
        np.floating
            Value of the Hannan-Quinn information criterion (HQIC).

        Raises
        ------
        ValueError
            If 'sigma_u_hat' is not a square (K x K) matrix,
            if 'T_eff' is not a positive integer,
            if 'm' is not a positive integer.

        Notes
        -----
        This follows Eq. 3.14 of the thesis.
        """
        # Sanity checks and alias for readability
        K = self.K
        if sigma_u_hat.ndim != 2 or sigma_u_hat.shape[0] != sigma_u_hat.shape[1]:
            raise ValueError(
                f"sigma_u_hat must be a square (K x K) matrix, got shape {sigma_u_hat.shape}")

        if not isinstance(T_eff, int) or T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {T_eff}.")

        if not isinstance(m, int) or m <= 0:
            raise ValueError(
                f"m (lag order) must be a positive integer, got {m}")

        return np.log((np.linalg.det(sigma_u_hat))) + (2 * m * K**2 * np.log(np.log(T_eff))) / T_eff

    def compute_SC(self, sigma_u_hat: NDArray[np.floating], m: int, T_eff: int) -> np.floating:
        """
        Computes the Schwarz Criterion (SC, also known as BIC) for lag selection.

        The SC is defined as:
            SC(m) = log(det(Sigma_u_hat)) + (m * K^2 * log(T)) / T

        where:
            - Sigma_u_hat is the estimated residual covariance matrix,
            - m is the lag order,
            - K is the number of endogenous variables,
            - T is the effective sample size.

        Parameters
        ----------
        sigma_u_hat : NDArray[np.floating]
            Estimated residual covariance matrix of shape (K, K).
            Must be symmetric positive semi-definite.
        m : int
            Candidate lag order (must be >= 1).
        T_eff : int
            The effective sample size

        Returns
        -------
        np.floating
            Value of the Schwarz Criterion (SC, a.k.a. BIC).

        Raises
        ------
        ValueError
            If 'sigma_u_hat' is not a square (K x K) matrix,
            if 'T_eff' is not a positive integer,
            if 'm' is not a positive integer.
        
        Notes
        -----
        This follows Eq. 3.15 of the thesis.
        """
        # Sanity checks
        K = self.K
        if sigma_u_hat.ndim != 2 or sigma_u_hat.shape[0] != sigma_u_hat.shape[1]:
            raise ValueError(
                f"sigma_u_hat must be a square (K x K) matrix, got shape {sigma_u_hat.shape}")

        if not isinstance(T_eff, int) or T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {T_eff}.")

        if not isinstance(m, int) or m <= 0:
            raise ValueError(
                f"m (lag order) must be a positive integer, got {m}")

        return np.log((np.linalg.det(sigma_u_hat))) + (m * K**2 * np.log(T_eff)) / T_eff
    
    def compute_FPE(self, sigma_u_hat: NDArray[np.floating], m: int, T_eff: int) -> np.floating:
        """
        Computes the Final Prediction Error (FPE) for lag selection.

        The FPE is defined as:
            FPE(m) = ((T + m * K + 1) / (T - m * K - 1))^K * det(Sigma_u_hat)

        where:
            - Sigma_u_hat is the estimated residual covariance matrix,
            - m is the lag order,
            - K is the number of endogenous variables,
            - T is the effective sample size.

        Parameters
        ----------
        sigma_u_hat : NDArray[np.floating]
            Estimated residual covariance matrix of shape (K, K).
            Must be symmetric positive semi-definite.
        m : int
            Candidate lag order (must be >= 1).
        T_eff : int
            The effective sample sizes

        Returns
        -------
        np.floating
            Value of the Final Prediction Error (FPE).

        Raises
        ------
        ValueError
            If 'sigma_u_hat' is not a square (K x K) matrix,
            if 'T_eff' is not a positive integer,
            if 'm' is not a positive integer.

        Notes
        -----
        This follows Eq. 3.16 of the thesis.
        """
        # Sanity checks
        K = self.K
        if sigma_u_hat.ndim != 2 or sigma_u_hat.shape[0] != sigma_u_hat.shape[1]:
            raise ValueError(
                f"sigma_u_hat must be a square (K x K) matrix, got shape {sigma_u_hat.shape}")

        if not isinstance(T_eff, int) or T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {self.T_eff}.")

        if not isinstance(m, int) or m <= 0:
            raise ValueError(
                f"m (lag order) must be a positive integer, got {m}")

        return ((T_eff + K * m + 1) / (T_eff - K * m - 1))**K * (np.linalg.det(sigma_u_hat))

    def compute_AIC(self, sigma_u_hat: NDArray[np.floating], m: int, T_eff: int) -> np.floating:
        """
        Computes the Akaike Information Criterion (AIC) for lag selection.

        The AIC is defined as:
            AIC(m) = log(det(Sigma_u_hat)) + (2 * m * K^2) / T

        where:
            - Sigma_u_hat is the estimated residual covariance matrix,
            - m is the lag order,
            - K is the number of endogenous variables,
            - T is the effective sample size.

        Parameters
        ----------
        sigma_u_hat : NDArray[np.floating]
            Estimated residual covariance matrix of shape (K, K).
            Must be symmetric positive semi-definite.
        m : int
            Candidate lag order (must be >= 1).
        T_eff : int
            The effective sample size

        Returns
        -------
        np.floating
            Value of the Akaike Information Criterion (AIC).

        Raises
        ------
        ValueError
            If 'sigma_u_hat' is not a square (K x K) matrix,
            if 'T_eff' is not a positive integer,
            if 'm' is not a positive integer.

        Notes
        -----
        This follows Eq. 3.17 of the thesis.
        """
        # Sanity checks
        K = self.K
        if sigma_u_hat.ndim != 2 or sigma_u_hat.shape[0] != sigma_u_hat.shape[1]:
            raise ValueError(
                f"sigma_u_hat must be a square (K x K) matrix, got shape {sigma_u_hat.shape}")

        if not isinstance(T_eff, int) or T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {T_eff}.")

        if not isinstance(m, int) or m <= 0:
            raise ValueError(
                f"m (lag order) must be a positive integer, got {m}")

        return np.log((np.linalg.det(sigma_u_hat))) + (2 * m * K**2) / T_eff

    def compute_likelihood_ratio_test_statistic(self, r0: int, eigvals: NDArray[np.floating]) -> np.floating:
        """
        Compute the Johansen trace test statistics for cointegration rank determination.

        The trace test statistic for a null hypothesis of at most r0 cointegration
        relations is given by:

            lambda_trace(r0) = -T * sum_{i=r0+1}^K log(1 - lambda_i),

        where:
            - T is the effective sample size,
            - K is the number of endogenous variables,
            - lambda_i are the in descending order sorted eigenvalues of the Johansen eigenproblem.

        Parameters
        ----------
        r0 : int
            The null hypothesis rank (number of cointegration relations).
            Must satisfy 0 <= r0 <= K.
        eigvals : NDArray[np.floating]
            Array of eigenvalues from the Johansen procedure, shape (K,).
            Must be real and sorted in descending order.

        Returns
        -------
        np.floating
            Trace statistic lambda_trace(r).

        Raises
        ------
        ValueError
            If T is not initialized properly or a positive integer,
            if 'eigvals' is not 1D or contains invalid values,
            or if 'r0' is not a valid integer index.
        Notes
        -----
        This follows Eq. 3.19 of the thesis.
        """
        # Sanity checks
        if not isinstance(self.T_eff, int) or self.T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {self.T_eff}.")

        # Alias for readability
        T_eff = self.T_eff

        eigvals = np.asarray(eigvals, dtype=float)
        if eigvals.ndim != 1:
            raise ValueError(
                f"'eigvals' must be 1D, got shape {eigvals.shape}.")

        if np.any(eigvals != np.sort(eigvals)[::-1]):  # Descending check
            raise ValueError(
                "'eigvals' must be sorted in non-increasing order.")

        K = self.K
        if not isinstance(r0, int) or not (0 <= r0 < K):
            raise ValueError(
                f"'r0' must be an integer in [0, {K-1}], got {r0}.")

        # Compute trace statistics
        lambda_trace = -T_eff * np.sum(np.log(1 - eigvals[r0:]))

        return lambda_trace

    def compute_max_eigenvalue_ratio_test_statistic(self, r0: int, eigvals: NDArray[np.floating]) -> np.floating:
        """
        Compute Johansen's maximum eigenvalue test statistic for cointegration rank.

        It is defined as:
            lambda_max(r0) = -T * log(1 - lambda_{r0+1})

        where:
            - T is the effective sample size,
            - lambda_i are the eigenvalues from the Johansen procedure,
            sorted in descending order (lambda_1 >= ... >= lambda_K >= 0).

        Parameters
        ----------
        r0 : int
            The null hypothesis rank. Must satisfy 0 <= r0 < K.
        eigvals : NDArray[np.floating]
            1D array of eigenvalues, shape (K,). Must be real and sorted
            in descending order.
        T : int
            Sample size.

        Returns
        -------
        np.floating
            The max-eigenvalue test statistic for the given null rank r0.

        Raises
        ------
        ValueError
            If sample size T is not initialized, if 'eigvals' is not 1D,
            if not sorted in descending order,
            if 'T' is not a positive integer,
            or if 'r0' is out of bounds.

        Notes
        -----
        This follows Eq. 3.20 of the thesis.
        """
        # Sanity checks
        if not isinstance(self.T_eff, int) or self.T_eff <= 0:
            raise ValueError(
                f"Effective sample size must be set to a positive int before building residual covaraince matrices; got {self.T_eff}.")

        # Alias for readability
        T_eff = self.T_eff

        eigvals = np.asarray(eigvals, dtype=float)
        if eigvals.ndim != 1:
            raise ValueError(
                f"'eigvals' must be 1D, got shape {eigvals.shape}.")

        if np.any(eigvals != np.sort(eigvals)[::-1]):  # Descending check
            raise ValueError(
                "'eigvals' must be sorted in non-increasing order.")

        K = self.K
        if not isinstance(r0, int) or not (0 <= r0 < K):
            raise ValueError(
                f"'r0' must be an integer in [0, {K-1}], got {r0}.")

        # Compute trace statistics
        lambda_max = -T_eff * np.sum(np.log(1 - eigvals[r0]))

        return lambda_max

    def determine_cointegration_rank(self, test_stats: NDArray[np.floating], crit_vals: NDArray[np.floating]) -> int:
        """
        Select the cointegration rank based on Johansen test statistics
        (trace or max-eigenvalue) and corresponding critical values.

        The procedure compares each statistic sequentially against its
        critical value. The selected rank r* is the smallest r such that
        test_stat[r] < crit_vals[r]. If no such r is found, the full rank
        (len(test_stats)) is returned.

        Parameters
        ----------
        test_stats : NDArray[np.floating]
            Test statistics for ranks r = 0, ..., K-1.
            Must be non-negative and of length K.
        crit_vals : NDArray[np.floating]
            Critical values at the chosen significance level for each rank r.
            Must have the same length as 'test_stats'.

        Returns
        -------
        int
            The estimated cointegration rank (0 <= r^* <= K).

        Raises
        ------
        ValueError
            If the inputs have mismatched lengths, contain non-numeric values,
            or contain invalid (negative) test statistics or critical values.

        Notes
        -----
        This follows Section 3.3.3 of the thesis.
        """
        # Sanity checks
        if test_stats.shape != crit_vals.shape:
            raise ValueError(
                f"'test_stats' and 'crit_vals' must have the same shape got {test_stats.shape} vs {crit_vals.shape}.")

        if np.any(test_stats < 0) or np.any(crit_vals < 0):
            raise ValueError(
                "All test statistics and critical values must be non-negative.")

        for r, stat in enumerate(test_stats):
            if stat < crit_vals[r]:
                return r

        return len(test_stats)

    @property
    def rank(self) -> int:
        """
        Get the cointegration rank r of the VECM.

        Returns
        -------
        int
            The cointegration rank r, i.e. the number of independent
            cointegration relations in the system.
            Satisfy 0 <= r <= K, where K is the number of endogenous variables.

        Raises
        ------
        ValueError
            If the rank has not been set yet (i.e., 'self._r' is None).
        """
        # Sanity check
        if self._r is None:
            raise ValueError("Cointegration rank has not been set yet.")
        return self._r

    @rank.setter
    def rank(self, r: int) -> None:
        """
        Set the cointegration rank r of the VECM.

        Parameters
        ----------
        r : int
            The number of cointegration relations.
            Must be a non-negative integer less than or equal to K,
            where K is the number of endogenous variables.

        Raises
        ------
        ValueError
            If 'r' is not a non-negative integer between 0 and K.
        """
        # Sanity checks
        if not isinstance(r, int) or r < 0:
            raise ValueError(
                f"'r' must be a non-negative integer, got {r} instead.")

        if r > self.K:
            raise ValueError(
                f"'r' must be a between 0 and K, with K being {self.K} got {r} instead.")

        self._r = r

    def estimate_phi(self, v: NDArray[np.floating], w: NDArray[np.floating]) -> np.floating:
        """
        Estimate scalar phi that minimizes || v - phi * w ||^2.

        Parameters
        ----------
        v : NDArray[np.floating]
            Observed series (vector).
        w : NDArray[np.floating]
            Projected series (vector).

        Returns
        -------
        np.floating
            Least-squares estimate of phi.

        Raises
        ------
        ValueError
            If w is the zero vector.
        """
        numerator = np.vdot(w, v)     # <w, v>
        denominator = np.vdot(w, w)   # ||w||^2

        if denominator == 0:
            raise ValueError(
                "Cannot estimate phi: projected vector w is zero.")

        return numerator / denominator
