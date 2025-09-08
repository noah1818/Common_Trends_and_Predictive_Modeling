from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.discrete.discrete_model import MNLogit

class MultinomialLogit:
    """
    Class for modeling lead-lag and cointegration relationships 
    between two time series using multinomial logistic regression.

    Provides methods for:
    - Detecting clusters of static bid quotes.
    - Testing cointegration between series.
    - Estimating lead-lag correlations.
    - Computing cluster-based returns.
    - Generating model-ready data for classification.
    - Fitting multinomial logit models.
    """
    
    def __init__(self) -> None:  
        """Initialize the MultinomialLogit model."""  
        pass

    def _define_clusters(self, price_x: pd.Series, price_y: pd.Series) -> pd.Series:
        """
        Identify clusters of consecutive static bids in two series.

        A "cluster" is defined as a consecutive sequence of time points where
        one of the bid series (X or Y) does not change (i.e., first differences
        are zero). Each cluster is assigned an integer ID, and periods outside
        clusters are marked as NaN.

        Parameters
        ----------
        price_x : pandas.Series
            The lagging time series of bid quotes for asset X. Must be 1D and aligned with 'price_y'.
        price_y : pandas.Series
            The leading time series of bid quotes for asset Y. Must be 1D and aligned with 'price_x'.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the following columns:
            - 'price_x' : original price_x series
            - 'price_y' : original price_y series
            - 'dx' : first differences of price_x
            - 'dy' : first differences of price_y
            - 'cluster_id_X' : cluster IDs for static periods in price_x
            - 'cluster_id_Y' : cluster IDs for static periods in price_y

            Cluster IDs are integers increasing sequentially, NaN for non-static periods.

        Raises
        ------
        ValueError
            If inputs are not pandas.Series or have different lengths.

        Notes
        -----
        This follows Section 6.1 of our paper. 
        """

        # Sanity checks
        if not isinstance(price_x, pd.Series) or not isinstance(price_y, pd.Series):
            raise ValueError("Inputs must be pandas.Series objects.")
        
        if len(price_x) != len(price_y):
            raise ValueError("price_x and price_y must have the same length.")

        dx: pd.Series = price_x.diff().fillna(0)
        dy: pd.Series = price_y.diff().fillna(0)

        x_static = dx.eq(0)
        y_static = dy.eq(0)

        cluster_ids_X: list[float] = []
        cluster_ids_Y: list[float] = []


        cluster_id: int = 0
        in_cluster: bool = False

        for is_static in y_static:
            if is_static:
                if not in_cluster:
                    cluster_id += 1
                    in_cluster = True
                cluster_ids_X.append(cluster_id)
            else:
                cluster_ids_X.append(np.nan)
                in_cluster = False


        cluster_id: int = 0
        in_cluster: bool = False

        for is_static in x_static:
            if is_static:
                if not in_cluster:
                    cluster_id += 1
                    in_cluster = True
                cluster_ids_Y.append(cluster_id)
            else:
                
                cluster_ids_Y.append(np.nan)
                in_cluster = False

        return pd.DataFrame({
            "price_x": price_x,
            "price_y": price_y,
            "dx": dx,
            "dy": dy,
            "cluster_id_X": cluster_ids_X,
            "cluster_id_Y": cluster_ids_Y,
        })
    
    def test_cointegration(self, x: pd.Series, y: pd.Series, max_obs: int = 5000) -> float:
        """
        Perform the Engle-Granger cointegration test between two time series.

        This applies the bivariate Engle-Granger test (regressing x on y) and
        returns the p-value of the test for the null hypothesis of no cointegration.

        This was proposed as a method besides checking for correlation.

        Parameters
        ----------
        x : pandas.Series or numpy.ndarray
            First time series, shape (T,). Must be aligned with 'y'.
        y : pandas.Series or numpy.ndarray
            Second time series, shape (T,). Must be aligned with 'x'.
        max_obs : int, default=50000
            Maximum number of observations to use in the test. 
            Only the first 'max_obs' samples are considered.  

        Returns
        -------
        float
            p-value of the Engle-Granger cointegration test. Small values (e.g. < 0.05)
            reject the null hypothesis of no cointegration.

        Raises
        ------
        ValueError
            If inputs are not 1D or have mismatched lengths.

        Notes
        -----
            - Using values much larger than 50,000 can significantly increase computation time depending on your hardware.
        """
        # Sanity checks
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"'x' and 'y' must be 1D arrays; got shapes {x.shape} and {y.shape}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"'x' and 'y' must have the same length; got {len(x)} and {len(y)}.")

        if max_obs is not None:
            x = x[:max_obs]
            y = y[:max_obs]

        _, pvalue, _ = coint(x, y)
        return pvalue
    
    def estimate_lead_lag_correlation(self, x: pd.Series, y: pd.Series, max_obs: int = 50000, max_lag: int = 50) -> Tuple[int]:
        """
        Estimate the lead-lag relationship between two time series by computing
        cross-correlations over a range of lags.

        The procedure computes the correlation coefficient Corr(x_{t+lag}, y_t)
        for lags in [-max_lag, ..., +max_lag]. The lag with the strongest absolute
        correlation is reported along with the corresponding correlation value.

        Parameters
        ----------
        x : pandas.Series or numpy.ndarray
            First time series, shape (T,). Must be aligned with 'y'.
        y : pandas.Series or numpy.ndarray
            Second time series, shape (T,). Must be aligned with 'x'.
        max_lag : int
            Maximum number of lags to consider in each direction. Correlations
            are computed for lags -max_lag, ..., +max_lag.
        max_obs : int, default=50000
            Maximum number of observations to use. If the series is longer,
            only the first 'max_obs' points are used. Using very large values
            may slow down computation.

        Returns
        -------
        best_lag : int
            The lag with the highest absolute correlation. A positive lag means
            'x' lags behind 'y', while a negative lag means 'x' leads 'y'.
        best_corr : float
            The correlation coefficient at the best lag.

        Raises
        ------
        ValueError
            If inputs are invalid or max_lag/max_obs are not positive.

        Notes
        -----
        - Correlations are computed using numpy's 'corrcoef'.
        - This follows Section 6 of our paper.
        """
        # Sanity checks
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"'x' and 'y' must be 1D arrays; got shapes {x.shape} and {y.shape}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"'x' and 'y' must have the same length; got {len(x)} and {len(y)}.")
        if not isinstance(max_lag, int) or max_lag <= 0:
            raise ValueError(f"'max_lag' must be a positive integer; got {max_lag}.")
        if not isinstance(max_obs, int) or max_obs <= 0:
            raise ValueError(f"'max_obs' must be a positive integer; got {max_obs}.")
        max_obs = min(max_obs, len(x))

        # Truncate to max_obs
        x = x[:max_obs]
        y = y[:max_obs]

        lags = np.arange(-max_lag, max_lag + 1)
        correlations = []

        for lag in lags:
            if lag >= 0:
                corr = np.corrcoef(x[lag:], y[:len(y) - lag])[0, 1]
            else:
                corr = np.corrcoef(x[:len(x) + lag], y[-lag:])[0, 1]
            correlations.append(corr)

        correlations = np.array(correlations)
        idx = np.argmax(np.abs(correlations))
        best_lag = int(lags[idx])

        return best_lag
    
    def compute_static_cluster_returns(self, price_x: pd.Series, price_y: pd.Series) -> pd.DataFrame:
        """
        Compute per-cluster cumulative bid changes when the *other* series is static.

        This function:
        1) Builds clusters where one bid series is constant (no change) across
            consecutive timestamps (using 'define_bid_clusters).
        2) For each such cluster, sums the *intra-cluster* first differences of
            the other series, **setting the first observation inside each cluster
            to zero** (so that cluster entry does not contribute to the sum).

        Concretely:
        - 'r_ci_X': for clusters where 'price_y' is static ('cluster_id_X'), sum of
            'Delta price_x' within each cluster with the first element zeroed.
        - 'r_ci_Y: for clusters where 'price_x' is static ('cluster_id_Y'), sum of
            'Delta price_y' within each cluster with the first element zeroed.

        Parameters
        ----------
        price_x : pandas.Series
            Time series of bid quotes for asset X. Must be aligned with 'price_x'.
        price_y : pandas.Series
            Time series of bid quotes for asset Y. Must be aligned with 'price_y'.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with:
            - 'price_x', 'price_y' : original series
            - 'dx', 'dy'       : first differences
            - 'cluster_id_X'   : cluster IDs where 'price_y' is static
            - 'cluster_id_Y'   : cluster IDs where 'price_x' is static
            - 'r_ci_X'         : per-row cluster total of 'dx' (first element in each
                                'cluster_id_X' set to 0 before summing)
            - 'r_ci_Y'         : per-row cluster total of 'dy' (first element in each
                                'cluster_id_Y' set to 0 before summing)

            Rows not belonging to a cluster (IDs = NaN) will have NaN in 'r_ci_X'/'r_ci_Y'.

        Raises
        ------
        ValueError
            If inputs are not aligned 1D Series of same length.

        Notes
        -----
        This follows Section 6.1 of our paper.
        """
        # Sanity checks
        if not isinstance(price_x, pd.Series) or not isinstance(price_y, pd.Series):
            raise ValueError("'price_x' and 'price_y' must be pandas.Series.")
        if len(price_x) != len(price_y):
            raise ValueError(f"'price_x' and 'price_y' must have the same length; got {len(price_x)} and {len(price_y)}.")

        cluster_df = self._define_clusters(price_x, price_y)

        # calculate r_C_i_X
        cluster_df['prev_cluster'] = cluster_df['cluster_id_X'].shift(1)
        cluster_df.loc[cluster_df['cluster_id_X'] != cluster_df['prev_cluster'], 'dx'] = 0
        # Step 2: Group by cluster_id and sum dx
        r_ci_X = cluster_df.groupby('cluster_id_X')['dx'].sum().rename('r_ci_X')

        # Step 3: (Optional) Merge result back into original DataFrame
        cluster_df = cluster_df.merge(r_ci_X, on='cluster_id_X', how='left')
        cluster_df = cluster_df.drop({'prev_cluster'}, axis = 1)

        # Calculate r_C_i_Y
        cluster_df['prev_cluster'] = cluster_df['cluster_id_Y'].shift(1)
        cluster_df.loc[cluster_df['cluster_id_Y'] != cluster_df['prev_cluster'], 'dy'] = 0
        # Step 2: Group by cluster_id and sum dy
        r_ci_Y = cluster_df.groupby('cluster_id_Y')['dy'].sum().rename('r_ci_Y')

        # Step 3: (Optional) Merge result back into original DataFrame
        cluster_df = cluster_df.merge(r_ci_Y, on='cluster_id_Y', how='left')
        cluster_df = cluster_df.drop({'prev_cluster'}, axis = 1)

        return cluster_df
    
    def generate_model_data(self, cluster_df: pd.DataFrame, D: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature-target pairs for multinomial logistic regression.

        Uses past cluster returns to predict returns in the next cluster.

        Parameters
        ----------
        cluster_df : pandas.DataFrame
            DataFrame with cluster information from compute_static_cluster_returns.
        D : int
            Number of past clusters to use as features.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (N-D-2, 2*D+1). Includes intercept + lagged returns.
        y : np.ndarray
            Target vector of length (N-D-2), containing next-cluster returns.

        Raises
        ------
        ValueError
            If feature generation fails (e.g. missing cluster IDs).
        """
        # Define number of clusters
        N = cluster_df.cluster_id_X.dropna().unique().shape[0]
        theta_rows, next_return_cluster = [], []

        # Precompute dictionaries for cluster returns
        cluster_X_dict = cluster_df.dropna(subset=['cluster_id_X']).set_index('cluster_id_X')['r_ci_X'].to_dict()
        cluster_Y_dict = cluster_df.dropna(subset=['cluster_id_Y']).set_index('cluster_id_Y')['r_ci_Y'].to_dict()
        last_index_for_cluster_X = (
            pd.Series(cluster_df.index, index=cluster_df["cluster_id_X"])
            .dropna()
            .groupby(level=0)
            .last()
        )

        for w in range(D, N-2):
            try:
                theta_row = [1]  # Intercept
                k = last_index_for_cluster_X.get(w)
                
                # Lagged returns from last D clusters
                theta_row.extend([cluster_X_dict.get(cid, 0) for cid in range(w - D, w)])
                theta_row.extend([cluster_Y_dict.get(cid, 0) for cid in range(w - D, w)])


               # Compute next-cluster return (relative change in price_y)
                next_ret = (
                    cluster_df[cluster_df["cluster_id_X"] == w + 2]["price_y"].values[0]
                    / cluster_df.loc[: k + 1]["price_y"].values[-1]
                ) - 1

                theta_rows.append(theta_row)
                next_return_cluster.append(next_ret)
                
            except Exception as exp:
                raise ValueError(f"exception occured in generate_model_data(...): {exp}")

        return np.asarray(theta_rows), np.asarray(next_return_cluster)
    
    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 1000, method: str = "lbfgs"):
        """
        Fit a multinomial logistic regression model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Target array, shape (n_samples,). Converted to sign(y).
        maxiter : int, default=1000
            Maximum number of iterations.
        method : str, default='lbfgs'
            Optimization method for MNLogit.

        Returns
        -------
        statsmodels.discrete.discrete_model.MNLogitResults
            Fitted multinomial logit model.

        Raises
        ------
        ValueError
            If X and y have mismatched lengths.
        """
        if X.shape[0] != len(y):
            raise ValueError(f"X and y must align; got {X.shape[0]} vs {len(y)}.")

        y_train = np.sign(y)  # Convert targets to {-1, 0, 1}
        X_train = X
        model = MNLogit(y_train, X_train).fit(method = method, maxiter = maxiter)
        return model
   