# Common Trends and Predictive Modeling on Tick-Based Order Book Data

This repository implements econometric and machine learning methods for analyzing lead–lag effects, cointegration, and short-run dynamics in financial time series.

It provides:
	•	Vector Error Correction Models (VECM)
	•	Standard and high-dimensional implementations.
	•	Frisch–Waugh residualization for rank and lag selection.
	•	Group Lasso–based selection procedures for rank and lag order.
	•	Multinomial Logit models for cluster-based prediction of lead–lag effects.
	•	Real-time data fetching utilities to apply models on streaming market data.


├── vecm_model_standard.py       # Standard VECM model
├── vecm_model_high_dim.py       # High-dimensional VECM (Frisch–Waugh residualization)
├── group_lasso_prox_rank.py     # Group Lasso rank selection
├── group_lasso_prox_lag.py      # Group Lasso lag selection
├── multinomial_logit_model.py   # Multinomial logit for cluster-based prediction
├── helpers.py                   # Utility functions (matrix ops, transformations)
│
├── fetch_real_time_data/        # Scripts to gather real-time market data
│
├── eval_performance.ipynb       # Notebook: evaluate model performance
├── test_standard_model.ipynb    # Notebook: standard VECM demo
├── test_highDim_model.ipynb     # Notebook: high-dimensional VECM demo
│
└── README.md                    # Project documentation
