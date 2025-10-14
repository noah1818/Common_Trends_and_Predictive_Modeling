# Common Trends and Predictive Modeling on Tick-Based Order Book Data

This repository contains the implementation of methods developed in the paper  
**“Common Trends and Predictive Modeling on Tick-Based Order Book Data”**  
by *Noah Boss* and *Marie-Christine Dueker*, applied to financial time series. 

It provides:

- **Vector Error Correction Models (VECM)**  
  Standard and high-dimensional implementations.
- **Frisch–Waugh residualization**  
  For rank and lag selection.
- **Group Lasso procedures**  
  For selecting cointegration rank and lag order.
- **Multinomial Logit models**  
  For cluster-based prediction of lead–lag effects.
- **Real-time data fetching utilities**  
  For streaming real-time market data from multiple exchanges.

---

## 📂 Repository Structure

```bash
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
```

---

Link: https://drive.google.com/file/d/1TbFgMp9YhpEZdqq2-EQsrPPb9z320mJc/view?usp=share_link
