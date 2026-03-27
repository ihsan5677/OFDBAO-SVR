# Title
Adaptive Optimization Framework for Support Vector Regression in Short-term Energy Consumption Forecasting

# Description
This project presents an adaptive optimization framework for Support Vector Regression (SVR) to improve short-term energy consumption forecasting accuracy. 
The proposed method integrates an enhanced opposition and fitness distance balance-based arithmetic optimizer with smart restart mechanism optimization algorithm for hyperparameter tuning
of the SVR.

# Dataset Information
The experiments are conducted on the REWD dataset.
Source: (https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html)

The dataset contains energy consumption data used for training and testing the model.

# Code Information
The code includes:
- Implementation of SVR (Main_OFDBAO_SVR.py)
- Evaluation metrics (RMSE, MAE, etc.)
- Proposed optimization algorithm (OFDBAO_SVR.py)

## Requirements
- Python 3.9 64bit | Qt 5.9.7 | PyQt5 5.9.2 | Windows 10
- NumPy
- Scikit-learn
- Pandas
- Matplotlib

# Usage Instructions
1. Load the dataset from the provided path by choosing proper location.
2. Run the main script file.
3. The model will automatically train and evaluate performance.
4. Results will be displayed and saved.

# Methodology
- Data loading and merging
- Data preprocessing
- Parameter optimization using OFDBAO
- Model training (SVR)
- Performance evaluation

# License and Contribution
This project is for research purposes.
Developed by Muhammad Ihsan and team.
