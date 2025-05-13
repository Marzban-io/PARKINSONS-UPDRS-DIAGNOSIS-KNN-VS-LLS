# Parkinson's Disease UPDRS Regression Analysis

This project explores different regression techniques to predict the Total Unified Parkinson's Disease Rating Scale (UPDRS) score using a dataset of voice measurements from individuals with Parkinson's disease. The primary goal is to compare a K-Nearest Neighbors based Linear Least Squares (K-NN LLS) approach with a standard Linear Least Squares (LLS) model.

## Main Goal

The main objective is to investigate whether a locally adaptive regression model (K-NN LLS), which tailors its predictions based on the nearest neighbors of a given test point, can outperform a global standard LLS model in predicting the `total_UPDRS` score. The validation set plays a crucial role in tuning the hyperparameter K for the K-NN LLS approach.

## Project Overview

The Unified Parkinson's Disease Rating Scale (UPDRS) is a standard scale used to track the progression of Parkinson's disease. This project aims to build regression models that can estimate the `total_UPDRS` score based on a set of 19 vocal features extracted from voice recordings.

The core of this laboratory exercise involves:
1.  **Data Preparation:** Loading the Parkinson's dataset, selecting relevant features, shuffling the data, and splitting it into training (50%), validation (25%), and test (25%) sets. Normalization is performed based on the statistics of the true training set.
2.  **K-Nearest Neighbors LLS (K-NN LLS):**
    *   For each point in the validation set (and later, the test set), its K nearest neighbors from the true training set are identified.
    *   A local Linear Least Squares (LLS) model (specifically, ridge regression for stability) is trained using only these K neighbors.
    *   The optimal value of K (`K_opt`) is determined by evaluating the Mean Squared Error (MSE) on the validation set for different K values.
3.  **Standard Linear Least Squares (LLS):**
    *   A global LLS model (ridge regression) is trained using a larger portion of the data (75% - combining the true training and validation sets).
    *   Predictions are made on the test set.
4.  **Model Evaluation and Comparison:** The performance of the K-NN LLS model (with `K_opt`) and the standard LLS model are evaluated and compared on the test set using metrics such as Mean Squared Error (MSE), Mean Error, Standard Deviation of Error, Correlation Coefficient, and R-squared.

## Key Libraries Used

*   **NumPy:** For numerical operations, especially array manipulations and linear algebra.
*   **Pandas:** For data loading (CSV file), data manipulation, and creating DataFrames.
*   **Matplotlib:** For creating static, animated, and interactive visualizations (e.g., scatter plots, histograms, MSE vs. K plots).
*   **SciPy:** Specifically, `scipy.spatial.distance.cdist` is used for efficiently calculating the distance matrix between sets of points, crucial for the K-NN 

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wdZXMTW6cRrquoB0BfWGQHZcSE_gRJZ6?usp=sharing)
