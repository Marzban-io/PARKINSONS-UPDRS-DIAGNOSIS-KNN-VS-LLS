# Parkinson's UPDRS Prediction: Comparing Regression Approaches

This project explores and compares two different regression approaches for predicting the total Unified Parkinson's Disease Rating Scale (UPDRS) score using the Parkinsons Telemonitoring Data Set from the UCI Machine Learning Repository.

## Dataset

The dataset used is `parkinsons_updrs_av.csv`. It contains various speech signal processing features and demographic data for individuals, along with their motor UPDRS and total UPDRS scores.
- **Source:** [UCI Machine Learning Repository: Parkinsons Telemonitoring Data Set](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
- **Citation:** A Tsanas, MA Little, PE McSharry, LO Ramig (2009)
'Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests',
IEEE Transactions on Biomedical Engineering.

**Ensure the `parkinsons_updrs_av.csv` file is present in the same directory as the Python script for the code to run correctly.**

## Project Goal

The primary goal is to implement and evaluate two distinct data preprocessing and modeling strategies for predicting the `total_UPDRS` score. The performance of these approaches is compared based on Mean Squared Error (MSE) and R-squared (R²) metrics on a held-out test set.

## Approaches Compared

1.  **Approach 1: K-NN Local Ridge Regression**
    *   **Preprocessing:**
        *   Data is shuffled.
        *   Normalization (mean/std scaling) is applied to all numeric features based on the training set statistics.
        *   Target variable (`total_UPDRS`) is also normalized using these global training set statistics.
        *   Specific features (`subject#`, `Jitter:DDP`, `Shimmer:DDA`, and the target itself) are dropped *after* normalization to create the feature set.
    *   **Modeling:**
        *   A K-Nearest Neighbors (K-NN) like approach is used. For each point in the validation/test set:
            *   Its `K` nearest neighbors are found in the training set.
            *   A local Ridge Regression model is trained using only these `K` neighbors.
            *   This local model then predicts the target for the point.
        *   The optimal value of `K` is determined by evaluating MSE on a validation set.
    *   **Evaluation:** Performance is measured on a test set using the optimized `K`.

2.  **Approach 2: Global Ridge Regression**
    *   **Preprocessing:**
        *   An explicit list of pre-selected features is used.
        *   Rows with any NaN values in the selected columns are dropped.
        *   Data is shuffled.
        *   Features are normalized using the mean and standard deviation from the training set's features.
        *   The target variable (`total_UPDRS`) is normalized separately using its own mean and standard deviation from the training set.
    *   **Modeling:**
        *   A single, global Ridge Regression model is trained on the entire (normalized) training set.
    *   **Evaluation:** Performance is measured on a test set.

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wdZXMTW6cRrquoB0BfWGQHZcSE_gRJZ6?usp=sharing)
