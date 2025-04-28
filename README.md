![Machine Learning](https://img.shields.io/badge/Machine_Learning-ML-blue?logo=python)
![Unsupervised Learning](https://img.shields.io/badge/Unsupervised-Learning-blueviolet?logo=sklearn)
![Gaussian Model](https://img.shields.io/badge/Gaussian_Model-Custom_Implementation-yellowgreen)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Gaussian Anomaly Detection from Scratch (Custom Implementation)

This project is a custom-built anomaly detection system using a Gaussian probability distribution without relying on prebuilt anomaly detection models. It simulates a real-world scenario where we detect outliers or anomalies in a dataset using unsupervised learning.

The primary goal is to model the normal behavior of the dataset and identify deviations (anomalies) based on a learned probability distribution.
It is a part of my portfolio to demonstrate the ability to:

Build a full pipeline: from data generation, preprocessing, model building, hyperparameter tuning, to final evaluation.

Implement a Gaussian-based anomaly detector manually using scipy, numpy, and sklearn.

Apply cross-validation, threshold tuning, and visualize decision boundaries.

---

## Key Features
- Simulated Dataset Creation: Created synthetic blobs and injected random outliers.
- Data Preprocessing: Applied StandardScaler for feature normalization.
- Custom GaussianAnomalyDetector Class:
    - Models the dataset as a multivariate Gaussian distribution.
    - Calculates probability densities for each point.
    - Flags points below a set threshold (epsilon) as anomalies.

- Threshold Optimization:
    - Used F1-score to find the best epsilon (threshold) to maximize anomaly detection performance.

- Cross-Validation:
    - 5-fold cross-validation to evaluate stability.

- Visualization:
    - Clear contour plots showing decision boundaries and detected anomalies.
    - Separation of normal and anomaly points with color-coded plots.

- Metrics:
    - Cross-validation scores.
    - Best F1-score achieved.

---

## How it Works
1. Data Generation:
    - Create a blob of normal data points.
    - Add manually created random anomalies.

2. Preprocessing:
    - Normalize features using StandardScaler to ensure the Gaussian model fits well.

3. Training the Detector:
    - Estimate the mean (mu) and covariance matrix (sigma) of the normal points.
    - Fit a multivariate Gaussian using scipy.stats.multivariate_normal.

4. Prediction:
    - Predict probabilities of points belonging to the learned distribution.
    - Flag points with probabilities lower than the chosen epsilon as anomalies.

5. Evaluation:
    - Optimize epsilon based on F1-score (trade-off between precision and recall).
    - Visualize results: normal vs anomalous points.

---

## What You Will Learn
- Building anomaly detection from scratch.
- Working with probability distributions (Gaussian).
- Hyperparameter tuning for unsupervised learning.
- Cross-validation for unsupervised models.
- Data visualization for anomaly detection.
- Best practices for model evaluation when no labels are initially available.

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---
