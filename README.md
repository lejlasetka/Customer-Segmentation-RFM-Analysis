# Customer Segmentation and Purchasing Behavior Analysis

**Author:** Lejla Setka  
**Course:** CSE-532 – Python & Data Analytics  
**Semester:** Fall 2026

## Project Overview

This project analyzes customer purchasing behavior using Recency, Frequency, and Monetary (RFM) analysis and unsupervised machine learning. The objective is to identify meaningful customer segments based on historical purchasing patterns without predefined customer labels.

The project emphasizes balancing quantitative clustering performance with interpretability and practical business applications.

## Dataset

The dataset used in this project is the Online Retail dataset from the UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/352/online+retail

- Original transaction records: 541,909
- Cleaned positive purchase records: 392,692
- Unique customer profiles: 4,338
- Observation period: December 2010–December 2011

## Methodology

1. Data cleaning and duplicate removal
2. Exploratory data analysis
3. Customer-level RFM feature engineering
4. Logarithmic transformation and standardization
5. K-Means clustering with k = 2–6
6. Agglomerative clustering and Gaussian mixture model comparisons
7. Silhouette and Davies–Bouldin evaluation
8. Random initialization stability analysis using the Adjusted Rand Index
9. Principal Component Analysis (PCA) for visualization
10. Customer segment interpretation

## Final Results

| Evaluation Metric | Result |
|---|---:|
| Total customers | 4,338 |
| Selected number of clusters | 2 |
| K-Means silhouette score | 0.4284 |
| Davies–Bouldin index | 0.8920 |
| PCA explained variance | 93.9% |
| Initialization stability (ARI) | 0.9972–1.0000 |

### Customer Segment Profiles

| Metric | More Engaged | Less Engaged |
|---|---:|---:|
| Customers | 1,652 | 2
