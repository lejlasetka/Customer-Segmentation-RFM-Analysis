CSE-532 Final Customer Segmentation Analysis
UCI Online Retail (December 2010–December 2011)
Executed November 21, 2026 with the actual UCI spreadsheet.

DATA CLEANING
Original invoice line items: 541,909
Exact duplicate rows: 5,268
After deduplication: 536,641
Rows with valid customer ID and date: 401,604
Positive, noncancellation purchase lines after cleaning: 392,692
Customers with positive valid purchase history: 4,338
NOTE: Earlier midpoint slides reported 397,884 positive lines; this rerun additionally removes exact duplicate lines. Do not mix counts from these analyses.

FINAL K-MEANS RESULTS
Features: Recency, Frequency, Monetary. log1p then standard scaling.
Tested k=2 through 6; k=2 highest sampled silhouette, 0.4284 on 1,500 sampled customers. Davies-Bouldin=0.8920.
Cluster 0: 1,652 customers; median recency 16 days; 6 invoices; monetary £2,077.92.
Cluster 1: 2,686 customers; median recency 95 days; 1 invoice; monetary £363.65.
PCA 2D projection shows 93.9% combined feature variance.

METHOD COMPARISON (k=2)
K-Means: silhouette 0.4284, Davies-Bouldin 0.8920
Agglomerative Ward: silhouette 0.3807, Davies-Bouldin 0.9469, ARI vs K-Means 0.6741
Gaussian mixture: silhouette 0.2732, Davies-Bouldin 1.0629, ARI vs K-Means 0.1946
Seed sensitivity ARI 0.9972–1.0. This is only random-initialization robustness, not proof of temporal stability.

INTERPRETABILITY
Lower recency means more recent activity; higher frequency and monetary value mean more frequent purchases and spending. Label clusters using observed behavioral differences, not true customer identities or motivations.
Silhouette is a measure of within-dataset separation, not classification accuracy.
Clustering cannot establish causal relations. Historical single-retailer data and exclusion of customers with missing IDs limit generalizability.

FILES
CSE532_Customer_Segmentation_Final_Analysis.ipynb: rerunnable notebook. Requires Online Retail.xlsx in same working directory.
finish_results.py: produces final comparison and interpretation charts from a generated customer-level features file.
PNG figures: ready to insert into slides/report.
CSV tables: cluster profile, method comparison, seed stability.

Source: Chen, D. (2015). Online Retail. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
