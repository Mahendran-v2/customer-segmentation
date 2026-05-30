# customer-segmentation
a K-means clustering algorithm to group customers of a retail store based on their purchase history.
Customer Segmentation using K-Means Clustering
This notebook demonstrates a customer segmentation analysis using the K-Means clustering algorithm on the Mall_Customers.csv dataset. The goal is to identify distinct groups of customers based on their demographic and spending habits to enable targeted marketing strategies.

Table of Contents
Data Loading and Initial Exploration
Data Preprocessing
Feature Visualization
Optimal Cluster Determination (Elbow Method)
K-Means Clustering
Cluster Visualization
Cluster Analysis and Summary
1. Data Loading and Initial Exploration
The Mall_Customers.csv dataset is loaded, and its structure, basic information, and descriptive statistics are examined to understand the data's characteristics.

2. Data Preprocessing
The categorical 'Gender' column is converted into a numerical format using one-hot encoding (pd.get_dummies) to prepare the data for the K-Means algorithm.

3. Feature Visualization
A scatter plot visualizes the relationship between 'Annual Income (k$)' and 'Spending Score (1-100)', which are the primary features used for clustering.

4. Optimal Cluster Determination (Elbow Method)
The Elbow Method is applied to determine the optimal number of clusters (K) by plotting the Within-Cluster Sum of Squares (WCSS) against the number of clusters. An optimal K of 5 was inferred from the plot.

5. K-Means Clustering
The K-Means algorithm is applied with n_clusters=5 to segment the customers. The resulting cluster labels are added back to the DataFrame.

6. Cluster Visualization
Customer segments are visualized using scatter plots:

'Annual Income (k$)' vs. 'Spending Score (1-100)', with points colored by their assigned cluster and centroids marked.
'Age' vs. 'Spending Score (1-100)', colored by cluster, to understand age-spending patterns.
Age distribution across clusters using box plots.
Gender distribution across clusters using a stacked bar chart.
7. Cluster Analysis and Summary
A summary table (cluster_summary) provides descriptive statistics (mean, median, min, max, std) for 'Annual Income (k$)' and 'Spending Score (1-100)' for each cluster. This table is crucial for characterizing each customer segment.

Key Insights from the analysis include:

Cluster 0: Average income, average spending.
Cluster 1: High income, high spending ('Target Customers').
Cluster 2: Low income, high spending ('Careful Spenders' on specific items).
Cluster 3: High income, low spending ('Careful Spenders' or less interested in retail).
Cluster 4: Low income, low spending ('Frugal Customers

The age and gender distributions within these clusters provide further insights into the demographics of each segment, aiding in the development of targeted marketing and business strategies.
