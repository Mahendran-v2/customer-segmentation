import pandas as pd

# Load the dataset
df = pd.read_csv('/content/Mall_Customers.csv')

# Display the first 5 rows of the DataFrame
display(df.head())

# Display basic information about the dataset
display(df.info())

# Display descriptive statistics of the numerical columns
display(df.describe())

# Convert categorical 'Gender' column to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Display the first few rows of the encoded DataFrame to verify
display(df_encoded.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the relationship between 'Annual Income (k$)' and 'Spending Score (1-100)'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df_encoded, s=100)
plt.title('Annual Income vs. Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show()

from sklearn.cluster import KMeans

# Select the features for clustering: Annual Income and Spending Score
X = df_encoded[['Annual Income (k$)', 'Spending Score (1-100)']]

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# Apply K-Means with the optimal number of clusters (e.g., 5, typically observed from Elbow Method)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Add the cluster labels to the DataFrame
df_encoded['Cluster'] = y_kmeans

# Display the first few rows with the new 'Cluster' column
display(df_encoded.head())

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df_encoded, palette='viridis', s=100, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('Customer Segments using K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Age', data=df_encoded, hue='Cluster', palette='viridis', legend=False)
plt.title('Age Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Age')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Group by Cluster and calculate descriptive statistics for 'Annual Income (k$)' and 'Spending Score (1-100)'
cluster_summary = df_encoded.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].agg(['mean', 'median', 'min', 'max', 'std'])

# Display the summary table
display(cluster_summary)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Cluster', data=df_encoded, palette='viridis', s=100, alpha=0.8)
# Optionally, if 'Age' and 'Spending Score' were part of the features used for clustering, we could plot centroids here.
# For now, we are just visualizing the existing clusters on these two dimensions.

plt.title('Age vs. Spending Score by Customer Cluster')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

"""### Gender Distribution Across Clusters"""

gender_cluster_distribution = df_encoded.groupby(['Cluster', 'Gender_Male']).size().unstack(fill_value=0)
gender_cluster_distribution.rename(columns={True: 'Male', False: 'Female'}, inplace=True)

fig = plt.figure(figsize=(10, 6))
gender_cluster_distribution.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))
plt.title('Gender Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Gender')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


