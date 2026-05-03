import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

# Paths to your data and output directories
data_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Customer_Segmentation_Using_K-Means_Clustering/data/rfm_clusters.csv'
output_dir = '/Users/paigeleeseberg/Downloads/Python-Projects/Customer_Segmentation_Using_K-Means_Clustering/results'

# Load the dataset
df_rfm = pd.read_csv(data_path)

# Features for clustering
features = ['Recency_scaled', 'Frequency_scaled', 'MonetaryValue_scaled']
X = df_rfm[features]

# Determine the optimal number of clusters using the Elbow Method (Optional)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method results (Optional)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig(os.path.join(output_dir, 'elbow_method.png'))
plt.show()

# Function to fit KMeans and print statistics for different cluster counts
def fit_kmeans_and_print_stats(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    df_rfm[f'Cluster_{n_clusters}'] = kmeans.fit_predict(X)

    # Save the RFM data with cluster labels
    output_file = os.path.join(output_dir, f'rfm_clusters_with_labels_{n_clusters}_clusters.csv')
    df_rfm.to_csv(output_file, index=False)
    print(f"RFM data with cluster labels saved to '{output_file}'.")

    # Print cluster sizes
    cluster_sizes = df_rfm[f'Cluster_{n_clusters}'].value_counts().sort_index()
    print(f"\nCluster Sizes ({n_clusters} clusters):")
    print(cluster_sizes)
    
    # Print summary statistics for each cluster
    summary_stats = df_rfm.groupby(f'Cluster_{n_clusters}').agg(['mean', 'median', 'std'])
    print(f"\nSummary Statistics for Each Cluster ({n_clusters} clusters):")
    print(summary_stats)

    # Print cluster centroids
    centroids = kmeans.cluster_centers_
    print(f"\nCluster Centroids ({n_clusters} clusters):")
    print(pd.DataFrame(centroids, columns=features))

    # Print Silhouette Score
    silhouette_avg = silhouette_score(X, df_rfm[f'Cluster_{n_clusters}'])
    print(f"\nSilhouette Score for {n_clusters} clusters: {silhouette_avg}")

    # Print Davies-Bouldin Index
    davies_bouldin_avg = davies_bouldin_score(X, df_rfm[f'Cluster_{n_clusters}'])
    print(f"Davies-Bouldin Index for {n_clusters} clusters: {davies_bouldin_avg}")

    # Print Within-Cluster Sum of Squares (WCSS)
    wcss = kmeans.inertia_
    print(f"Within-Cluster Sum of Squares (WCSS) for {n_clusters} clusters: {wcss}")

# Fit KMeans and print statistics for 3, 6, and 9 clusters
for n_clusters in [3, 6, 9]:
    fit_kmeans_and_print_stats(n_clusters)

# Plotting and saving scatter plots with black outline for each point
def plot_and_save_scatter(x_col, y_col, title, filename):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns
    for i, num_clusters in enumerate([3, 6, 9]):
        scatter = axs[i].scatter(df_rfm[x_col], df_rfm[y_col], c=df_rfm[f'Cluster_{num_clusters}'], cmap='viridis', alpha=0.6, edgecolor='black')
        axs[i].set_title(f'{title} (Clusters: {num_clusters})')
        axs[i].set_xlabel(x_col)
        axs[i].set_ylabel(y_col)
        fig.colorbar(scatter, ax=axs[i], label='Cluster')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{x_col}_vs_{y_col}_clusters_comparison.png'))
    plt.show()

# Recency vs Frequency
plot_and_save_scatter('Recency_scaled', 'Frequency_scaled', 'Recency vs Frequency by Cluster', 'recency_vs_frequency')

# Recency vs Monetary Value
plot_and_save_scatter('Recency_scaled', 'MonetaryValue_scaled', 'Recency vs Monetary Value by Cluster', 'recency_vs_monetary_value')

# Frequency vs Monetary Value
plot_and_save_scatter('Frequency_scaled', 'MonetaryValue_scaled', 'Frequency vs Monetary Value by Cluster', 'frequency_vs_monetary_value')

# Plot cluster sizes
plt.figure(figsize=(10, 6))
for num_clusters in [3, 6, 9]:
    cluster_sizes = df_rfm[f'Cluster_{num_clusters}'].value_counts().sort_index()
    plt.plot(cluster_sizes.index, cluster_sizes.values, marker='o', label=f'{num_clusters} Clusters')
plt.title('Cluster Sizes for Different Number of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.legend()
plt.savefig(os.path.join(output_dir, 'cluster_sizes_comparison.png'))
plt.show()

# Examine distribution of scaled features
def plot_feature_distribution(feature, filename):
    plt.figure(figsize=(10, 6))
    df_rfm[feature].hist(bins=30, edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

plot_feature_distribution('Recency_scaled', 'recency_distribution.png')
plot_feature_distribution('Frequency_scaled', 'frequency_distribution.png')
plot_feature_distribution('MonetaryValue_scaled', 'monetary_value_distribution.png')
