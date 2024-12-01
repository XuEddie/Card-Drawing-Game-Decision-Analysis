import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

data = pd.read_csv('data.csv')

data['expected_draws_ratio'] = data['expected_draws'] / data['guaranteed_draws']
data['available_draws_ratio'] = data['available_draws'] / data['guaranteed_draws']
data['already_draws_ratio'] = data['already_draws'] / data['guaranteed_draws']
data['total_draws_ratio'] = data['total_draws'] / data['guaranteed_draws']

features = ['expected_draws_ratio', 'available_draws_ratio', 
            'preference', 'already_draws_ratio', 'total_draws_ratio']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

colors = ['red', 'blue', 'green', 'orange', 'purple']  

plt.figure(figsize=(10, 6))
for cluster in data['cluster'].unique():
    cluster_data = data[data['cluster'] == cluster]
    plt.scatter(cluster_data['available_draws_ratio'], cluster_data['total_draws_ratio'], label=f'Cluster {cluster}')

for i, center in enumerate(cluster_centers):
    plt.scatter(center[0], center[4], s=200, c=colors[i], marker='*', edgecolors='black', label=f'Cluster {i} Center')

plt.xlabel('Already Draws Ratio (Already/Guaranteed)', fontsize=12)
plt.ylabel('Total Draws Ratio (Total/Guaranteed)', fontsize=12)
plt.title('Player Clusters: Total Draws Ratio vs. Already Draws Ratio', fontsize=14)
plt.legend()
plt.grid()

if os.path.exists("total_vs_already.png"):
    os.remove("total_vs_already.png")
plt.savefig("total_vs_already.png")
plt.show()
