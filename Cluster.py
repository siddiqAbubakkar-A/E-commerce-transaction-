
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Merge data
data = pd.merge(customers, transactions, on='CustomerID')

# Feature engineering
data['SignupYear'] = pd.to_datetime(data['SignupDate']).dt.year
data['SignupMonth'] = pd.to_datetime(data['SignupDate']).dt.month
data['Tenure'] = (pd.to_datetime('today') - pd.to_datetime(data['SignupDate'])).dt.days

# Aggregate transaction data
transaction_features = transactions.groupby('CustomerID').agg({
    'Amount': ['sum', 'mean'],
    'TransactionDate': ['count', 'max']
}).reset_index()
transaction_features.columns = ['CustomerID', 'TotalSpent', 'AverageTransactionValue', 'Frequency', 'LastTransactionDate']

# Merge transaction features
data = pd.merge(data, transaction_features, on='CustomerID')

# Calculate Recency
data['Recency'] = (pd.to_datetime('today') - pd.to_datetime(data['LastTransactionDate'])).dt.days

# Select features for clustering
features = ['Tenure', 'TotalSpent', 'AverageTransactionValue', 'Frequency', 'Recency']
X = data[features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_profile = data.groupby('Cluster')[features].mean()
print(cluster_profile)

# Visualize clusters (using PCA for 2D visualization)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments')
plt.show()