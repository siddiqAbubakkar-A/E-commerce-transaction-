import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the data
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge customer and product data based on transactions
merged_df = pd.merge(transactions_df, products_df, on='ProductID')
customer_product_df = pd.merge(merged_df, customers_df, on='CustomerID')

# Aggregate data to get customer-level features
customer_features = customer_product_df.groupby('CustomerID').agg({
    'Category': ['count', lambda x: '|'.join(x)],
    'Price': ['sum', 'mean']
}).reset_index()
customer_features.columns = ['CustomerID', 'TotalTransactions', 'Categories', 'TotalSpent', 'AvgSpent']

# Preprocess the data
customer_features.fillna(0, inplace=True)

# Normalize numerical features
scaler = StandardScaler()
customer_features[['TotalTransactions', 'TotalSpent', 'AvgSpent']] = scaler.fit_transform(customer_features[['TotalTransactions', 'TotalSpent', 'AvgSpent']])

# Calculate similarity matrix
similarity_matrix = cosine_similarity(customer_features[['TotalTransactions', 'TotalSpent', 'AvgSpent']])

# Function to get top 3 lookalikes
def get_top_3_lookalikes(similarity_matrix, customer_index):
    sim_scores = list(enumerate(similarity_matrix[customer_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Exclude self-similarity
    return [(customer_features.iloc[i[0]]['CustomerID'], i[1]) for i in sim_scores]

# Generate lookalike recommendations for the first 20 customers
lookalike_dict = {}
for i in range(20):
    cust_id = customer_features.iloc[i]['CustomerID']
    lookalike_dict[cust_id] = get_top_3_lookalikes(similarity_matrix, i)

# Save the results to a CSV file
lookalike_df = pd.DataFrame(list(lookalike_dict.items()), columns=['CustomerID', 'Lookalikes'])
lookalike_df.to_csv('Lookalike.csv', index=False)

# Print the results
print(lookalike_df)
