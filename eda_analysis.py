import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Display basic information about the datasets
print("Customers Data:")
print(customers.info())
print("\nProducts Data:")
print(products.info())
print("\nTransactions Data:")
print(transactions.info())

# Convert date columns to datetime format
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Handle missing values
customers.fillna({'Region': 'Unknown'}, inplace=True)
products.dropna(subset=['Price'], inplace=True)
transactions.fillna({'TotalValue': transactions['TotalValue'].mean()}, inplace=True)

# Remove duplicates
customers.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)
transactions.drop_duplicates(inplace=True)

# Check for missing values
print("\nMissing Values in Customers Data:")
print(customers.isnull().sum())
print("\nMissing Values in Products Data:")
print(products.isnull().sum())
print("\nMissing Values in Transactions Data:")
print(transactions.isnull().sum())

# Descriptive Statistics
print("\nDescriptive Statistics for Customers Data:")
print(customers.describe(include='all'))
print("\nDescriptive Statistics for Products Data:")
print(products.describe(include='all'))
print("\nDescriptive Statistics for Transactions Data:")
print(transactions.describe())

# Customer Signup Over Time
plt.figure(figsize=(10, 6))
customers.set_index('SignupDate').resample('M').size().plot(kind='line', marker='o')
plt.title('Customer Signups Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Signups')
plt.show()

# Product Category Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=products, x='Category', palette='viridis')
plt.title('Product Category Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.show()

# Transaction Value Distribution
plt.figure(figsize=(10, 6))
sns.histplot(transactions['TotalValue'], bins=30, kde=True, color='blue')
plt.title('Transaction Value Distribution')
plt.xlabel('Total Value')
plt.ylabel('Frequency')
plt.show()

# Region-wise Transaction Distribution
region_transactions = transactions.merge(customers, on='CustomerID')['Region'].value_counts()
plt.figure(figsize=(8, 6))
region_transactions.plot(kind='pie', autopct='%1.1f%%', colors=['red', 'green', 'blue', 'orange'])
plt.title('Region-wise Transaction Distribution')
plt.ylabel('')
plt.show()

# Most Popular Products
popular_products = transactions['ProductID'].value_counts().head(10).reset_index()
popular_products = popular_products.merge(products, left_on='index', right_on='ProductID')
plt.figure(figsize=(10, 6))
sns.barplot(x=popular_products['ProductName'], y=popular_products['ProductID'], palette='coolwarm')
plt.title('Top 10 Most Popular Products')
plt.xlabel('Product Name')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.show()

# Customer-Transaction Relationship
customer_transactions = transactions['CustomerID'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(customer_transactions, bins=20, kde=True, color='purple')
plt.title('Number of Transactions per Customer')
plt.xlabel('Number of Transactions')
plt.ylabel('Frequency')
plt.show()

# Region-wise Customer Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=customers, x='Region', palette='Set2')
plt.title('Region-wise Customer Distribution')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.show()

# Price Distribution of Products
plt.figure(figsize=(10, 6))
sns.histplot(products['Price'], bins=30, kde=True, color='orange')
plt.title('Price Distribution of Products')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Correlation Analysis
plt.figure(figsize=(8, 6))
sns.scatterplot(data=transactions, x='Quantity', y='TotalValue', alpha=0.6)
plt.title('Correlation Between Quantity and TotalValue')
plt.xlabel('Quantity')
plt.ylabel('Total Value')
plt.show()
