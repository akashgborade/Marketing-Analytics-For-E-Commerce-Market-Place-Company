#!/usr/bin/env python
# coding: utf-8

# In[189]:


import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime


# In[190]:


## import all data

customers = pd.read_csv(r"F:\Akash data\AnalytixLabs\Python\5. Python Foundation End to End Case Study_E-Commerce Analytics Project\CUSTOMERS.csv")

sellers = pd.read_csv(r"F:\Akash data\AnalytixLabs\Python\5. Python Foundation End to End Case Study_E-Commerce Analytics Project\SELLERS.csv")

products = pd.read_csv(r"F:\Akash data\AnalytixLabs\Python\5. Python Foundation End to End Case Study_E-Commerce Analytics Project\PRODUCTS.csv")

orders = pd.read_csv(r"F:\Akash data\AnalytixLabs\Python\5. Python Foundation End to End Case Study_E-Commerce Analytics Project\ORDERS.csv")

order_items = pd.read_csv(r"F:\Akash data\AnalytixLabs\Python\5. Python Foundation End to End Case Study_E-Commerce Analytics Project\ORDER_ITEMS.csv")

order_payments = pd.read_csv(r"F:\Akash data\AnalytixLabs\Python\5. Python Foundation End to End Case Study_E-Commerce Analytics Project\ORDER_PAYMENTS.csv")

order_review_ratings = pd.read_csv(r"F:\Akash data\AnalytixLabs\Python\5. Python Foundation End to End Case Study_E-Commerce Analytics Project\ORDER_REVIEW_RATINGS.csv")


# In[191]:


## merge all data together


# In[192]:


merged_data = customers.merge(orders, on='customer_id', how='left')
merged_data = merged_data.merge(order_review_ratings, on='order_id', how='left')
merged_data = merged_data.merge(order_payments, on='order_id', how='left')
merged_data = merged_data.merge(order_items, on='order_id', how='left')
merged_data = merged_data.merge(products, on='product_id', how='left')
merged_data = merged_data.merge(sellers, on='seller_id', how='left')
merged_data = pd.merge(merged_data, gio_location, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left' )


# In[193]:


merged_data.dtypes


# In[180]:


merged_data.head()


# In[194]:


merged_data.isna().sum()


# In[195]:


merged_data.dropna(inplace=True)
print(merged_data)


# In[202]:


merged_data


# In[198]:


## define and calculate high level matrics


# In[207]:


total_revenue = merged_data['price'].sum()

total_quantity = merged_data['order_item_id'].sum()

total_products = merged_data['product_id'].nunique()

total_categories = merged_data['product_category_name'].nunique()

total_sellers = merged_data['seller_id'].nunique()

total_locations = merged_data['customer_zip_code_prefix'].nunique()

total_channels = merged_data['payment_type'].nunique()

total_payment_methods = merged_data['payment_type'].nunique()

print("Total Revenue:", total_revenue)
print("Total Quantity:", total_quantity)
print("Total Products:", total_products)
print("Total Categories:", total_categories)
print("Total Sellers:", total_sellers)
print("Total Locations:", total_locations)
print("Total Channels:", total_channels)
print("Total Payment Methods:", total_payment_methods)



# In[ ]:





# In[208]:


## new customer acquired.


# In[212]:


merged_data['order_purchase_date'] = pd.to_datetime(merged_data['order_purchase_timestamp']).dt.to_period('M')

new_customers_per_month = merged_data.groupby('order_purchase_date')['customer_unique_id'].nunique()
print("New Customers Acquired Each Month:\n", new_customers_per_month)


# In[ ]:





# In[213]:


# c. Understand the retention of customers on month on month basis


# In[217]:


merged_data['order_purchase_date'] = pd.to_datetime(merged_data['order_purchase_timestamp']).dt.to_period('M')

new_customers_per_month = merged_data.groupby('order_purchase_date')['customer_unique_id'].nunique()
print("New Customers Acquired Each Month:\n", new_customers_per_month)


# In[ ]:





# In[218]:


## d. How the revenues from existing/new customers on month on month basis


# In[226]:


merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])

merged_data['registration_date'] = pd.to_datetime(merged_data['customer_unique_id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])

merged_data['IsNewCustomer'] = merged_data['order_purchase_timestamp'].dt.to_period('M') == merged_data['registration_date'].dt.to_period('M')

revenues_by_customer_type = merged_data.groupby(['order_purchase_timestamp', 'IsNewCustomer'])['payment_value'].sum().unstack(fill_value=0)
print("Revenues from Existing/New Customers:\n", revenues_by_customer_type)


# In[ ]:





# In[228]:


## e. Understand the trends/seasonality of sales, quantity by category, location, month, 
# week, day, time, channel, payment method etc…


# In[231]:


import matplotlib.pyplot as plt

# Convert 'order_purchase_timestamp' to datetime
merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])

# Extract year and month from the 'order_purchase_timestamp'
merged_data['order_year_month'] = merged_data['order_purchase_timestamp'].dt.to_period('M')

# Group by year and month and calculate total sales
monthly_sales = merged_data.groupby('order_year_month')['payment_value'].sum()

# Plot the sales trend
plt.figure(figsize=(12, 6))
monthly_sales.index = monthly_sales.index.astype(str)  # Convert Period objects to strings
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[232]:


## over product category


# In[234]:


category_sales = merged_data.groupby('product_category_name')['payment_value'].sum()

category_sales = category_sales.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
category_sales.plot(kind='bar')
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.show()


# In[235]:


## sales by custmer state


# In[237]:


state_sales = merged_data.groupby('customer_state')['payment_value'].sum()

state_sales = state_sales.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
state_sales.plot(kind='bar')
plt.title('Sales by Customer State')
plt.xlabel('Customer State')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


# In[238]:


## f. Popular Products by month, seller, state, category


# In[240]:


product_sales = merged_data.groupby('product_id')['payment_value'].sum().reset_index()

popular_products = product_sales.sort_values(by='payment_value', ascending=False)

top_N = 10
top_popular_products = popular_products.head(top_N)

print("Top", top_N, "Popular Products:")
print(top_popular_products)


# In[ ]:





# In[241]:


# g. Popular categories by state, month


# In[244]:


merged_data['order_purchase_month'] = merged_data['order_purchase_timestamp'].dt.to_period('M')
merged_data['order_purchase_state'] = merged_data['customer_state']

popular_categories_by_state_month = merged_data.groupby(['order_purchase_month', 'order_purchase_state', 'product_category_name'])['product_category_name'].count().reset_index(name='popularity')

popular_categories_by_state_month = popular_categories_by_state_month.sort_values(by=['order_purchase_month', 'order_purchase_state', 'popularity'], ascending=[True, True, False])
popular_categories_by_state_month = popular_categories_by_state_month.drop_duplicates(subset=['order_purchase_month', 'order_purchase_state'])

print("Popular Categories by State and Month:")
print(popular_categories_by_state_month)


# In[ ]:





# In[245]:


#3 h. List top 10 most expensive products sorted by price


# In[258]:


sorted_products = merged_data.sort_values(by='price', ascending=False)

top_10_expensive_products = sorted_products['product_category_name'].head(10).reset_index(0)

print("Top 10 Most Expensive Products:")
print(top_10_expensive_products)


# In[ ]:





# In[ ]:


## 2. Performing Customers/sellers Segmentation
##   a. Divide the customers into groups based on the revenue generated
##   b. Divide the sellers into groups based on the revenue generated 


# In[262]:


## customer segmentatin

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

customer_scaler = StandardScaler()
merged_data['price_scaled'] = customer_scaler.fit_transform(merged_data[['price']])

wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(merged_data[['price_scaled']])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (Customers)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

num_clusters_customers = 3  # Choose the number of clusters based on the Elbow method
kmeans_customers = KMeans(n_clusters=num_clusters_customers, init='k-means++', random_state=42)
merged_data['customer_cluster'] = kmeans_customers.fit_predict(merged_data[['price_scaled']])

customer_cluster_centers = pd.DataFrame(customer_scaler.inverse_transform(kmeans_customers.cluster_centers_), columns=['price'])
customer_cluster_centers['customer_cluster'] = range(num_clusters_customers)

print("Cluster Centers (Average Price per Customer Cluster):")
print(customer_cluster_centers)


merged_data['customer_segment'] = merged_data['customer_cluster'].map({
    0: "Low Price",
    1: "Medium Price",
    2: "High Price",
})

# Display the customer segmentation results
print("Customer Segmentation Results:")
print(merged_data[['customer_id', 'price', 'customer_segment']])


# In[266]:


## seller segmentation

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

seller_scaler = StandardScaler()
merged_data['price_scaled'] = seller_scaler.fit_transform(merged_data[['price']])

wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(merged_data[['price_scaled']])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (Sellers)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

num_clusters_sellers = 3  # Choose the number of clusters based on the Elbow method
kmeans_sellers = KMeans(n_clusters=num_clusters_sellers, init='k-means++', random_state=42)
merged_data['seller_cluster'] = kmeans_sellers.fit_predict(merged_data[['price_scaled']])

seller_cluster_centers = pd.DataFrame(seller_scaler.inverse_transform(kmeans_sellers.cluster_centers_), columns=['price'])
seller_cluster_centers['seller_cluster'] = range(num_clusters_sellers)

print("Cluster Centers (Average Price per Seller Cluster):")
print(seller_cluster_centers)

merged_data['seller_segment'] = merged_data['seller_cluster'].map({
    0: "Low Price",
    1: "Medium Price",
    2: "High Price",
})

# Display the seller segmentation results
print("Seller Segmentation Results:")
print(merged_data[['seller_id', 'price', 'seller_segment']])


# In[ ]:





# In[269]:


## 3. Cross-Selling (Which products are selling together)


# In[270]:


pip install mlxtend


# In[283]:


from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Assuming you have a DataFrame named 'merged_data' with transaction information
# and 'order_id', 'product_id', and 'order_item_id' are the relevant columns

# Create a binary-encoded DataFrame for the products
encoded_transactions = pd.crosstab(index=merged_data['order_id'], columns=merged_data['product_id'], values=merged_data['order_item_id'], aggfunc='max').fillna(0)

# Perform Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(encoded_transactions, min_support=0.02, use_colnames=True)

# Generate association rules from frequent itemsets
association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Filter and sort association rules
top_rules = association_rules_df.sort_values(by=["lift"], ascending=False).head(10)

# Display the top combinations of products that sell together
print(top_rules)


# In[272]:


merged_data.dtypes


# In[284]:


## 4. Payment Behaviour
#a. How customers are paying?
#b. Which payment channels are used by most customers?


# In[289]:


import matplotlib.pyplot as plt

payment_counts = merged_data['payment_type'].value_counts()

plt.figure(figsize=(10, 6))
payment_counts.plot(kind='bar', color='skyblue')
plt.title('Payment Types Used by Customers')
plt.xlabel('Payment Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()


payment_percentage = (payment_counts / len(merged_data)) * 100
print(payment_percentage)


# In[288]:


payment_channel_counts = merged_data['payment_sequential'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(payment_channel_counts, labels=payment_channel_counts.index, autopct='%1.1f%%', startangle=140, colors=['gold', 'lightcoral', 'lightskyblue'])
plt.title('Payment Channels Used by Customers')
plt.axis('equal')
plt.show()


# In[ ]:





# In[290]:


## 5. Customer satisfaction towards category & product
#a. Which categories (top 10) are maximum rated & minimum rated?
#b. Which products (top10) are maximum rated & minimum rated?
#c. Average rating by location, seller, product, category, month etc.
#Etc.


# In[293]:


##a. Which categories (top 10) are maximum rated & minimum rated?

category_ratings = merged_data.groupby('product_category_name')['review_score'].mean()

top_10_max_rated_categories = category_ratings.sort_values(ascending=False).head(10)

top_10_min_rated_categories = category_ratings.sort_values(ascending=True).head(10)

print("Top 10 Maximum Rated Categories:")
print(top_10_max_rated_categories)

print("\nTop 10 Minimum Rated Categories:")
print(top_10_min_rated_categories)


# In[295]:


#b. Which products (top 10) are maximum rated & minimum rated?
product_ratings = merged_data.groupby('product_id')['review_score'].mean()

top_10_max_rated_products = product_ratings.sort_values(ascending=False).head(10)

top_10_min_rated_products = product_ratings.sort_values(ascending=True).head(10)

print("Top 10 Maximum Rated Products:")
print(top_10_max_rated_products)

print("\nTop 10 Minimum Rated Products:")
print(top_10_min_rated_products)


# In[297]:


location_ratings = merged_data.groupby('customer_city')['review_score'].mean()


location_ratings = location_ratings.sort_values(ascending=False)

print("Average Rating by Location (Top 10):")
print(location_ratings.head(10))


# In[ ]:




