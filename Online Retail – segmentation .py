#!/usr/bin/env python
# coding: utf-8
# In[1]:

# Overview
# Online retail is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 
# for a UK-based and registered non-store online retail. 
# The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# Business Goal
# We aim to segement the Customers based on RFM so that the company can target its customers efficiently.

# The steps are broadly divided into:
# Step 1: Reading and Understanding the Data
# Step 2: Data Cleansing
# Step 3: Data Preparation
# Step 4: Model Building
# Step 5: Final Analysis

# In[2]:

# Step 1 : Reading and Understanding Data
# import required libraries for dataframe and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
# In[3]:
# Reading the data on which analysis needs to be done
retail = pd.read_csv('OnlineRetail.csv', sep=",", encoding="ISO-8859-1", header=0)
retail.head()
# In[4]:
# shape of df
retail.shape
# In[5]:


# df info

retail.info()


# In[6]:


# df description

retail.describe()


# In[7]:


# Step 2 : Data Cleansing
# Calculating the Missing Values % contribution in DF

df_null = round(100*(retail.isnull().sum())/len(retail), 2)
df_null


# In[8]:


# Droping rows having missing values

retail = retail.dropna()
retail.shape


# In[9]:


# Changing the datatype of Customer Id as per Business understanding

retail['CustomerID'] = retail['CustomerID'].astype(str)


# In[10]:


# Step 3 : Data Preparation
# We are going to analysis the Customers based on below 3 factors:
# R (Recency): Number of days since last purchase
# F (Frequency): Number of tracsactions
# M (Monetary): Total amount of transactions (revenue contributed)

# New Attribute : Monetary

retail['Amount'] = retail['Quantity']*retail['UnitPrice']
rfm_m = retail.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()


# In[11]:


# New Attribute : Frequency

rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
rfm_f.head()


# In[12]:


# Merging the two dfs

rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm.head()


# In[13]:


# New Attribute : Recency

# Convert to datetime to proper datatype

retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')


# In[14]:


# Compute the maximum date to know the last transaction date

max_date = max(retail['InvoiceDate'])
max_date


# In[15]:


# Compute the difference between max date and transaction date

retail['Diff'] = max_date - retail['InvoiceDate']
retail.head()


# In[16]:


# Compute last transaction date to get the recency of customers

rfm_p = retail.groupby('CustomerID')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p.head()


# In[17]:


# Extract number of days only

rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()


# In[18]:


# Merge tha dataframes to get the final RFM dataframe

rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
rfm.head()


# In[19]:


# There are 2 types of outliers and we will treat outliers as it can skew our dataset
# Statistical
# Domain specific

# Outlier Analysis of Amount Frequency and Recency

attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')


# In[20]:


# Removing (statistical) outliers for Amount
Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Recency
Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Frequency
Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]


# In[21]:


# Rescaling the Attributes
# It is extremely important to rescale the variables so that they have a comparable scale.
# There are two common ways of rescaling:
# 1. Min-Max scaling
# 2. Standardisation (mean-0, sigma-1)
# Here, we will use Standardisation Scaling.


# Rescaling the attributes

rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[22]:


rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
rfm_df_scaled.head()


# In[23]:


# Step 4 : Building the Model
# K-Means Clustering
# K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.

# The algorithm works as follows:

# First we initialize k points, called means, randomly.
# We categorize each item to its closest mean and we update the mean’s coordinates, 
# which are the averages of the items categorized in that mean so far.
# We repeat the process for a given number of iterations and at the end, we have our clusters.

# k-means with some arbitrary k

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[24]:


kmeans.labels_


# In[25]:


# Finding the Optimal Number of Clusters
# Elbow Curve to get the right number of Clusters
# A fundamental step for any unsupervised algorithm is to determine the optimal number of clusters into which 
# the data may be clustered. The Elbow Method is one of the most popular methods to determine this optimal value of k.

# Elbow-curve/SSD

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)


# In[26]:


# Silhouette Analysis
# silhouette score=(p−q)/max(p,q)
 
# p  is the mean distance to the points in the nearest cluster that the data point is not a part of
# q  is the mean intra-cluster distance to all the points in its own cluster.
# The value of the silhouette score range lies between -1 to 1.
# A score closer to 1 indicates that the data point is very similar to other data points in the cluster,
# A score closer to -1 indicates that the data point is not similar to the data points in its cluster.

# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# In[27]:


# Final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)


# In[28]:


kmeans.labels_


# In[29]:


# assign the label
rfm['Cluster_Id'] = kmeans.labels_
rfm.head()


# In[30]:


# Box plot to visualize Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='Amount', data=rfm)


# In[31]:


# Box plot to visualize Cluster Id vs Frequency

sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)


# In[32]:


# Box plot to visualize Cluster Id vs Recency

sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)


# In[33]:


print('''
Step 5 : Final Analysis
Inference:
K-Means Clustering with 3 Cluster Ids

a. Customers with Cluster Id 1 are the customers with high amount of transactions as compared to other customers.
b. tomers with Cluster Id 1 are frequent buyers.
c. Customers with Cluster Id 2 are not recent buyers and hence least of importance from business point of view.
''')


# In[ ]:




