#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# In[12]:


data = pd.read_csv("segmentation data.csv")


# In[5]:


get_ipython().system('pip install scikit-learn-extra')


# In[13]:


print(data.head())


# In[14]:


print(data.describe())


# In[15]:


print(data.isnull().sum())


# In[16]:


data.dropna(inplace=True)


# In[17]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


# In[18]:


def find_optimal_k_kmeans(data, max_k):
    distortions = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal K in KMeans')
    plt.show()


# In[19]:


find_optimal_k_kmeans(scaled_data, 10)


# In[20]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(scaled_data)
data['KMeans_Cluster'] = kmeans.labels_

# Visualize clusters using two features
plt.scatter(data['Age'], data['Income'], c=data['KMeans_Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('KMeans Clustering')
plt.show()


# In[21]:


def find_optimal_k_kmedoids(data, max_k):
    silhouette_scores = []
    for i in range(2, max_k + 1):
        kmedoids = KMedoids(n_clusters=i, random_state=42)
        kmedoids.fit(data)
        silhouette_scores.append(silhouette_score(data, kmedoids.labels_))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Elbow Method for Optimal K in KMedoids')
    plt.show()


# In[22]:


find_optimal_k_kmedoids(scaled_data, 10)


# In[23]:


kmedoids = KMedoids(n_clusters=8, random_state=42)
kmedoids.fit(scaled_data)
data['KMedoids_Cluster'] = kmedoids.labels_

# Visualize clusters using two features
plt.scatter(data['Age'], data['Income'], c=data['KMedoids_Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('KMedoids Clustering')
plt.show()


# In[ ]:




