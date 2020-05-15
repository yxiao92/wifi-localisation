import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

def kmeans_clustering(data, n_clusters, verbose=True):
    clusters = defaultdict(dict)
    for i in range(2, n_clusters + 1):
        # np.random.seed(1)
        clustering = KMeans(n_clusters=i, random_state=1).fit(data)
        labels = clustering.predict(data)
        silhouette_vals = silhouette_samples(data, labels)
        clusters['c' + str(i)]['labels'] = labels
        clusters['c' + str(i)]['silhouette_values'] = silhouette_vals
        clusters['c' + str(i)]['sse'] = clustering.inertia_
        
        if verbose == True:
            avg_score = np.mean(silhouette_vals)
            print("For [%d] clusters, the average silhouette score is: %.3f" % (i, avg_score))
        
    return clusters

