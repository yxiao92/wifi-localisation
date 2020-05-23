import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_rand_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def scatter3d(data, clusters, k=2, title='3D scatter plot', save=False):
    fig = px.scatter_3d(data, 
                        x='LONGITUDE', 
                        y='LATITUDE', 
                        z='FLOOR',
                        color=clusters['c' + str(k)]['labels'].astype(str))
    fig.update_layout(
        height=700, 
        width=800,
        scene = dict(
            zaxis = dict(range=[data['FLOOR'].astype(int).min() - 1, data['FLOOR'].astype(int).max() + 1])
        ),
        legend_title='Cluster',
        title_text=title)

    fig.show()
    
    if save == True:
        fig.write_html("../visualisation/clustering/" + title + ".html")

def kmeans_evaluation(data, clusters, plot_name='building', save=False):
    n_clusters = len(clusters)
    # labels = [values['labels'] for cluster, values in clusters.items()]
    rooms = data.loc[:, 'LATITUDE'].astype(int)
    
    sse = [values['sse'] for cluster, values in clusters.items()]
    avg_score = [np.mean(values['silhouette_values']) for cluster, values in clusters.items()]
    h_score = [homogeneity_score(rooms, values['labels']) for cluster, values in clusters.items()]
    c_score = [completeness_score(rooms, values['labels']) for cluster, values in clusters.items()]
    v_measure = [v_measure_score(rooms, values['labels']) for cluster, values in clusters.items()]
    rand_idx = [adjusted_rand_score(rooms, values['labels']) for cluster, values in clusters.items()]

    fig, ((ax1, ax2), (ax3, ax4)) =  plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax1.set_title("Inertia")
    ax1.plot(sse)
    ax1.set_xticks(np.arange(0, n_clusters, 2))
    ax1.set_xticklabels(np.arange(2, n_clusters + 2, 2))
    ax1.set_xlabel("k")
    ax1.set_ylabel("Sum of squared errors")

    ax2.set_title("Average silhouette coefficient")
    ax2.plot(avg_score)
    ax2.set_xticks(np.arange(0, n_clusters, 2))
    ax2.set_xticklabels(np.arange(2, n_clusters + 2, 2))
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette coefficient")

    ax3.set_title("Completeness/homogeneity/v-measure")
    ax3.plot(c_score)
    ax3.set_xticks(np.arange(0, n_clusters, 2))
    ax3.set_xticklabels(np.arange(2, n_clusters + 2, 2))
    ax3.set_xlabel("k")
    ax3.set_ylabel("Score")

    ax3.plot(h_score)
    ax3.set_xticks(np.arange(0, n_clusters, 2))
    ax3.set_xticklabels(np.arange(2, n_clusters + 2, 2))

    ax3.plot(v_measure)
    ax3.set_xticks(np.arange(0, n_clusters, 2))
    ax3.set_xticklabels(np.arange(2, n_clusters + 2, 2))
    ax3.legend("CHV")

    ax4.set_title("Adjusted Rand index")
    ax4.plot(rand_idx)
    ax4.set_xticks(np.arange(0, n_clusters, 2))
    ax4.set_xticklabels(np.arange(2, n_clusters + 2, 2))
    ax4.set_xlabel("k")
    ax4.set_ylabel("Score")


    fig.suptitle("KMeans clustering - " + plot_name, fontsize=14)
    fig.show()
    if save == True:
        fig.savefig('../visualisation/clustering/kmeans_' + plot_name + '.png', dpi=300)