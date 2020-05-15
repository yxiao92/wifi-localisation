import pandas as pd
import numpy as np
import scipy.stats
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
# from sklearn.exceptions import FitFailedWarning
# from sklearn.exceptions import UserWarning

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def classification(data, wap, target, classifier, cv=10):
    if target == 'building':
        target = data['BUILDINGID']
    elif target == 'floor':
        target = data['FLOOR']
    elif target == 'room':
        target = data['LOC']
        
    if classifier == 'lr':
        clf = LogisticRegression(random_state=1)
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3, p=2, algorithm='kd_tree')
    elif classifier == 'svm':
        clf = SVC(kernel='linear', C=0.1, random_state=1)          

    cvs = cross_val_score(clf, wap, target, 
                          scoring=make_scorer(balanced_accuracy_score), cv=cv)

    return cvs
    # print("Average balanced accuracy: %.2f%% ± %.2f%%" % (mean_ci(cvs)[0]* 100, mean_ci(cvs)[1] * 100))

def classification_zone_in_floor(building, k, classifier, verbose=True):
    # warnings.filterwarnings(action='ignore', category=FitFailedWarning)
    warnings.filterwarnings(action='ignore')
    
    if classifier == 'logistic':
        clf = LogisticRegression(n_jobs=-1, random_state=1)
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3, p=2, algorithm='kd_tree', n_jobs=-1)
    elif classifier == 'svm':
        clf = SVC(kernel='linear', C=0.1, random_state=1)
        
    if building == 'b0':
        data, waps = df_train_b0, df_train_b0_wap
        clusters = clusters_b0
    elif building == 'b1':
        data, waps = df_train_b1, df_train_b1_wap
        clusters = clusters_b1
    elif building == 'b2':
        data, waps = df_train_b2, df_train_b2_wap
        clusters = clusters_b2
    
    cluster_labels = clusters['c' + str(k)]['labels'].reshape(-1, 1)
    floors = np.array(data['FLOOR']).reshape(-1, 1)
    
    accuracy_scores = []
    
    for f in np.unique(floors):
        cluster_idx = np.where(floors == f)[0]
        floor_clusters = cluster_labels[cluster_idx].ravel()
        floor_waps = waps[cluster_idx]
        
        # print("Cluster classification - floor [%s]" % f)
        
        cvs = cross_val_score(clf, floor_waps, floor_clusters, scoring=make_scorer(balanced_accuracy_score), cv=5)
        avg_accuracy = np.mean(cvs * 100)
        accuracy_scores.append(avg_accuracy)
        
        if verbose == True:
            for idx, accuracy in enumerate(cvs):
                print("Cross-validation accuracy - [{}]: {}%".format((idx + 1), round(accuracy * 100, 2)))
                print("Average accuracy: %.2f%%\n" % avg_accuracy)    
        
    return accuracy_scores

def classification_room_in_cluster(building, k, classifier):
    warnings.filterwarnings(action='ignore')
    if classifier == 'logistic':
        clf = LogisticRegression(n_jobs=-1, random_state=1)
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=1, p=1, algorithm='kd_tree', n_jobs=-1)
    elif classifier == 'svm':
        clf = SVC(kernel='linear', random_state=1)
        
    if building == 'b0':
        data, clusters = df_train_b0, clusters_b0
        floor_count = range(0, 4)
    elif building == 'b1':
        data, clusters = df_train_b1, clusters_b1
        floor_count = range(0, 4)
    elif building == 'b2':
        data, clusters = df_train_b2, clusters_b2
        floor_count = range(0, 5)
    
    accuracy_scores = defaultdict(dict)
    sample_count = defaultdict(dict)
    room_count = defaultdict(dict)
    
    building = data.reset_index(drop=True)
    cluster_labels = clusters['c' + str(k)]['labels']
    df_cluster_labels = pd.DataFrame({"CLUSTER": cluster_labels})
    df = pd.concat([building, df_cluster_labels], axis=1)
    
    for f in floor_count:
        floor = df.loc[df['FLOOR'] == str(f)]
        unique_cluster_labels = floor['CLUSTER'].unique()
        accuracy = defaultdict(dict)
        samples = defaultdict()
        unique_rooms = defaultdict()
        for cluster_label in unique_cluster_labels:
            rooms = floor.loc[floor['CLUSTER'] == cluster_label, 'LATITUDE'].astype(str)
            waps = StandardScaler().fit_transform(floor.loc[floor['CLUSTER'] == cluster_label, :'WAP519'])
            # samples['c' + str(cluster_label)] = len(rooms)
            samples['c' + str(cluster_label)] = waps.shape[0]
            # unique_rooms['c' + str(cluster_label)] = len(rooms)
            unique_rooms['c' + str(cluster_label)] = rooms.nunique()
            try:
                cvs = cross_val_score(clf, waps, rooms, scoring=make_scorer(balanced_accuracy_score), cv=3)
                accuracy['c'+ str(cluster_label)] = round(np.mean(cvs) * 100, 2)
            except ValueError:
                print("Floor [%d] - Cluster [%d]: No enough samples to be split." % (f, cluster_label))
                accuracy['c'+ str(cluster_label)] = np.nan
        sample_count['f' + str(f)] = samples
        room_count['f' + str(f)] = unique_rooms
        accuracy_scores['f' + str(f)] = accuracy
    
    df_sample_count = pd.DataFrame(sample_count, dtype='Int64').T
    df_sample_count = df_sample_count[np.sort(df_sample_count.columns)]
    df_room_count = pd.DataFrame(room_count, dtype='Int64').T
    # df_room_count = pd.DataFrame(room_count).T
    df_room_count = df_room_count[np.sort(df_room_count.columns)]
    df_accuracy = pd.DataFrame(accuracy_scores).T
    df_accuracy = df_accuracy[np.sort(df_accuracy.columns)]
    return df_sample_count, df_room_count, df_accuracy
    
#### Deprecated ###############################################################

"""def predict(train_data, train_wap, test_wap, target, classifier):
    if target == 'building':
        target = train_data['BUILDINGID']
    elif target == 'floor':
        target = train_data['FLOOR']
    elif target == 'room':
        target = train_data['LOC']
        
    if classifier == 'lr':
        clf = LogisticRegression(random_state=1)
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3, p=2, algorithm='kd_tree')
    elif classifier == 'svm':
        clf = SVC(kernel='linear', C=0.1, random_state=1)          

    clf.fit(train_wap, target)
    pred = clf.predict(test_wap)

    return pred"""