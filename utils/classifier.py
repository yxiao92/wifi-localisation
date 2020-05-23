import pandas as pd
import numpy as np
import scipy.stats
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
from sklearn.model_selection import GridSearchCV
# from sklearn.exceptions import FitFailedWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", UserWarning)

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def classification(data, wap, target, classifier, cv=10):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
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

def tune_parameter(data, wap, target, classifier, cv=10):
    if target == 'building':
        target = data['BUILDINGID']
    elif target == 'floor':
        target = data['FLOOR']
    elif target == 'room':
        target = data['LOC']
    
    if classifier == 'lr':
        clf = LogisticRegression(random_state=1)
        params = {'penalty': ['l1', 'l2', 'none'],
                  'C':       [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    elif classifier == 'knn':
        clf = KNeighborsClassifier(algorithm='kd_tree')
        params = {'n_neighbors': [1, 3],
                  'p':           [1, 2]}
    elif classifier == 'svm':
        clf = SVC(kernel='linear', random_state=1)   
        params = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    
    gscv = GridSearchCV(clf, params, cv=cv).fit(wap, target)
    
    print("BEST PARAMS:", gscv.best_params_)
    top_param = np.argmin(gscv.cv_results_['rank_test_score'])
    splits = ['split' + str(i) + '_test_score' for i in range(10)]
    scores = [gscv.cv_results_[i][top_param] for i in splits]
    
    return gscv.best_estimator_, scores

def classification_room_in_cluster(data, clusters, k, classifier, cv=10):

    warnings.filterwarnings(action='ignore')
    
    if classifier == 'lr':
        clf = LogisticRegression(random_state=1)
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3, p=2, algorithm='kd_tree')
    elif classifier == 'svm':
        clf = SVC(kernel='linear', C=0.1, random_state=1)
    
    accuracy_scores = defaultdict(dict)
    sample_count = defaultdict(dict)
    room_count = defaultdict(dict)
    floor_count = data['FLOOR'].nunique()

    building = data.reset_index(drop=True)
    cluster_labels = clusters['c' + str(k)]['labels']
    df_cluster_labels = pd.DataFrame({"CLUSTER": cluster_labels})
    df = pd.concat([building, df_cluster_labels], axis=1)
    
    for f in range(floor_count):
        floor = df.loc[df['FLOOR'] == f]
        unique_cluster_labels = floor['CLUSTER'].unique()
        accuracy = defaultdict(dict)
        samples = defaultdict()
        unique_rooms = defaultdict()
        for cluster_label in unique_cluster_labels:
            rooms = floor.loc[floor['CLUSTER'] == cluster_label, 'LOC']
            waps = StandardScaler().fit_transform(
                floor.loc[floor['CLUSTER'] == cluster_label, :'WAP519']
            )
            # samples['c' + str(cluster_label)] = len(rooms)
            samples['c' + str(cluster_label)] = waps.shape[0]
            # unique_rooms['c' + str(cluster_label)] = len(rooms)
            unique_rooms['c' + str(cluster_label)] = rooms.nunique()
            try:
                cvs = cross_val_score(clf, waps, rooms, 
                        scoring=make_scorer(balanced_accuracy_score), cv=cv)
                accuracy['c'+ str(cluster_label)] = mean_ci(cvs)
                
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

def tune_room_in_cluster(data, clusters, k, classifier, cv=10):
    warnings.filterwarnings(action='ignore')

    if classifier == 'lr':
        clf = LogisticRegression(random_state=1)
        params = {'penalty': ['l1', 'l2', 'none'],
                  'C':       [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    elif classifier == 'knn':
        clf = KNeighborsClassifier(algorithm='kd_tree')
        params = {'n_neighbors': [1, 3],
                  'p':           [1, 2]}
    elif classifier == 'svm':
        clf = SVC(kernel='linear', random_state=1)   
        params = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    
    floor_count = data['FLOOR'].nunique()
    building = data.reset_index(drop=True)
    cluster_labels = clusters['c' + str(k)]['labels']
    df_cluster_labels = pd.DataFrame({"CLUSTER": cluster_labels})
    df = pd.concat([building, df_cluster_labels], axis=1)
    
    # gscv_results = defaultdict(lambda: defaultdict(dict))
    sample_count = defaultdict(dict)
    room_count =  defaultdict(dict)
    best_params = defaultdict(dict)
    best_scores = defaultdict(dict)

    for f in range(floor_count):
        floor = df.loc[df['FLOOR'] ==  f]
        unique_cluster_labels = floor['CLUSTER'].unique()
        # best_params = defaultdict()
        # best_scores = defaultdict()
        for cluster_label in unique_cluster_labels:
            rooms = floor.loc[floor['CLUSTER'] == cluster_label, 'LOC']
            waps = StandardScaler().fit_transform(
                floor.loc[floor['CLUSTER'] == cluster_label, :'WAP519']
            )
            # sample_count = waps.shape[0]
            # room_count = rooms.nunique()
            try:
                gscv = GridSearchCV(clf, params, cv=cv).fit(waps, rooms)
                top_param = np.argmin(gscv.cv_results_['rank_test_score'])
                splits = ['split' + str(i) + '_test_score' for i in range(10)]
                scores = [gscv.cv_results_[i][top_param] for i in splits]
                sample_count['f' + str(f)][int(cluster_label)] = waps.shape[0]
                room_count['f' + str(f)][int(cluster_label)] = rooms.nunique()
                best_params['f' + str(f)][int(cluster_label)] = gscv.best_params_
                mean, ci = mean_ci(scores)
                best_scores['f' + str(f)][int(cluster_label)] = \
                    str(round(mean * 100, 2)) + '% ± ' + str(round(ci * 100, 2)) + '%'
                # gscv_results['f' + str(f)][int(cluster_label)]['room_count'] = room_count
                # gscv_results['f' + str(f)][int(cluster_label)]['best_params'] = gscv.best_params_
                # gscv_results['f' + str(f)][int(cluster_label)]['best_scores'] = scores
                # best_params[int(cluster_label)] = gscv.best_params_
                # best_scores[int(cluster_label)] = scores
            except ValueError:
                # gscv_results['f' + str(f)][int(cluster_label)]['sample_count'] = sample_count
                # gscv_results['f' + str(f)][int(cluster_label)]['room_count'] = room_count
                # gscv_results['f' + str(f)][int(cluster_label)]['best_params'] = np.nan
                # gscv_results['f' + str(f)][int(cluster_label)]['best_scores'] = np.nan
                sample_count['f' + str(f)][int(cluster_label)] = waps.shape[0]
                room_count['f' + str(f)][int(cluster_label)] = rooms.nunique()
                best_params['f' + str(f)][int(cluster_label)] = np.nan
                best_scores['f' + str(f)][int(cluster_label)] = np.nan
    # return gscv_results 
    df_sample_count = pd.DataFrame.from_dict(sample_count, orient='index').sort_index(axis=1)
    df_room_count = pd.DataFrame.from_dict(room_count, orient='index').sort_index(axis=1)
    df_best_params = pd.DataFrame.from_dict(best_params, orient='index').sort_index(axis=1)
    df_best_scores = pd.DataFrame.from_dict(best_scores, orient='index').sort_index(axis=1)

    return  df_sample_count, df_room_count, df_best_params, df_best_scores
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