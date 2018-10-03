import ot
import pandas as pd
import time
import numpy as np
import scipy
from sklearn import preprocessing
from collections import Counter 
from sklearn import manifold
#from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pickle
import random
#from mpl_toolkits.mplot3d import Axes3D


#This is a collection that I use for various things and don't keep very clean





def lists_to_metric_specified_sublist(LL, sublist):  
    exponent = 3   #We add an a small eta and an exponent of the markov matrix to get rid of 0 transition probability 
    eta = 0.05
    LS = [set(l) for l in LL]
    LSS = [l.intersection(sublist) for l in LS] 
    labels = sublist
    BL = [] 
    M = np.zeros([len(LL),len(labels)])
    for i in range(len(LL)):
        for p in LL[i]:
            if p in labels: M[i,labels.index(p)]=1
    SM = preprocessing.normalize(M, norm='l1')   #Now it's a right stochastic matrix, ie. each row sums to 1.  
    MSM = M.transpose().dot(SM) #This matrix is now the transition probability matrix.
    MSM = preprocessing.normalize(MSM, norm='l1')
    Mexp = np.linalg.matrix_power(MSM,exponent)
    NMSM = preprocessing.normalize(MSM+eta*Mexp, norm='max')
    distance_metric = -np.log(NMSM)/2
    return(distance_metric)


def lists_to_metric(LL, top):  
    exponent = 3   #We add an a small eta and an exponent of the markov matrix to get rid of 0 transition probability 
    eta = 0.05
    flat_list = [item for sublist in LL for item in sublist] 
    flc = Counter(flat_list)
    common_rt = set([hh[0] for hh in flc.most_common(top)])
    top = len(common_rt)
    threshold = flc.most_common(top)[top-1][1]
    LS = [set(l) for l in LL]
    LSS = [l.intersection(common_rt) for l in LS] 
    BL = [] 
    labels = list(common_rt)
    M = np.zeros([len(LL),len(labels)])
    for i in range(len(LL)):
        for p in LL[i]:
            if p in labels: M[i,labels.index(p)]=1
    SM = preprocessing.normalize(M, norm='l1')   #Now it's a right stochastic matrix, ie. each row sums to 1.  
    MSM = M.transpose().dot(SM) #This matrix is now the transition probability matrix.
    MSM = preprocessing.normalize(MSM, norm='l1')
    Mexp = np.linalg.matrix_power(MSM,exponent)
    NMSM = preprocessing.normalize(MSM+eta*Mexp, norm='max')
    distance_metric = -np.log(NMSM)/2
    return(NMSM,distance_metric,labels,threshold)

def convert_hot(l, labels):                 # each user now has a binary vector indexed by retweet, with '1' if user retweeted
    cl = [k in l for k in labels]
    return(np.array([int(c) for c in cl]))
    




def get_W(v1,v2,dsq):  #vector1, vector2, and the metric to be used
    L = len(v1)
    inds=[i for i in range(L) if (v1[i]+v2[i])>0]
    lv1 = np.asfarray(v1[inds])
    if lv1.sum()==0: return(dsq.max()) 
    lv2 = np.asfarray(v2[inds])
    lv1 = lv1/lv1.sum()
    lv2 = lv2/lv2.sum()
    ldsq = dsq[:,inds][inds]
    try : mm = ot.emd(lv1,lv2,list(ldsq))
    except : return(dsq.max())  #I believe the errors are for when vecs are empty
    value = (mm*ldsq).sum()
    return(value)


def Wmetric_with_pre(S, object_labels, object_metric, threshold=6, size_cap = 500, RandomAboveThreshold = False):  #pass a series S consisting of list.  It will return a dataframe for objects that have a few elements in the list.  
    object_list = list(S)
    object_list = [set(s) for s in object_list]
    object_list = [list(s.intersection(object_labels)) for s in object_list]
    hotvecs = [convert_hot(i, object_labels) for i in object_list]
    total_selected = [a.sum() for a in hotvecs]
    indices=[i for i in range(len(S)) if total_selected[i]>threshold]
    while(len(indices)>size_cap):
        if RandomAboveThreshold:
            indices = random.sample(indices, size_cap)
            break
        threshold+=1
        indices=[i for i in indices if total_selected[i]>threshold]
    print "with initial threshold of ", threshold, " came up with ", len(indices), "accounts to compare"
    list_of_hot_vecs = [hotvecs[i] for i in indices]
    BIG = np.nanmax(object_metric)
    om = np.nan_to_num(object_metric)
    big = BIG*np.ones(om.shape)
    om = np.minimum(om,big)
    metric_on_users = [[get_W(p,q, om) for p in list_of_hot_vecs] for q in list_of_hot_vecs]
    return(metric_on_users, indices)
#I rewrote this slightly (from Wmetric) to not use dataframes as these can be slower
    




def LMdistances(landmarks, full_series, object_labels, object_metric):  #landmark is set of (numbered) indices . full_series is full list of lists.  pass the precomputed object labels and object metric.  It will return an array of distances 
    hotvecs = [convert_hot(i, object_labels) for i in list(full_series)]
    small_list = [hotvecs[i] for i in landmarks]
    metric_on_users = [[get_W(p,q, object_metric) for p in list_of_hot_vecs] for q in small_list]
    return(metric_on_users)

    
    
#small note.  The get_W works as long as one the vectors is non empty.  
    


def plot_W(Wmetric):
    from matplotlib import pyplot as plt
    fw = scipy.sparse.csgraph.floyd_warshall(Wmetric)
    sfw = (fw+fw.transpose())/2
    mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
    pos = mds.fit(sfw).embedding_
    plt.scatter(pos[:,0], pos[:,1]) 
    plt.show()


def plot_W3d(Wmetric):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    fw = scipy.sparse.csgraph.floyd_warshall(Wmetric)
    sfw = (fw+fw.transpose())/2
    mds = manifold.MDS(n_components=3, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
    pos3 = mds.fit(sfw).embedding_
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos3[:,0], pos3[:,1],pos3[:,2]) 
    plt.show()


def LM1(list1,z, metric):  #takes the list of accepted landmarks, return the distance for adding a new index z.  metric is a list of lists
    list2 = list1+[z]
    size = len(metric)
    v1 = np.zeros(size)
    v2 = np.ones(size)/size
    for ind in list2:
        v1[ind] = 1
    v1= v1/v1.sum()
    mm = ot.emd(v1,v2,metric)
    value = (mm*metric).sum()
    return(value)
   


def W_landmark(metric, k=10):
    landmarks = []
    metric = np.asarray(metric)
    s = metric.sum(axis = 1)
    landmarks +=[np.argmin(s)] 
    print "first index", np.argmin(s)
    for j in range(k):
        dists = [LM1(landmarks,z, list(metric)) for z in range(metric.shape[0])]
        dists = np.asarray(dists)
        #print dists
        maxd = np.max(dists)
        #print maxd
        for ind in landmarks:   #just stupid thing so the min doesn't choose already landmarks
            dists[ind]=maxd
        landmarks +=[np.argmin(dists)]
        print "chosen landmark index ", j, " which is label ", np.argmin(dists)
    return(landmarks)
   


def neighbors(df, name, k): #this will take a dataframe with screen_names as index, and real numbers in all other fields.  Returns knn. 
    from sklearn.metrics.pairwise import euclidean_distances 
    i = list(df.index).index(name)
    dd = df.as_matrix()
    compare = dd[i][None,:]
    dists = euclidean_distances(compare, dd)[0]
    maxd = dists.max()
    dists[i] = maxd
    minindex = np.argmin(dists)
    close =[]
    close_distances = []
    close+=[minindex]
    close_distances+= [dists[minindex]]
    for h in range(k):
        dists[minindex]=maxd
        minindex = np.argmin(dists)    
        close+=[minindex]
        close_distances+= [dists[minindex]]
        print dists[minindex]
    return [df.index[p] for p in close], close_distances


## idea in next two functions is that if i have a criterion that splits a list, I can determine how good this is at predicing retweets by measuring the relative entropy for the distruibution of the slice.   The smaller the slice, the greater the entropy, so we need something (perhaps simple as multply entropy by size?)   
## also can be used for word entropies between different wordbags

def relative_entropy(background, mu):
    if len(background)!=len(mu) : return("error - list lengths don't match")
    integr = [mu[i]*np.log(mu[i]/background[i]) for i in range(len(mu)) if mu[i]>0] 
    return(sum(integr))
        

def slice_entropy(LL, top,subsetlist):  #LL is a list of lists.  subset is a list of subset of list indices. 
    flat_list = [item for sublist in LL for item in sublist] 
    flc = Counter(flat_list)
    common_rt = set([hh[0] for hh in flc.most_common(top)])    
    weight_rt = [hh[1] for hh in flc.most_common(top)]
    total = sum(weight_rt)
    common_rt = list(common_rt)
    background_mu = np.asarray(weight_rt)/np.float(total)
    entropy_list = []
    for subset in subsetlist:
        slicelist = [LL[i] for i in subset] 
        flat_list_slice = [item for sublist in slicelist for item in sublist] 
        flc_slice = Counter(flat_list_slice)
        weight_rt_slice = [flc_slice[k] for k in common_rt]
        total_slice = sum(weight_rt_slice)
        background_mu_slice = np.asarray(weight_rt_slice)/np.float(total_slice)
        entropy_list+=[relative_entropy(background_mu, background_mu_slice)]
        print relative_entropy(background_mu, background_mu_slice)
    return(entropy_list)




def Wmetric(df7, threshold=6, top_objects=1000, size_cap = 500, RandomAboveThreshold = False):  #pass a series consisting of list.  It will return a dataframe for objects that have a few elements in the list.  
    object_list = list(df7)
    print "creating metric for ", top_objects, " most common objects..."
    object_metric = lists_to_metric(object_list, top_objects)
    labels = object_metric[2]
    print "produced metric on ", len(labels), " objects"
    hotvecs = [convert_hot(i, labels) for i in object_list]
    total_selected = [a.sum() for a in hotvecs]
    indices=[i for i in range(len(object_list)) if total_selected[i]>threshold]
    while(len(indices)>size_cap):
        if RandomAboveThreshold:
            indices = random.sample(indices, size_cap)
            break
        threshold+=1
        indices=[i for i in indices if total_selected[i]>threshold]
    print "with initial threshold of ", threshold, " came up with ", len(indices), "accounts to compare"
    list_of_hot_vecs = [hotvecs[i] for i in indices]
    BIG = np.nanmax(object_metric[1])
    om = np.nan_to_num(object_metric[1])
    big = BIG*np.ones(om.shape)
    om = np.minimum(om,big)
    metric_on_users = [[get_W(p,q, om) for p in list_of_hot_vecs] for q in list_of_hot_vecs]
    return(metric_on_users, indices)


def LMdistancesOld(landmarks, full_series, top_objects = 1200):  #It will return a dataframe for objects that have a few elements in the list.
    try: 
        df7 = full_series[landmarks] 
        object_list = list(df7)
    except: 
        object_list = [full_series[i] for i in landmarks]
    print "creating metric for ", top_objects, " most common objects..."
    object_metric = lists_to_metric(object_list, top_objects)
    labels = object_metric[2]
    object_list = list(full_series)
    hotvecs = [convert_hot(i, labels) for i in object_list]
    small_list = [hotvecs[i] for i in landmarks]
    BIG = np.nanmax(object_metric[1])
    om = np.nan_to_num(object_metric[1])
    big = BIG*np.ones(om.shape)
    om = np.minimum(om,big)
    metric_on_users = [[get_W(p,q, om) for p in hotvecs] for q in small_list]
    return(metric_on_users)

    
    

