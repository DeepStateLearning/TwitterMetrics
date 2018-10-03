import pandas as pd
from collections import Counter
import time
from twitter import *
import pickle


#step 0, which I don't include, is to generate 1000's of names.   This was done by doing a geographic search for several locations around the US, over a period of time.  The accounts that show up repeated appear to be users that are actually in that region.  For some reasons that I can't quite tell, many results from a geographic search are sporadically from places around the world - but accounts that show up, say over 10 times in some time period are a pretty good bet.  This is just my observation after looking through a bunch of these results.   In any case, I was able to come up with a list of user names, and pickled it.  So our first step was to unpickle it:


allnames = pickle.load(open("allnames.pickle", "r"))

#step 1  Download user data for a certain epoch from twitter

#first hook up to twitter api 

config = {}
execfile("config.py", config)
twitter = Twitter(auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))



MID = 1037846305224224768  #This is a max id that I grabbed when I started this.  I'll call this Epoch 1 and it will extend backwards 4*10^16 tweets   
SID = MID - 40000000000000000

def get_timeline(sn):   #This pickles all the trimmed status data, doesn't save a lot of user info.  Since I'm not sure if I want to mine it more later, I pickle it all for now in a directory 'status_data/'
    all_statuses = []
    howmany = 0 
    try : statuses = twitter.statuses.user_timeline(screen_name=sn, count = 200, max_id = MID, include_rts = True, since_id = SID, trim_user = True, exclude_replies = False)
    except : 
        print ('fail')
        time.sleep(10)
        return('fail')
    print ("success")
    all_statuses+=statuses
    if len(all_statuses)==0 : return('empty')
    ids = [k['id'] for k in statuses]
    mid = min(ids)
    time.sleep(.2)
    while(True):
        statuses = twitter.statuses.user_timeline(screen_name=sn, count = 200, max_id = mid, include_rts = True, since_id = SID, trim_user = True, exclude_replies = False)
        all_statuses+=statuses
        ids = [k['id'] for k in statuses]
        if(len(ids))<10: break
        mid = min(ids)
        time.sleep(.8)
        print mid, sn, howmany
        howmany+=1
    pickle.dump(all_statuses, open('status_data/'+sn+".all_statuses","w"))
    

error_names = []
for sn in allnames[0:25000]:   ## This actually takes quite a while like a couple days
    if(get_timeline(sn)=='fail'):
        error_names+=[sn]   ## save these names so I can remember to remove them from list.  Many accounts gets deleted or go private. 
    print allnames.index(sn)



###Step 2:  Use this data to create a data frame


def get_pickled_status(sn):#This may throw an error if no file, but the exception handling is easier elsehwere
    statuses = pickle.load(open('status_data/'+sn+".all_statuses", "r")) 
    return statuses



def user_data_create(sn):   #takes a screen name, opens the pickled file and returns a data frame line.   Also could throw error if no file, but deal with this elsewhere
    stas = get_pickled_status(sn)
    numb = len(stas)
    hashtags =  [  [i['text'] for i in s['entities']['hashtags'] ] for s in stas if s['entities']['hashtags'] !=[] ]  #unflattened list of lists
    sources = [s['source'].split(">")[1].split("<")[0] for s in stas]
    replied_to_status=[s['in_reply_to_status_id'] for s in stas if s['in_reply_to_status_id']!=None]
    replied_to_screen_names=[s['in_reply_to_screen_name'] for s in stas if s['in_reply_to_screen_name']] 
    retweeted_id = [s['retweeted_status']['id'] for s in stas if 'retweeted_status' in s.keys()]
    retweeted_user_id = [s['retweeted_status']['user']['id'] for s in stas if 'retweeted_status' in s.keys()]
    how_many_retweets = len(retweeted_id)
    sc = Counter(sources)
    texts = [s['text'].replace('.', ' ') for s in stas]
    texts = [s.replace(',', ' ') for s in texts]
    texts = [s.split(" ") for s in texts]
    flat_words = [item for sublist in texts for item in sublist]
    flat_set = set(flat_words)
    return [sn, numb, hashtags, replied_to_status, replied_to_screen_names, retweeted_id, retweeted_user_id, how_many_retweets, sc.most_common(3), flat_set]



### Define a dataframe here
df5 = pd.DataFrame(columns = ['sn', 'numb', 'hashtags', 'replied_to_status', 'replied_to_screen_names', 'retweeted_id', 'retweeted_user_id', 'how_many_retweets', 'sc.most_common(3)', 'word_set'])

sample_names = allnames[0:25000]
for ss in sample_names:
    try : df5.loc[len(df5)] = user_data_create(ss)
    except : 
        print "error"
        error_names +=[ss] 
    print ss, sample_names.index(ss)

#this took almost 7000 seconds to do 25000 name, end up with about 17000 good names


#Step 3.  Take the dataframe and find the common words

names = list(df5.sn)
all_sets = [list(s) for s in list(df5.word_set)] #only a few seconds
all_words = [item for sublist in all_sets for item in sublist] # a few seconds
cc = Counter(all_words)  #about a minute.  There's about 9 million distinct objects here

#Next, we want to shrink this to the top 25000 'words'

top25000set = [h[0] for h in cc.most_common(25000)] 
all_sets_top_words = [s.intersection(top25000set) for s in list(df5.word_set)]
df5['word_set']  = all_sets_top_words



#Step 4.  Find the words that have highest "entropy"

#Next, we look at the retweets and create a metric on the retweets.  We will be using metric_utilities.py file

execfile("metric_utilities.py")

retweet_list = list(df5.retweeted_id)

retweet_metric = lists_to_metric(retweet_list, 2000)  #took about 4 minutes.  What it does it takes the 2000 most common retweeted tweets, and puts a diffusion distance metric on these.  
retweet_metric = retweet_metric[1:3]  #this is the useful info, first item is the metric, second is labels 

LL = list(df5.retweeted_id)
LL = [set(l) for l in LL]
LL = [l.intersection(retweet_metric[1]) for l in LL]
LL = [list(l) for l in LL]  ##this bit is quick and reduces retweets in list of retweets to those that are in the top2000

#next goal to get entropy of the words, over the space of common tweets.  A word will have more entropy, if that word predicts contains more predictive information about which users will retweet certain tweets.  

def get_word_indices(word): #uses global df5.  This will return the index set of users that used this word
    truefalse = [word in l for l in df5.word_set]
    return [i for i in range(len(truefalse)) if truefalse[i]]
    
def get_word_entropy(word_list):  #uses global list of tweets LL, allows you to pass a word list and returns the entropy values for all words
    word_inds = [get_word_indices(word) for word in word_list]
    word_count = [len(ww) for ww in word_inds]
    return(slice_entropy(LL, 2000, word_inds), word_count)  #slice_entropy is defined in metric_utilities.py


all_entropies = get_word_entropy(top25000set)[0]  #Took almost ten minutes 

#we have entropy, now the tricky part : Selecting a subset of 25000 words that have high entropy but that the words themselves are not very similar to other words. We want to create a metric on the set of words, but 25000 is to cumbersome, so we want to reduce this.  Our first naive attempts were to finds words that had the most entropy, but these turned out to be the words that contain very similar information. So in order to get around this, we broke the words into smaller groups, put a distance metric on these words, and then clustered these into smaller groups. From the smaller groups, we choose the top 12-15 percent according to entropy.  

def get_top_indices(list_of_numbers, how_many): #Silly little function that returns the arg max for top n 
    values = np.asarray(list_of_numbers).copy()   
    minv = values.min()
    maxindex = np.argmax(values)
    top =[]
    top+=[maxindex]
    for h in range(how_many-1):
        values[maxindex]=minv
        maxindex = np.argmax(values)   
        top+=[maxindex]
    return top
    


from sklearn.cluster import KMeans
all_sets_top_words =list(df5.word_set)

splitnumber = 40    #(25000/40 = 625 which is manageable
PERCENT = 15  #shoot for 15 but integer rounds down, so closer to 12
NCLUSTERS=15  #they are uneven, but this averages about 40 per cluster
best_words = []
for i in range(splitnumber):
    print "tranche ",i , " ...............###"
    indexset = [splitnumber*k+i for k in range(25000/splitnumber)]
    wordset =[list(top25000set)[k] for k in indexset]
    wordentropies=[all_entropies[k] for k in indexset]
    print "computing metric..." 
    tt = time.time()
    a = lists_to_metric_specified_sublist(all_sets_top_words, wordset)  #this is heavy, time consuming
    elapsed = time.time() -tt
    print "that took ", elapsed, " seconds, now computing kmeans..."
    kmeans = KMeans(n_clusters=NCLUSTERS, random_state=0).fit(a)
    for r in range(NCLUSTERS):
        print " cluster ", r, " ..."
        cluster = [s for s in range(len(wordset)) if kmeans.labels_[s]==r]  #wordset indices of this cluster
        cluster_entropies = [wordentropies[s] for s in cluster]
        top_number = PERCENT*len(cluster)/100
        top_indices = get_top_indices(cluster_entropies, top_number)
        top_indices =  [cluster[s] for s in top_indices]
        top_words = [wordset[s] for s in top_indices]
        print top_words
        best_words+=top_words

##took nearly 5 hours.  

#3501 best words


##STEP 5.  Find a metric on these chosen words

A = lists_to_metric_specified_sublist(all_sets_top_words, best_words)  # less than hour. This is THE METRIC on the best words
 
all_sets_best_words = [s.intersection(best_words) for s in list(df5.word_set)]
df5['best_words'] = all_sets_best_words

##Step 6.  Compute metric on subset of accounts, based on word metric

qq = Wmetric_with_pre(df5.best_words, best_words, A, threshold=80, size_cap = 500, RandomAboveThreshold = True)  #Took 3-4 hours. this step is finding 500 users who have been at least moderately active, and computes a dense metric user Wasserstein distanc eon these users, based on the word metrc.  

#STEP 7.   Based on that metric on subset of users choose a representive sample set of users as landmarks. 

landmarks = W_landmark(qq[0], 25)
landmark_inds = [qq[1][x] for x in landmarks]
landmark_screen_names = df5.sn[landmark_inds]


#Step 8.  Compute the distance of each user to the landmark users, save this as coordinates in a dataframe.  

g1 = LMdistancesOld(landmark_inds, df5.best_words, top_objects = 2000)   #some hours

#add these to the data frame 
newlabels = ["word_l"+str(i) for i in range(len(g1))]
for nl in newlabels:
    df5[nl] = g1[newlabels.index(nl)]


#Step 9  Now we compute a Wasserstein Metric based on set of retweets


qq = Wmetric_with_pre(df5.retweeted_id, retweet_metric[1], retweet_metric[0], threshold=8, size_cap = 300, RandomAboveThreshold = True) 


landmarks = W_landmark(qq[0], 15)
landmark_inds = [qq[1][x] for x in landmarks]
landmark_screen_names = df5.sn[landmark_inds]

g2 = LMdistancesOld(landmark_inds, df5.retweeted_id, top_objects = 2000) 


newlabels = ["rt_l"+str(i) for i in range(len(g2))]
for nl in newlabels:
    df5[nl] = g2[newlabels.index(nl)]

#Now our data frame should have 25 columns expressing "word distance" from certain users, and 15 columns expressing "retweet distance" from certain users


#Step 10  Now we compute a Wasserstein Metric based on set of hashtags 
#note first needed to flatten hashtag lists, as they come back from twitter as lists

flat_tags = [[item for sublist in h for item in sublist] for h in list(df5.hashtags)]

qq = Wmetric(flat_tags, threshold=6, top_objects=1000, size_cap = 300, RandomAboveThreshold = True)

landmarks = W_landmark(qq[0], 10)
landmark_inds = [qq[1][x] for x in landmarks]
landmark_screen_names = df5.sn[landmark_inds]


g3 = LMdistancesOld(landmark_inds, flat_tags, top_objects = 1000) 

newlabels = ["ht_l"+str(i) for i in range(len(g3))]
for nl in newlabels:
    df5[nl] = g3[newlabels.index(nl)]

##so now we have three different sets of measurements, that is distances from landmark accounts.   To find neighbors, we do the following
#for each separate measurement, we get neighbors

#first words


cols = [c for c in df5.columns if "word_l" in c]+['sn']  
dfn = df5[cols]
dfn.set_index('sn', inplace=True)

neighbors(dfn, names[99], 15)   ## This computers 15 nearest neighbors to names[5] (whoever that is) wrt word metric


#similar for retweets  - this might not distinguish as well if the user didn't retweet alot 

cols = [c for c in df5.columns if "rt_l" in c]+['sn']  
dfn = df5[cols]
dfn.set_index('sn', inplace=True)
neighbors(dfn, names[5], 15)

#similar for hashtags  - similarly, if user wasn't a big hashtagger this might not give as much info

cols = [c for c in df5.columns if "ht_l" in c]+['sn'] 
dfn = df5[cols]
dfn.set_index('sn', inplace=True)
neighbors(dfn, names[5], 15)


out = neighbors(dfn, names[9106], 1000)

with open('1000names.txt', 'w') as f:
    for item in out[0]:
        f.write("%s\n" % item)



