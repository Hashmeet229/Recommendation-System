# importing various requried libariers
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

# Loading the Data
excel= pd.ExcelFile('mycollected_data.xlsx')
df=excel.parse('Sheet1')
df = df[df['Calendar Month'] !='Result']

# Grouping the data, to get frequency per customer
df_1=df.groupby(['Smart Card','Class.1']).count()
df_1.reset_index(inplace=True)
df_1=df_1[['Smart Card','Class.1','Bill Number']]
df_1.rename(index=str, columns={"Bill Number": "freq"},inplace=True)
df_1.head()

# removing the customers with less than 50 Transactions and products with less than 200 purchased frequency
min_product_freq = 200
filter_products = df_1['Class.1'].value_counts() > min_product_freq
filter_products = filter_products[filter_products].index.tolist()
min_user_freq = 50
filter_users = df_1['Smart Card'].value_counts() > min_user_freq
filter_users = filter_users[filter_users].index.tolist()
df_new = df_1[(df_1['Class.1'].isin(filter_products)) & (df_1['Smart Card'].isin(filter_users))]
print('The original data frame shape:\t{}'.format(df_1.shape))
print('The new data frame shape:\t{}'.format(df_new.shape))
df_new.to_csv('final_df.csv')

df = pd.read_csv('final_df.csv')
df.drop(columns=['Unnamed: 0'], inplace= True)

# Getting DataFrame with frequency with customers
freq = pd.DataFrame(df.drop(columns=['Class.1']))

df_cl = pd.read_csv('clusters_df.csv')
df_cl.drop(columns=['Unnamed: 0'], inplace= True)
df_cl.head()
df_n=pd.merge(df, df_cl,right_on='Smart Card',left_on='Smart Card')
df_n.head()

# grouping the customers with frequency Quantiles formed
def q2(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.5)
def q4(x):
    return x.quantile(0.75)
df1=df_n.groupby(['Smart Card','Class.1','Cluster']).agg({'freq':['min',q2,q3,q4,'max']})
df1.reset_index(inplace=True)
df1.columns = ['_'.join(col).strip() for col in df1.columns.values]
df1['freq_q2'] = df1['freq_q2'].astype(int)
df1['freq_q3'] = df1['freq_q3'].astype(int)
df1['freq_q4'] = df1['freq_q4'].astype(int)
del df_n

df_c = pd.merge(df1, freq,left_on='Smart Card_',right_on='Smart Card')
del df1
df_c.drop(columns=['Smart Card'], inplace= True)

df_c_s = df_c.groupby(['Smart Card_','Class.1_','Cluster_']).agg({'freq_min':'mean','freq_q2':'mean','freq_q3':'mean','freq_q4':'mean','freq_max':'mean','freq':'mean'})
df_c_s.reset_index(inplace=True)
print(df_c_s.head())
del df_c

# Splitting into different DataFrame according to cluster id
df_c1 = df_c_s[df_c_s['Cluster_']==0]
df_c1.reset_index(inplace=True)
df_c2 = df_c_s[df_c_s['Cluster_']==1]
df_c2.reset_index(inplace=True)
df_c3 = df_c_s[df_c_s['Cluster_']==2]
df_c3.reset_index(inplace=True)
df_c4 = df_c_s[df_c_s['Cluster_']==3]
df_c4.reset_index(inplace=True)
df_c5 = df_c_s[df_c_s['Cluster_']==4]
df_c5.reset_index(inplace=True)
list_df = [df_c1, df_c2, df_c3, df_c4, df_c5]
for i in range(0,len(list_df)):
  print(list_df[i].shape[0])

# Grouping Various Clusters according to the frequency quantile.
def test(a,b,c,d,e,f):
  if (a == b):
    return 1
  elif (c < a) & (a <= d):
    return 3
  elif (d < a) & (a <= e):
    return 4
  elif (e < a) & (a <= f):
    return 5
  elif (b < a) & (a <= c):
    return 2
  elif (a < b):
    return 6
  elif (a > f):
    return 7

import multiprocessing
import tqdm
import concurrent.futures

# Now performing for all clusters through Parallel processing
num_processes = multiprocessing.cpu_count()
with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
    df_c1['group']=list(tqdm.tqdm(pool.map(test, df_c1['freq'], df_c1['freq_min'],
                                           df_c1['freq_q2'],df_c1['freq_q3'],
                                           df_c1['freq_q4'],df_c1['freq_max'],
                                           chunksize=10), total=df_c1.shape[0]))
    df_c2['group']=list(tqdm.tqdm(pool.map(test, df_c2['freq'], df_c2['freq_min'],
                                           df_c2['freq_q2'],df_c2['freq_q3'],
                                           df_c2['freq_q4'],df_c2['freq_max'],
                                           chunksize=10), total=df_c2.shape[0]))
    df_c3['group']=list(tqdm.tqdm(pool.map(test, df_c3['freq'], df_c3['freq_min'],
                                           df_c3['freq_q2'],df_c3['freq_q3'],
                                           df_c3['freq_q4'],df_c3['freq_max'],
                                           chunksize=10), total=df_c3.shape[0]))
    df_c4['group']=list(tqdm.tqdm(pool.map(test, df_c4['freq'], df_c4['freq_min'],
                                           df_c4['freq_q2'],df_c4['freq_q3'],
                                           df_c4['freq_q4'],df_c4['freq_max'],
                                           chunksize=10), total=df_c4.shape[0]))
    df_c5['group']=list(tqdm.tqdm(pool.map(test, df_c5['freq'], df_c5['freq_min'],
                                           df_c5['freq_q2'],df_c5['freq_q3'],
                                           df_c5['freq_q4'],df_c5['freq_max'],
                                           chunksize=10), total=df_c5.shape[0]))

for i in range(0,len(list_df)):
  print((list_df[i][list_df[i]['group'] == '']).shape[0])
list_df_1 = [df_c1, df_c2, df_c3, df_c4, df_c5]
df_c = pd.concat(list_df_1)
df_c.drop(columns=['index','freq_q2','freq_q3','freq_q4','freq_min','freq_max'], inplace= True)
print(df_c.head())
df_c.to_csv('final_df_before_recommendation.csv')

# ! pip install scikit-surprise

# Importing the Required Libraries
from surprise import evaluate
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise.dataset import Dataset
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split
from surprise import accuracy

reader = Reader(rating_scale=(1,7))
data = Dataset.load_from_df(df_c1[['Smart Card_','Class.1_','freq']], reader)

# getting the most effective Algorithm for Recommendation System
benchmark = []
for algorithm in [SVD(), NMF(), SVDpp(), SlopeOne(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(),
                  KNNWithZScore(), BaselineOnly(), CoClustering()]:
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=True)
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
print(pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse'))

# Splitting data into Train and Test Dataset
trainset , testset = train_test_split(data, test_size=0.25)
algo_svd = SVDpp(verbose=True)
predictions_svd = algo_svd.fit(trainset).test(testset)
print('for 1 cluster svd ',accuracy.rmse(predictions_svd))


def get_Iu(uid):
    """
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0
def get_Ui(iid):
    """
      the number of users that have rated the item.
    """
    try:
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
df = pd.DataFrame(predictions_svd, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)
best_predictions = df.sort_values(by='err')[:10]
worst_predictions = df.sort_values(by='err')[-10:]
print(df.head())

# Recommending for all customers
sc = list(df_c['Smart Card_'].unique())
len(sc)

recommend = []
for i in range(0,len(sc)):
    # Getting products, which customer has not purchased and building recommendation with rest of the products
    iids = df_c['Class.1_'].unique()
    idds584 = df_c.loc[df_c['Smart Card_']==sc[i],'Class.1_']
    iids_to_pred = np.setdiff1d(iids, idds584)
    test_set = [[sc[i], iid, 6] for iid in iids_to_pred ]
    predictions = algo_svd.test(test_set)
    pred_rating = np.array([pred.est for pred in predictions])
    ## index of maxi predicted frq
    #i_max = pred_rating.argmax()
    # i_max = np.argpartition(pred_rating,)
    ## find coressponding product
    #iid = iids_to_pred[i_max]
    #print('top item for user 200000025584 has iid {0} with predicted rating {1} '. format(iid,pred_rating[i_max]))

    # getting top 10 (n) recommendations
    import heapq
    l=list(pred_rating)
    def nth_largest(n, l):
      return heapq.nlargest(n, l)[-1]

    n = 10
    lar = nth_largest(n, l)
    ind = np.where(pred_rating > lar)
    recommend.append(iids_to_pred[ind])
    print(i)

# Creating a dataframe with recommendation by collaborative filtering with unique Smart Card
rec_df = pd.DataFrame(recommend)
rec_df['Recommended_Collaborative_Filtering'] = rec_df[rec_df.columns[0:]].apply(lambda x: ', '.join(x.dropna().astype(str)),axis=1)
recommend_df = pd.DataFrame(rec_df['Recommended_Collaborative_Filtering'])
print(recommend_df.head())
smart_card = pd.DataFrame({'Smart Card':sc})

# Merging the Recommendations with the Customers id's
df_c=pd.merge(smart_card, recommend_df,left_index=True, right_index=True)
print(df_c.head())
df_c.to_csv('Final_recommendations_using_CollaborativeFiltering.csv')