# importing various requried libariers
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

# loading the data
excel= pd.ExcelFile("mycollected_data.xlsx")
df=excel.parse('Sheet1')
print(df.head())
print('shape of df',df.shape)
df.info()

# checking for any null values in data
df.isnull().sum()

df['Class.1'].nunique()
df['Family.1'].nunique()
df['Material.1'].nunique()
df.describe()
df = df[df['Calendar Month'] !='Result']

# converting to date-time dtype
df['Billing Date'] = pd.to_datetime(df['Billing Date'])
df['Time'] = pd.to_datetime(df['Time'])
# extracting month, year from date-time
df['month'] =df['Billing Date'].apply(lambda x: x.month)
df['year']=df['Billing Date'].apply(lambda x: x.year)

# Creating dataframe considering unique smart card with frequency of sales per year and per month
df_1=df.groupby(['year','month','Smart Card']).count()
df_1.reset_index(inplace=True)
df_1=df_1[['year','month','Smart Card','Unnamed: 0']]
df_1.rename(index=str, columns={"Unnamed: 0": "freq"},inplace=True)
print(df_1.head())

# GroupBy with smart card, year, month with mean of frequency
df2=df_1.groupby(['Smart Card','year','month']).agg({'freq':'mean'})
df2.reset_index(inplace=True)
print(df2.head())
df2_1=df2.groupby(['Smart Card']).agg({'freq':'mean'})
df2_1.reset_index(inplace=True)

# GroupBy with smart card taking median and mean of Gross sales
df3=df.groupby(['Smart Card']).agg({"Gross Sales (Rs)":['mean','median']})
df3.reset_index(inplace=True)
df3.columns = ['_'.join(col).strip() for col in df3.columns.values]
print(df3.head())

df_4=df.groupby(['Smart Card','year','month','Brick']).agg({'Sales Qty':{'Sales Qty': 'sum'},'Gross Sales (Rs)':{'Gross Sales (Rs)': 'sum'}})
df_4.reset_index(inplace=True)
df_4.columns = df_4.columns.droplevel(1)
print(df_4.head())

# GroupBy with smart card and brick with median of Sales Qty and Gross sales
df_5=df_4.groupby(['Smart Card','Brick']).agg({'Sales Qty':['median'],'Gross Sales (Rs)':['median']})
df_5.columns = df_5.columns.droplevel(1)
df_5.reset_index(inplace=True)
print(df_5.head())

# Forming a loop to split dataframe with unique smart card and then converting to wide format on Sales Qty and Gross Sales
smart_c=list(set(df_5['Smart Card']))
df_5['Brick']=df_5['Brick'].astype('str')
n=2000
final_list= [smart_c[i * n:(i + 1) * n] for i in range((len(smart_c) + n - 1) // n )]
len(final_list)
list_df=[]
for i in range(0, len(final_list)):
    df_5_loop=df_5[df_5['Smart Card'].isin(final_list[i])] 
    p=pd.pivot_table(df_5_loop,values=['Sales Qty',"Gross Sales (Rs)"], index=['Smart Card'], columns='Brick',aggfunc=np.sum,fill_value=0)
    p.columns = list(map("_".join, p.columns))
    list_df.append(p)

# Joining all dataframes formed above
df_c =pd.concat(list_df)
print(df_c.head())
df_c.shape
df_c.info()

# This Function reduces the 'int' size in Data, use if memory error and data contains many columns  with 'int'.

# import gc
# # Function to reduce size of int datatypes in dataframe, reducing memory usage
# def downcast_df_int_columns(df):
#     list_of_columns = list(df.select_dtypes(include=["int32", "int64"]).columns)
#
#     if len(list_of_columns)>=1:
#         max_string_length = max([len(col) for col in list_of_columns]) # finds max string length for better status printing
#         print("downcasting integers for:", list_of_columns, "\n")
#
#         for col in list_of_columns:
#             print("reduced memory usage for:  ", col.ljust(max_string_length+2)[:max_string_length+2],
#                   "from", str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8), "to", end=" ")
#             df[col] = pd.to_numeric(df[col], downcast="integer")
#             print(str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8))
#     else:
#         print("no columns to downcast")
#
#     gc.collect()
#
#     print("done")
# downcast_df_int_columns(df_c)

del df_4
del df_5
del list_df

# merging previous dataframe to final concat dataframe
df_c=pd.merge(df_c, df2_1,right_on='Smart Card',left_on='Smart Card')
df_c=pd.merge(df_c, df3,right_on='Smart Card_',left_on='Smart Card')

del df2
del df3
df_c_s = pd.DataFrame(df_c['Smart Card'])

# preparing final dataframe for clustering
df_c.reset_index(inplace=True)
df_c=df_c.drop(columns=['Smart Card','Smart Card_','index'])
print(df_c.head())
# replacing null values with zero
df_c.fillna(0,inplace=True)

# # Normalize the Data, to check for any alteration in clustering
# x=list(df_c.columns)
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(df_c)
# df_s=pd.DataFrame(x_scaled)
# df_s.columns = x

## Calculating silhoutte score for each cluster using mini-batch KMeans, hence finding optimal value of 'k'
# from sklearn.cluster import KMeans
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.metrics import silhouette_score
# range_n_clusters = list(range(2,15))
# for n_clusters in range_n_clusters:
#    clusterer =  MiniBatchKMeans(n_clusters=n_clusters, batch_size=3000)
#    preds = clusterer.fit_predict(df_c)
#    centers = clusterer.cluster_centers_
#    score = silhouette_score (df_c, preds, metric='euclidean')
#    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

# Elbow Method
from sklearn.cluster import MiniBatchKMeans
Sum_of_squared_distances = []
K = range(2,15)
for k in K:
    km = MiniBatchKMeans(n_clusters=k,
                         random_state=0,
                         batch_size=2000,
                         verbose=10)
    km = km.fit(df_c)
    Sum_of_squared_distances.append(km.inertia_)
from matplotlib.pyplot import figure
figure(figsize=(10, 6))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Optimal k')
plt.show()

# Optimal value of k is 5
from sklearn.cluster import MiniBatchKMeans
kmeans_1 = MiniBatchKMeans(n_clusters=5,
                         random_state=0,
                         batch_size=2000,
                          verbose=10)
preds_1 = kmeans_1.fit_predict(df_c)
l=pd.DataFrame({'Cluster':preds_1})

# Merging clusters to the final dataframe
l=pd.DataFrame({'Cluster':preds_1})
df_c=pd.merge(df_c,l,left_index=True,right_index=True)
print(df_c.head())

sc = list(df['Smart Card'].unique())
smart_card = pd.DataFrame({'Smart Card':sc})
df_c1 = pd.merge(smart_card, l,left_index=True, right_index=True)
df_c1.to_csv('clusters_df.csv')