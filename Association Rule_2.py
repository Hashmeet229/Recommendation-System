# importing various requried libariers
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori

# Loading the dataset
excel= pd.ExcelFile('mycollected_data.xlsx')
df=excel.parse('Sheet1')
df = df[df['Calendar Month'] !='Result']

# Grouping the data, to get frequency per customer
df_1=df.groupby(['Smart Card','Class.1']).count()
df_1.reset_index(inplace=True)
df_1=df_1[['Smart Card','Class.1','Bill Number']]
df_1.rename(index=str, columns={"Bill Number": "freq"},inplace=True)
print(df_1.head())
del df

df_cl = pd.read_csv('clusters_df.csv')
df_cl.drop(columns=['Unnamed: 0'], inplace= True)

# Merging the dataframe with Clusters assigned to each customer
df_n=pd.merge(df_1, df_cl,right_on='Smart Card',left_on='Smart Card')
print(df_n.head())
del df_1
del df_cl

df_c_n = df_n.groupby(['Smart Card','Class.1','Cluster']).agg({'freq':'mean'})
df_c_n.reset_index(inplace=True)
print(df_c_n.head())
df_c_n.to_csv('final_df_apriori.csv')

# Splitting into different DataFrame according to cluster id.
df_c1 = df_c_n[df_c_n['Cluster']==0]
df_c1.reset_index(inplace=True)
df_c1.drop(columns=['index'],inplace=True)
df_c2 = df_c_n[df_c_n['Cluster']==1]
df_c2.reset_index(inplace=True)
df_c2.drop(columns=['index'],inplace=True)
df_c3 = df_c_n[df_c_n['Cluster']==2]
df_c3.reset_index(inplace=True)
df_c3.drop(columns=['index'],inplace=True)
df_c4 = df_c_n[df_c_n['Cluster']==3]
df_c4.reset_index(inplace=True)
df_c4.drop(columns=['index'],inplace=True)
df_c5 = df_c_n[df_c_n['Cluster']==4]
df_c5.reset_index(inplace=True)
df_c5.drop(columns=['index'],inplace=True)

# storing above dataframes to list
list_df = [df_c1, df_c2, df_c3, df_c4, df_c5]
for i in range(0,len(list_df)):
  print(list_df[i].shape[0])

# Forming pivot table from each dataframe above taking unique Smart Card to avoid duplication of customers
list_df1 = []
for j in range(0,len(list_df)):
  smart_c = list(set(list_df[j]['Smart Card']))
  n=2000
  final_list= [smart_c[i * n:(i + 1) * n] for i in range((len(smart_c) + n - 1) // n )]
  for i in range(0, len(final_list)):
    df_loop = list_df[j][list_df[j]['Smart Card'].isin(final_list[i])]
    p = pd.pivot_table(df_loop,values=['freq'], index=['Smart Card'], columns='Class.1',aggfunc=np.sum,fill_value=0)
    list_df1.append(p)
del df_c_n

# Install Apyori
# ! pip install apriori
# ! pip install apyori
for i in range(0,len(list_df1)):
    list_df1[i].columns = list_df1[i].columns.droplevel()

# Function to perform this association for each cluster
# Function input is DataFrame with unique cluster (stored in list: list_df)
def apriori_recommend(df):
  def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
  df_for_ar = df.applymap(encode_units)

  # Performing Association Rule using Apriori
  frequent_itemsets = apriori(df_for_ar, min_support=0.06, max_len=4, use_colnames=True)
  rules1 = association_rules(frequent_itemsets, metric="lift", min_threshold=5)
  rules1.drop(columns=['antecedent support', 'consequent support'], inplace=True)
  print(rules1.sort_values(by='lift', ascending=False)[:10])
  rules1.reset_index(inplace=True)

  # Removing the duplications in Association Output
  for i in range(0,rules1.shape[0]-1):
    if (rules1.loc[i]['antecedents']==rules1.loc[i+1]['consequents'] and rules1.loc[i]['consequents']==rules1.loc[i+1]['antecedents']) :
      rules1.at[i+1,'antecedents'] = 'None'
      rules1.at[i+1,'consequents'] = 'None'
    print(i)
  indexNames = rules1[(rules1['antecedents']=='None') & (rules1['consequents']=='None')].index
  rules1.drop(indexNames,inplace=True)

  orders1 = []
  for i in range(0,df.shape[0]):
    orders1.append([str(df.columns[j]) for j in range(0,df.shape[1]) if (df.values[i,j] != 0.0)])

  # Getting a DataFrame with products a customer has purchased
  df_c11 = pd.DataFrame(orders1)
  df_c1_p = pd.DataFrame()
  df_c1_p['Purchased'] = df_c11[df_c11.columns[0:]].apply(lambda x: ','.join(x.dropna()),axis=1)
  df_c1_p['Purchased'] = df_c1_p['Purchased'].apply(lambda x: x.split(","))
  df_c1_p['Purchased'] = df_c1_p['Purchased'].apply(lambda  x: set(x))
  print(df_c1_p.head())
  del df_c11
  rules1.reset_index(inplace=True)

  # getting consequents from association output, if all antecedents had been Purchased by customer
  recommend = []
  for i in range(0,df_c1_p.shape[0]):
    print(i)
    recommend_1 = set()
    for j in range(0,rules1.shape[0]):
      if  len(rules1.loc[j]['antecedents'].intersection(df_c1_p.loc[i]['Purchased'])) >= len(rules1.loc[j]['antecedents']):
        recommend_1.add(rules1.loc[j]['consequents'])
    recommend.append(recommend_1)

  new_recommend = []
  for i in range(0,len(recommend)):
    new_recommend.append([list(x) for x in recommend[i]])

  new_list = []
  for i in range(0,len(new_recommend)):
    cust_list = []
    for j in range(0,len(new_recommend[i])):
      for z in range(0,len(new_recommend[i][j])):
        cust_list.append(new_recommend[i][j][z])
    new_list.append(cust_list)

  # Putting the Recommendations to a new dataframe as string
  rec_df = pd.DataFrame(new_list)
  rec_df['Recommended_Association_Rule'] = rec_df[rec_df.columns[0:]].apply(lambda x: ', '.join(x.dropna().astype(str)),axis=1)
  rec_df['Recommended_Association_Rule'] = rec_df['Recommended_Association_Rule'].apply(lambda x: x.split(","))
  rec_df['Recommended_Association_Rule'] = rec_df['Recommended_Association_Rule'].apply(lambda  x: set(x))
  rec_df['Recommended_Association_Rule'] = rec_df['Recommended_Association_Rule'].apply(lambda  x: ', '.join(str(s) for s in x))
  recommend_df = pd.DataFrame(rec_df['Recommended_Association_Rule'])
  del rec_df
  print(recommend_df.head())

  sc = list(df_c1['Smart Card'].unique())
  len(sc)
  smart_card = pd.DataFrame({'Smart Card':sc})
  # Merging the recommendations with the Customers in unique cluster id DataFrame
  df_c1_p = pd.merge(smart_card, recommend_df,left_index=True, right_index=True)
  print(df_c1_p.head())
  df_c1_p.to_csv('Final_recommendations_using_association_clusters.csv')

# Recommending for customers in all Clusters
apriori_recommend(list_df1[0])
apriori_recommend(list_df1[1])
apriori_recommend(list_df1[2])
apriori_recommend(list_df1[3])
apriori_recommend(list_df1[4])