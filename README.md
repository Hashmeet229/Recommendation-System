# Recommendation System 

Considering Smart Card as Unique Ids of the customers and Class as categories of products purchased by customers.
Recommendations for each customer are performed through customer segmentation and then for each segmentation recommendations 
are made for customers in that particular segment.
Firstly, clustering is performed based on customer behavior i.e. frequency of buying of particular product per month along 
with each customer paying price for particular product and quantity of that product purchased by each customer. 
From this pivot table is constructed with columns as Class (products), index as smart card (customer id) and 
values as Gross sales and Sales quantity for each product purchased by customer. This act as final matrix for clustering to be performed.

After Clustering is performed based on above customer behavior, then recommendations are performed on these clusters using two methods:
By Association rule and By Collaborative filtering.
These recommendations are made, considering products that are not been already purchased by that customer and recommending products
other than that purchased product.

Association rule: This is performed on each cluster (customer segment) by using Apriori algorithm with parameters:
min_support =0 .06, max_len=4, min_threshold=5 (lift). This gives various pairs of products (of various length) that 
can been purchased by customer through association rule (based on frequency).

Collaborative Filtering: This is performed on each cluster (customer segment) by forming an item based matrix which gives out 
a product based on similarity score between the products/item matrix (products which are nearer to each other are related).
This gives recommendations based on similarity between items which are irrespective of any customer.


DOCUMENTATION :

Problem Statement:
Categorize the customer into various segment according to their buying behavior. We are having sales data of various products since last 3 years and would be predicting the future sales of various products according to different customer requirement i.e. recommending the relevant product to the customer according to the past behavior of various customers lying in that particular segment (Building a Recommendation System)
This will enhance the sales and efficiency of our client through better categorization and recommendation of products to their customers.

Dataset Description: 
- Containing Bill number & Smart Card as customer unique ids with billing date & time.
- Products details are given under Brick (Main category of product), Class (Sub-category), Material (Product’s name) along with their price (MRP-Values) and their Segments.
- Company’s Gross Sales (Rs.) is also been provided along with the plant details and region of that plant.

Tools Used:
Coding Language: Python 3.0
Libraries: Pandas, Numpy, Matplotlib, sklearn, Surprise, Apriori, mlxtend
Platform: PyCharm
Algorithm: Mini-Batch K-Means Clustering, Apriori, NMF, SVD, SVDpp 

Solution:
Considering Smart Card as Unique Ids of the customers and extracting various features as required. User based filtering is done according to the various products (Bricks) and then Feeding it to the Mini-Batch K-Means Clustering to get clusters of customers and then on each cluster Recommendations are made through Association Rule and Collaborative Filtering.

PART: 1 (Clustering)

Steps:
-	Importing required Libraries.
-	Merging the data from various years/months to a single dataframe.
-	Loading the Data through pandas (as df).
-	General overview of the data like checking for the shape of the data, df.info(), if any null value exits in the data ( df.isnull().sum() ), df.describe.
-	Now converting the date-time columns to datetime datatype and extracting month and year from that and also converting categorical columns to numeric using Label Encoder.
-	Now, extracting the frequency of transactions (Smart Card) made by customers per month per         year into new dataframe (df_1) and then GroupBy with Smart Card taking mean of frequency.                                                                              
-	In new dataframe (df_4), GroupBy with Smart Card and Brick per month per year and           summing up Gross Sales and Sales Quantity as aggregation function and then again new   dataframe (df_5) GroupBy previous dataframe with Smart Card and Brick taking   median of Gross Sales and Sales Quantity.                   
-	Now converting these Brick into features for clustering (as pivot table).
-	Step 7, creates a problem of converting because of huge size of data so needed to be broken into batches and performing the required operation.
- Creating a list a unique Smart Card (if not created, some duplicates tends to add up in the data) and selecting first 2000 unique Smart Cards and converting to pivot table with Bricks as columns for Gross Sales and Sales Quantity, storing in new dataframes.
-	Joining these split datarames to create final dataframe (df_c) for the model.
-	Merging this new dataframe (df_c) with the initial dataframe (df2_1, containing frequency) and also merging this dataframe with another dataframe (df3, containing median values) and resetting the index and dropping down some unwanted columns.
-	As the final dataframe has huge size, K-Means clustering leads to high usage of memory, so the alternate solution was to do with Mini-Batch K-Means Clustering with batch size of 2000.
-	Final dataframe has been developed. Now, it is to be find the optimized value of k (Number of Clusters).
-	This optimized value of k is evaluated using Elbow Method and Silhouette Score.  And the optimal value evaluated is: k = 19 clusters. 
-	After finding out the k value, predicting the clusters for each customer using the optimal k value and adding this new column (Cluster) to the final dataframe.     
                    
PART: 2 (Recommendations using Apriori)

Steps: 
-	Load Dataframe with frequency w.r.t Customer (Smart card) and Product along with the Cluster id for each customer.
-	Now, splitting the Dataframe for each Cluster id and storing these in list. 
-	Forming pivot table for each dataframe containing unique cluster using unique smart card, avoiding duplication of customers in pivot table.
-	Now, a function is created in which data (for first cluster) is given to Apriori algorithm and associations rule are generated (support is calculated from frequency data) and then a loop is run to remove duplications of associations (redundancy). Then products are extracted from dataframe that are purchased by each customer and stored in a dataframe as set. Now, if antecedents are subset of purchased items by customer then consequents are added to the recommended list for that customer. 
-	Then these recommended products are converted into sets and compared with others for same customer, removing repetition of products and then it is stored in dataframe w.r.t Smart Card as final output for that cluster.
-	This function is executed for other dataframes containing different clusters ids.
PART: 3 (Recommendations using Collaborative Filtering)
Steps: 
-	Load Dataframe with frequency w.r.t Customer (Smart card) and Product along with the Cluster id for each customer (same as previous part).
-	Removing customers with frequency lower than 50 and product frequency lower than 200, as these two does not have much impact on the data due very less transactions in long time period and recommendation may also be less efficient for these customers.
-	Now, this data is divided into some quantiles on frequency and then based on this, grouping is done with different dataframes containing unique cluster id’s (these groups represents rating) and this is final dataframe, this act as rating by each customer. 
-	Then Surprise library is imported with its various algorithms and final dataframe is fed to reader with rating from 1 to 7. And then a loop is run through various algorithm like NMF, SVD, SVDpp and KNNBasics etc. and most efficient algorithm is selected and then data is split into train and test dataset.
-	These algorithm includes Matrix factorization, cosine similarity matrix, decomposing a matrix, using nearest neighbor etc. and NMF (matrix factorization) is selected algorithm for training the data for user-item matrix and splitting it to from individual matrix.
-	Now, this is used for all customers to recommend products (as per similarity matrix) but the products which are already purchased by that customer are removed while recommending new products to that customers and recommend top 10 products. 
-	Then these recommendations are merged with smart card as final output.    
 
