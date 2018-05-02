# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:19:53 2018

@author: USER-1
"""

import sys
sys._enablelegacywindowsfsencoding()
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import array
import re, math
from collections import Counter
#from IPython.display import display
#matplotlib.pyplot.ion()
import pandas as pd
from sklearn.cluster import KMeans


#import numpy as np
# Importing the dataset

'''
Sample dataset is read. 
Dataset contains:
    title of book
    description
    price
    genre 
    and other relevant information.
'''
#dataset = pd.read_csv('C:\\Users\\USER-1\\Downloads\\sample_csv.csv')


 
# creating file handler for 
# our example.csv file in
# read mode
file_handler = open("C:\\Users\\USER-1\\.spyder\\combined.csv", "r")
 
# creating a Pandas DataFrame
# using read_csv function 
# that reads from a csv file.
dataset = pd.read_csv(file_handler, sep = ",")
 
# closing the file handler
file_handler.close()
 
# creating a dict file 
gender = {'Philosophy': 1,'Spiritual': 2, 'Fiction':3,'Classics':4,'City Book/Travel Guide':5,'History':6,'Biography':7,'Reseach Magazines/Technical magazines':8,'Cooking':9,'NonFiction':10,'Dictionary':11,'Poetry':12,'Sports':13,'Gardening':14,'Business':15,'Self-help':16,'Health':17,'Music':18,'Humour':19,'Animals':20,'Education':21,'Photography':22,'Enviornment':23,'Entertainment':24, 'Comedy':25}
 
# traversing through dataframe
# Gender column and writing
# values where key matches
dataset.genre = [gender[item] for item in dataset.genre]
#print(dataset)

#df = pd.read_json(r'C:\\Users\USER-1\\Downloads\\sample.json')
#dataset = df.to_csv('sample.json', sep='\t', encoding='utf-8')
'''
K-means clustering is applied on the genre and price of books. 
'''
X = dataset.iloc[:, [4,7]].values
# y = dataset.iloc[:, 3].values



'''
Using the elbow method to find the optimal number of clusters
'''

wcss = []
for i in range(1, 25):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
#line=plt.plot(range(1, 25), wcss)
'''
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Books')
plt.show()
#fig1=plt.figure()
'''
#line=fig1.gca().get_lines()[4]
#xd=line[0].get_xdata()
#yd=line[0].get_ydata()
#print (xd)
#print (yd)

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 12).fit(X)
y_kmeans = kmeans.fit_predict(X)
labels=kmeans.labels_
centroids = kmeans.cluster_centers_


'''
Extracting necessary data from each cluster.
'''

cluster_map = pd.DataFrame()
#cluster_map['asin'] = dataset.asin.values
cluster_map['title'] = dataset.title.values
cluster_map['genre']= dataset.genre.values
cluster_map['price']= dataset.price.values
cluster_map['description'] = dataset.description.values
cluster_map['cluster'] = kmeans.labels_
'''
print (cluster_map[cluster_map.cluster == 0])
print (cluster_map[cluster_map.cluster == 1])
print (cluster_map[cluster_map.cluster == 2])
print (cluster_map[cluster_map.cluster == 3])
print (cluster_map[cluster_map.cluster == 4])
print (cluster_map[cluster_map.cluster == 5])
print (cluster_map[cluster_map.cluster == 6])
print (cluster_map[cluster_map.cluster == 7])
print (cluster_map[cluster_map.cluster == 8])
print (cluster_map[cluster_map.cluster == 9])
print (cluster_map[cluster_map.cluster == 10])
print (cluster_map[cluster_map.cluster == 11])

'''
'''
print cluster_map[cluster_map.cluster == 12]
print cluster_map[cluster_map.cluster == 13]
print cluster_map[cluster_map.cluster == 14]
print cluster_map[cluster_map.cluster == 15]
print cluster_map[cluster_map.cluster == 16]
print cluster_map[cluster_map.cluster == 17]
print cluster_map[cluster_map.cluster == 18]
print cluster_map[cluster_map.cluster == 19]
'''
#inp= input ('Enter book name: ')



WORD = re.compile(r'\w+')

'''
  Function to: Calculate cosine similarity score
  
  Returns: Cosine similarity between two texts
  
  Arguments: Two vectors from texts/descriptions
'''
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

'''
    Method to: Convert text to vector
    
    Returns: A vector representation of text
    
    Arguments: Text/String
'''
def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

'''
    Getting the appropriate cluster
    
    Function to: Recommend books
    
    Argument: User Input (Book Name)
'''
def callback(inp):
    
    #print (inp)
    cosine_sum = 0 
    count = 0
    ##reuire genre settinmg
    req_genre = 1
    for i, row in cluster_map.iterrows():
            cluster_title = row ['title']
            if (str(cluster_title).lower() == str(inp.lower())):
                req_cluster = row ['cluster']
                req_genre = row ['genre']
                req_desc = row ['description']
                text1 = req_desc
                count = 0 
                arr_cosine = array.array('d', []) 
                arr_title = []
                arr_desc = []
                arr_price = array.array('d',[])
                for j, row in cluster_map.iterrows():
                    curr_genre = row ['genre'] 
                    if (curr_genre == req_genre ):
                        if (req_cluster == row ['cluster']):
                             #list_of_books['title']= row ['title']
                             #print row['title']
                             if (row ['title'] != inp):
                                 title = row ['title']
                                 price = row ['price']
                                 description = row ['description']
                                 text2 = description
                                 #list_of_books = list_of_books.append({'description': description, 'title': title, 'price': price}, ignore_index=True)
                                 count= count +1
                                 #print row['asin'], row['cluster']
                                 vector1 = text_to_vector(str(text1))
                                 if(text2!=""):
                                     vector2 = text_to_vector(str(text2))
                                     cosine = get_cosine(vector1, vector2)
                                     cosine_sum = cosine_sum + cosine
                                     arr_cosine.append(cosine)
                                     arr_title.append(title)
                                     arr_desc.append(description)
                                     arr_price.append(price)
                             #print ('Cosine:', cosine)
                                 else:
                                    arr_cosine.append(1)
                                    arr_title.append(title)
                                    arr_desc.append(description)
                                    arr_price.append(price)
                                     
                break
            
    #print cluster_map[cluster_map.cluster == req_cluster] 
                
    '''
    Recommending the top similar books 
    '''
    try:
        list_of_books = pd.DataFrame();
        cosine_mean  = (cosine_sum/  count )
        #print ('cosine mean')
        #print( cosine_mean)
        result = ''
        for index in range (count):
            if (arr_cosine[index] >= cosine_mean) :
                if ((arr_title[index]!="")):
                    
                    if ((arr_title[index].lower() != inp.lower())):
                        list_of_books= list_of_books.append({'description': arr_desc[index], 'title': arr_title[index], 'price': arr_price[index]}, ignore_index=True)
                        result+=   str((arr_title[index]))
                        result+= '\n'#+'\t'+arr_price[index]+'\n'))
           # print (list_of_books ['title'])
            
    except ZeroDivisionError:
            result =""
    if(result == ""):
        compensate_title = [ ]
        compensate_description =[ ]
        compensate_price = []
        ct=0;
        for index, row in dataset.iterrows():
            if (row['genre']==req_genre):
                title = row ['title']
                price = row ['price']
                description = row ['description']
                compensate_title.append(title)
                compensate_description.append(description)
                compensate_price.append(price)
                if(ct==5):
                 break;
                ct=ct+1;
    print('come')
    print(compensate_title)    
    
    
    print (result)
    
inpBookName = input("Enter book name! ")

inpUserID = input("Enter USER ID ")

callback(inpBookName)


#text1 = 'J. Krishnamurti (1895-1986) was a renowned spiritual teacher whose lectures and writings have inspired thousands.'
#text2 = 'J. Krishnamurti (1895-1986) was a renowned spiritual teacher .'







# Visualising the clusters
'''
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'brown', label = 'Cluster 6')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 100, c = 'purple', label = 'Cluster 7')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Books')
plt.xlabel('Book Index in File')
plt.ylabel('Price')
plt.legend()
plt.show()
#print(sortedR)
'''

