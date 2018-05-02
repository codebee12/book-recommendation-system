# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:21:48 2018

@author: USER-1
"""
from collections import defaultdict
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split




def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

dataset = pd.read_csv('C:\\Users\\USER-1\\Downloads\\review_sample.csv')

temp_data = pd.DataFrame();

temp_data= dataset[['reviewerID','asin','overall']]
intermediate_data = pd.DataFrame();
arr_sents = pd.DataFrame()
#for index, row in dataset.iterrows():
#temp_data['review_score'] = temp_data['reviewText'].apply(lambda x : get_tweet_sentiment(x))
#temp_data.drop(['reviewText'], axis= 1, inplace =True )    
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0,5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(temp_data[['reviewerID', 'asin', 'overall']], reader)

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)
for i in range(0,9):
    print (top_n['A2XQ5LZHTD4AFT'][i][0]);