# book-recommendation-system
Book recommendation system using content based as well as collaborative approach on Amazon dataset. 
# Book Recommendation System

The system is a hybrid approach based book recommendation system. The hybrid approach is a combination of both content as well as collaborative filtering techniques.

# Dataset

The dataset has been taken from Amazon. (will add the credits soon).

A structured dataset representation from the unstructured dataset to build the proposed system.

# Files Description

- combined.csv: Contains data about different books. This file is used in contentbased.py
- contentbased.py: Python file to implement content based filtering method to recommend books. 
                   Input: A book name (that exists in combined.csv)
                   Output: Recommendations according to input
                   Notes and issues: In case there are no recommendations or invalid input , five books of genre 'Humour' will be shown. 
- collaborativeFilter.py : Python file to implement collaborative filtering method. 
                                     
