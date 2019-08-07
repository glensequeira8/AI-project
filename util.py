"""
    CSI-535 || Artificial Intelligence Team project
    Team : Reviewers
    Topic : Sentiment analysis using product review data
    Authors: Rahul Chhapgar, Glen Sequeira, Nilesh Chakraborty
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import TaggerI
from nltk.classify.maxent import MaxentClassifier
from nltk.corpus import treebank
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import json
import nltk
import numpy as np
from twython import Twython
import time
from sklearn.svm import SVC

### ---------------------------------------------------- Global -------

# Global lists and Dictionaries
Global_dict = {}
Sentiment_Score = {}
NOV = []    # Negation of Adjectives
NOA = []    # Negation of Adverbs

# defining lists of Tokens which are generated based on Tokens names from NLTK Word-Tokens
POS_arr = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
VERB = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']    # Verbs
ADJ = ['JJ', 'JJR', 'JJS']  # Adjectives
ADV = ['RB', 'RBR', 'RBS']  # Adverbs

stopwords = \
    ['b', 'c' 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
     'x', 'y', 'z', '@', '#', '..', '.', '`', '~', '!', '$', '%', '^', '*', '(', ')', '-', '+', '/"', '//',
     '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '``', ';']

Count_1 = 0
Count_2 = 0
Count_3 = 0
Count_4 = 0
Count_5 = 0
