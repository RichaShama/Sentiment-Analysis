#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Corona tweet Sentiment Analysis

Natural Language Processing 

@author: Richa Sharma
"""


# Importing the main libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset_train = pd.read_csv('Corona_NLP_train.csv', encoding='latin_1')
dataset_test = pd.read_csv('Corona_NLP_test.csv', encoding='latin_1')


print(dataset_train.head())
print(dataset_test.head())

savename='Word_cloud_All'
savename1='word_counts_training_set'
savename2='word_counts_test_set'


# Converting five to three sentiments



def change_sentiment(sentiment):
    if sentiment == "Extremely Positive":
        return 'positive'
    elif sentiment == "Extremely Negative":
        return 'negative'
    elif sentiment == "Positive":
        return 'positive'
    elif sentiment == "Negative":
        return 'negative'
    else:
        return 'neutral'

dataset_train['Sentiment'] = dataset_train['Sentiment'].apply(lambda x: change_sentiment(x))
dataset_test['Sentiment'] = dataset_test['Sentiment'].apply(lambda x: change_sentiment(x))


# Plotting Sentiment counts

import seaborn as sns

plt.figure()
sns.countplot(dataset_train['Sentiment'])
plt.savefig(savename1+'.pdf', dpi=600, bbox_inches='tight')

plt.figure()
sns.countplot(dataset_test['Sentiment'])
plt.savefig(savename2+'.pdf', dpi=600, bbox_inches='tight')


# We only need two columns for sentiment analysis

dataset_train = dataset_train.iloc[:,4:]
dataset_test = dataset_test.iloc[:,4:]
print(dataset_train.head())




# Cleaning the text

import re
import nltk # to download ensemble of stop words
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # to apply stemming




def clean(text):

    #     remove urls
    text = re.sub(r'http\S+', " ", text)

    #     remove mentions
    text = re.sub(r'@\w+',' ',text)

    #     remove hastags
    text = re.sub(r'#\w+', ' ', text)

    #     remove digits
    text = re.sub(r'\d+', ' ', text)

    #     remove html tags
    text = re.sub('r<.*?>',' ', text)
    
    #     convert anything left like commas etc. to a space
    text = re.sub('[^a-zA-Z]', ' ', text)


    #     remove all the upper case letters
    text = text.lower()
    text = text.split()
        
    #     stemming, eg. loved become love
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
    
    #     remove stop words    

    text = " ".join(text)
    
      
    return text


dataset_train['OriginalTweet'] = dataset_train['OriginalTweet'].apply(lambda x: clean(x))
dataset_test['OriginalTweet'] = dataset_test['OriginalTweet'].apply(lambda x: clean(x))


print(dataset_train.head())
print(dataset_test.head())



# Mapping the sentiment ---> 0, 1, 2

mapping = {"neutral":0, "positive":1,"negative":2}

dataset_train['Sentiment'] = dataset_train['Sentiment'].map(mapping)
dataset_test['Sentiment']  = dataset_test['Sentiment'].map(mapping)


#print(dataset_train.head())
#print(dataset_test.head())




# Word Cloud of the cleaned tweets

text_tweet = ' '.join([i for i in dataset_train['OriginalTweet']]) # conver all tweets into a single string to generate wordcloud


from wordcloud import WordCloud, STOPWORDS


plt.figure(figsize=(18,18))

cloud = WordCloud().generate(text_tweet)
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.savefig(savename+'.pdf', dpi=600, bbox_inches='tight')

plt.show()



corpus = []

corpus = dataset_train['OriginalTweet']


# Creating the Bag-of-Words model


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 800)
X1 = cv.fit_transform(corpus).toarray() # training set 
y1 = dataset_train.iloc[:, -1].values


#print(len(X[0])) to check number of columns = 22501


# Transforming Test data to vector

test_data = dataset_test['OriginalTweet']

X_test = cv.transform(test_data).toarray()
y_test = dataset_test.iloc[:, -1].values


print(X1.shape)
print(X_test.shape)




# Training the NAIVE BAYES Model on Training Set

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X1, y1)


# Predicting the Test set results


y_pred = classifier.predict(X_test)

#print(X_test.dtype)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Making the Confusion Matrix


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))





















