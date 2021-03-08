## Sentiment Analysis - COVID-19 Tweets

# Project Overview
In this project, I have created a classification system for COVID-19 tweets: positive, negative, neutral.

I have used Naive Bayes classification and bag-of-words features.

# Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn
**Kaggle:** dataset

# Data Cleaning

* removed urls
* removed twitter usernames that begins with @
* removed hashtags
* removed digits
* converted commas etc. to space
* converted all letters to lowercase
* performed stemming so that only root of the words remain
* removed stop words except 'not'

# Data Visualisation

![alt text](https://github.com/RichaShama/Sentiment-Analysis/blob/main/Word_cloud_All.pdf?raw=true)

![alt text](https://github.com/RichaShama/Sentiment-Analysis/blob/main/word_counts_training_set.pdf "Number of COVID-19 tweets in training dataset")

![alt text](https://github.com/RichaShama/Sentiment-Analysis/blob/main/word_counts_test_set.pdf "Number of COVID-19 tweets in test dataset")



# Model Building 

I have tried the following two models. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried the following models:
*	**Naive Bayes**
*	**Logistic Regression** 

# Model performance
*	**Naive Bayes**: accuracy score = 0.61
*	**Logistic Regression**: accuracy score = 



