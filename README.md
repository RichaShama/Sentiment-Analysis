## Sentiment Analysis - COVID-19 Tweets

# Project Overview
In this project, I have created a classification system for COVID-19 tweets: positive, negative, neutral.

I have used Naive Bayes and Logistic Regression classifications and bag-of-words features.

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

<p align="center">
  <img width="45%" src="https://github.com/RichaShama/Sentiment-Analysis/blob/main/Word_cloud_All.png" alt></p>
  <p  align="center"> <em>Word Cloud of COVID-19 tweets.</em>
</p> 
<p align="center">
  <img width="65%" src="https://github.com/RichaShama/Sentiment-Analysis/blob/main/word_counts_training_set.png" alt></p>
  <p  align="center"> <em>Number of tweets in training dataset.</em>
</p>
<p align="center">
  <img width="65%" src="https://github.com/RichaShama/Sentiment-Analysis/blob/main/word_counts_test_set.png" alt></p>
  <p  align="center"> <em>Number of tweets in test dataset.</em>
</p>

# Model Building 

I have tried the following models.   

*	**Naive Bayes**
*	**Logistic Regression**

# Model performance
*	**Naive Bayes**: accuracy score = 0.61
*	**Logistic Regression**: accuracy score = 0.82



