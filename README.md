### Sentiment Analysis - COVID-19 Tweets

# Project Overview
In this project, I have created a classification system for COVID-19 tweets: positive, negative, neutral.

I have used Naive Bayes classification and bag-of-words features.

# Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn
**Kaggle ** dataset

# Data Cleaning

* removed urls
* removed twitter usernames that begins with @
* removed hashtags
* removed digits
* converted commas etc. to space
* converted all letters to lowercase
* performed stemming so that only root of the words remain
* removed stop words except 'not'

