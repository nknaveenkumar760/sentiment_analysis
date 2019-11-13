What is Sentiment Analysis?

Sentiment analysis is the automated process of analyzing text data and classifying opinions as negative, positive or neutral. Usually, besides identifying the opinion, these systems extract attributes of the expression e.g.:

Polarity: if the speaker express a positive or negative opinion,
Subject: the thing that is being talked about,
Opinion holder: the person, or entity that expresses the opinion.

Sentiment Analysis Metrics and Evaluation

There are many ways in which you can obtain performance metrics for evaluating a classifier and to understand how accurate a sentiment analysis model is. One of the most frequently used is known as cross-validation.

What cross-validation does is splitting the training data into a certain number of training folds (with 75% of the training data) and a the same number of testing folds (with 25% of the training data), use the training folds to train the classifier, and test it against the testing folds to obtain performance metrics (see below). The process is repeated multiple times and an average for each of the metrics is calculated.

