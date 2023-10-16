""" 
            Ex 1. Running Classical ML Models using OpenNLP

OpenNLP provides 6 classical machine learning tools for NLP. 
1. Decision Tree
2. Random Forest 
3. Multinomial Bayesian classification model
4. Gradient Boost 
5. Adaboost 
6. Lineaer SVC
"""

# Import denpendencies 
from opennlp.run.ml import ClassicalML

# Create an instance from the class
# Test size is set as 0.2 (Default), users can change it. 
classical_ml=ClassicalML(data_path='./data/sample_sentiment.csv', # Your data path
                         input_col='tweets', # Your data input column
                         output_col='labels', # Your data output column
                         seed=42) # Your random seed

# Run Decision Tree
classical_ml.run_DecisionTree()

# Run random forest
classical_ml.run_RandomForest(n_estimators=200)

# Run MNB
classical_ml.run_MNB(alpha=1.0)

# Run Gradboost
classical_ml.run_GradBoost(n_estimators=200)

# Run Adaboost
classical_ml.run_AdaBoost(n_estimators=200)

# Run SVC
classical_ml.run_SVC()

"""
Results of those models will be saved at './Results/<model name>_<parameters>'
For example, if you run Random forest model with 200 estimators,
Results will be saved in ./Results/RandomForest_estimator200
Results include runtime,confusion matrix and clcassification report.
"""