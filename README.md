# CharityML

## Project Motivation
In this project, I employ several supervised algorithms to accurately model individuals' income 
using data collected from the 1994 U.S. Census. 

I then choose the best candidate algorithm from preliminary results
and further optimize this algorithm to best model the data. 

The goal is to construct a model that accurately predicts whether an individual makes more than $50,000. 
The dataset for this project originates from the UCI Machine Learning Repository.

## Libraries Used
* numpy
* pandas
* time
* seaborn 
* matplotlib.pyplot
* Scikitlearn

## Files in this repository
finding_donors.html: HTML version of finding_donors.ipynb
finding_donors.ipynb: Jupyter notebook where the code for this project is developed and tested
finding_donors.py: A python script containing the code developed in finding_donors.ipynb
visuals.py: A python script containing code for creating visualizations in finding_donors.py

## Results of Analysis
I tested three machine learning algorithms: Gaussian Naive Bayes, Adaptive Boosting, and the 
Random Forest Algorithm.

The Random Forest model had the best performance in terms of accuracy and computational time,
and so I used it to build the prediction model I needed. 

Also, after reducing the number of features used for prediction to 5, there was only a 1.5% 
decrease in accuracy and a 4% decrease in F-score. This is not too large, so if computational 
efficiency was a factor, then I would consider working with only the top 5 features in the dataset.
