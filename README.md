In this project logistic regression has been used on an advertising database, to predict whether a user will click on ad or not.
Initially visualisations have been done to slightly interpret the data.
Then we have used three models to predict, 1-Using the orignal dataset for Lor,
2-Doing feature selection using recursive forward elimination with cross validation and then LoR,
3-Doing feature extraction using Principal component analysis and then LoR.
It was found that rfecv used 4 features for best results. The feature not selected was found out by comparing initial dataset with one transformed using rfecv.
Getting an insight from rfecv, PCA was used to reduce data into 4 features.
The pca components for the 4 feautres was calculated.
Predictions were done on the test data for all the three models and classification reports were made.
We find that model 3 gave best result with F1 score of 0.97, followed by model 2 with F1 score of 0.93 and lastly model 1 with f1 score of 0.91.
