import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline use this for jupyter notebook
ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')
sns.jointplot(x='Age',y='Area Income',data=ad_data)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
l= LogisticRegression()
rfecv = RFECV(estimator=l, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfecv.fit(X_train, y_train)
rfecv.transform(X_train)
print('Number of best suited features using RFFECV')
print(rfecv.n_features_)
p1=rfecv.predict(X_test)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print('model 1')
print(classification_report(y_test,predictions))
print('model 2')
print(classification_report(y_test,p1))





