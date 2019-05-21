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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
l1 = LogisticRegression()
l1.fit(X_train,y_train)
p1 = l1.predict(X_test)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
l2= LogisticRegression()
rfecv = RFECV(estimator=l2, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X_train, y_train)
print(rfecv.transform(X_train)[:1,:])
print(X_train.head(1))
print('By comparing the two we find the feature not selected')
print('Number of best suited features using RFFECV')
print(rfecv.n_features_)
p2=rfecv.predict(X_test)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaled_data = scaler.transform(X_train)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(scaled_data)
xtrain_pca = pca.transform(scaled_data) 
xtest_pca=pca.transform(scaler.transform(X_test))
l3=LogisticRegression()
l3.fit(xtrain_pca,y_train)
p3 = l3.predict(xtest_pca)
df_comp = pd.DataFrame(pca.components_,columns=X.columns)
print('PCA components for the features')
print(df_comp)
from sklearn.metrics import classification_report
print('model 1')
print(classification_report(y_test,p1))
print('model 2')
print(classification_report(y_test,p2))
print('model 3')
print(classification_report(y_test,p3))




