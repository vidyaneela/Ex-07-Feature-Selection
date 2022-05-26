# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE:
```
Developed by:Vidya Neela
Reg No:212221230120
```
```
#Importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()

df.columns = boston['feature_names']
df.head()

df['PRICE']= boston['target']
df.head()

df.info()

plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()

#FILTER METHODS

X=df.drop("PRICE",1)
y=df["PRICE"]

from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape

#1.Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)

#2.Information gain/Mutual Information
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

#3.SelectKBest Model
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2) 
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))

#4.Correlation Coefficient
cor=df.corr()
sns.heatmap(cor,annot=True)

#5.Mean Absolute Difference
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='teal')

#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)

#6.Chi Square Test
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

#7.SelectPercentile method
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape

#WRAPPER METHOD

#1.Forward feature selection

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X, y)
sfs.k_feature_names_

#2.Backward feature elimination

sbs = SFS(LinearRegression(),
         k_features=10,
         forward=False,
         floating=False,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#3.Bi-directional elimination

sffs = SFS(LinearRegression(),
         k_features=(3,7),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_

#4.Recursive Feature Selection

from sklearn.feature_selection import RFE
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=7)
rfe.fit(X, y)
print(X.shape, y.shape)
rfe.transform(X)
rfe.get_params(deep=True)
rfe.support_
rfe.ranking_

#EMBEDDED METHOD

#1.Random Forest Importance

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X,y_transformed)
importances=model.feature_importances_

final_df=pd.DataFrame({"Features":pd.DataFrame(X).columns,"Importances":importances})
final_df.set_index("Importances")
final_df=final_df.sort_values("Importances")
final_df.plot.bar(color="teal")
```

# OUPUT:

![pics1](https://user-images.githubusercontent.com/94169318/170408106-fc70065f-c83f-49e2-9b19-a1121f3a0080.jpeg)
![pics2](https://user-images.githubusercontent.com/94169318/170408132-911b0b71-462b-40a0-b5aa-3f3e880508fa.jpeg)
![pics3](https://user-images.githubusercontent.com/94169318/170408358-47134aa9-c0ea-4cbb-8140-b9c6acf3ab08.jpeg)
![pics4](https://user-images.githubusercontent.com/94169318/170408365-631cee50-4996-423f-bab3-f0cbf83fdc6f.jpeg)
![pics5](https://user-images.githubusercontent.com/94169318/170408373-e3f756a8-be51-4b0e-abb5-fd92832deedf.jpeg)
![pics6](https://user-images.githubusercontent.com/94169318/170408382-ff66246d-1f07-4686-9ddc-3524d819a0f2.jpeg)
![pics7](https://user-images.githubusercontent.com/94169318/170408390-1c624742-cc32-4197-8966-744738a387e9.jpeg)
## FILTER METHODS:
![pics8](https://user-images.githubusercontent.com/94169318/170408400-e198a097-61e0-4985-8553-fa659ab618db.jpeg)![pics10](https://user-images.githubusercontent.com/94169318/170408528-d4eeb3d1-0a69-4c3f-99a3-010323fbbb56.jpeg)

## 2.Information gain/Mutual Information:
![pics9](https://user-images.githubusercontent.com/94169318/170408411-957c8662-afdc-408f-9337-28ac9588c3e7.jpeg)

## 3.SelectKBest Model:
![pics10](https://user-images.githubusercontent.com/94169318/170408535-c930faed-3096-41f8-af68-840d70e48023.jpeg)

## 5.Mean Absolute Difference:
![pics11](https://user-images.githubusercontent.com/94169318/170408545-a3ea44e1-1681-43dd-9e1f-3042ef84f957.jpeg)

![pics12](https://user-images.githubusercontent.com/94169318/170408555-3add27fe-4b07-49f9-b83b-73b560b3a422.jpeg)

## 6.Chi Square Test:
![pics13](https://user-images.githubusercontent.com/94169318/170408562-3d71867f-7339-4139-9ad7-b02dcad6f94b.jpeg)

## 7.SelectPercentile method
![pics14](https://user-images.githubusercontent.com/94169318/170408569-5346ba02-85a6-4c82-bd5c-7b707b9d05bf.jpeg)

# WRAPPER METHOD

## 1.Forward feature selection:
![pics15](https://user-images.githubusercontent.com/94169318/170408580-1ab01083-c7d4-4118-bd8f-4f4b3d4c7b4f.jpeg)

## 2.Backward feature elimination:
![pics16](https://user-images.githubusercontent.com/94169318/170408592-b94d8ce4-fe9c-4f84-9527-dcae68239a63.jpeg)

## 3.Bi-directional elimination
![pics17](https://user-images.githubusercontent.com/94169318/170408601-ba67fafe-e3c7-4519-86b8-7b26ab16845f.jpeg)

## 4.Recursive Feature Selection
![pics18](https://user-images.githubusercontent.com/94169318/170408624-0e9c5e31-6271-4637-a4d1-aafb6561a1b6.jpeg)

## EMBEDDED METHOD

## 1.Random Forest Importance:
![pics19](https://user-images.githubusercontent.com/94169318/170408653-23600260-3caa-4c84-9371-66075fcea96f.jpeg)

## RESULT:
























