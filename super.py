import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df=sns.load_dataset('titanic')
print(df.head())
print(df.columns)
# print(df.info())
print(df.isnull().sum())
df.drop(['deck', 'embark_town','alive','class','who','adult_male'],axis=1,inplace=True)
print(df)
print(df.info())
df['age']=df['age'].fillna(df['age'].mean(),inplace=True)
df.dropna(subset=['embarked'],inplace=True)
print(df.info())
#label encode we have to do for other than integer
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['sex']=le.fit_transform(df['sex'])
df['embarked']=le.fit_transform(df['embarked'])
df=df.astype(int)
print(df.head())
x=df.drop('survived',axis=1)
y=df['survived']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(y_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)
#knn-knearest neighbour
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train,y_train)
y_pred_knn=knn_model.predict(x_test)
print(y_pred_knn)
print(y_test)
print(accuracy_score(y_test,y_pred_knn))
print(confusion_matrix(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn))
#navie bayes
from sklearn.naive_bayes import GaussianNB
model_nb=GaussianNB()
model_nb.fit(x_train,y_train)
y_pred_nb=model_nb.predict(x_test)
print(y_pred_nb)
print(y_test)
print(accuracy_score(y_test,y_pred_nb))
print(confusion_matrix(y_test,y_pred_nb))
print(classification_report(y_test,y_pred_nb))
#decision tree
from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier(random_state=42)
model_dt.fit(x_train_scaled,y_train)
y_pred_dt=model_dt.predict(x_test_scaled)
print(accuracy_score(y_test,y_pred_dt))
print(confusion_matrix(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))
#support vector machine
from sklearn.svm import SVC
model_svm=SVC(kernel='rbf')
model_svm.fit(x_train_scaled,y_train)
y_pred_svm=model_svm.predict(x_test_scaled)
print(accuracy_score(y_test,y_pred_svm))
print(confusion_matrix(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm))