import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('heart.csv')
df.head()
print(df)
#EDA
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe)
print(df.duplicated().sum())
print(df['HeartDisease'].value_counts())
print(df.isnull().sum())
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True)
plotting('Age',1)   
plotting('RestingBP',2)   
plotting('Cholesterol',3)   
plotting('MaxHR',4)
plt.tight_layout()
plt.show()
print(df['Cholesterol'].value_counts())
ch_mean=df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()
df['Cholesterol']=df['Cholesterol'].replace(0,ch_mean)
print(df['Cholesterol'].head())
df['Cholesterol']=df['Cholesterol'].round(2)
resting_bp_mean=df.loc[df['RestingBP']!=0,'RestingBP'].mean()
df['RestingBP']=df['RestingBP'].replace(0,resting_bp_mean)
print(df['RestingBP'].head())
df['RestingBP']=df['RestingBP'].round(2)
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True)
plotting('Age',1)   
plotting('RestingBP',2)   
plotting('Cholesterol',3)   
plotting('MaxHR',4)
plt.tight_layout()
plt.show()
import sheryanalysis as sh
sh.analyze(df)
sns.countplot(x=df['Sex'],hue=df['HeartDisease'])
plt.show()
sns.countplot(x=df['ChestPainType'],hue=df['HeartDisease'])
plt.show()
sns.countplot(x=df['FastingBS'],hue=df['HeartDisease'])
plt.show()
sns.boxplot(x='HeartDisease',y='Cholesterol',data=df)
plt.show()
sns.violinplot(x='HeartDisease',y='Age',data=df)
plt.show()
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()
#data  preprocessing and cleaning
df_encode=pd.get_dummies(df,drop_first=True)
print(df_encode)
df_encode=df_encode.astype(int)
print(df_encode)
from sklearn.preprocessing import StandardScaler
numerical_cols=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
scalar=StandardScaler()
df_encode[numerical_cols]=scalar.fit_transform(df_encode[numerical_cols])
df_encode.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
x=df_encode.drop('HeartDisease',axis=1)
y=df_encode['HeartDisease']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.fit_transform(x_test)
models={
    "logistic Regression":LogisticRegression(),
    "knn":KNeighborsClassifier(),
    "navie bayes":GaussianNB(),
    "decision tree":DecisionTreeClassifier(),
    "svm":SVC()
}
result=[]
for name,model in models.items():
    model.fit(x_train_scaler,y_train)
    y_pred=model.predict(x_test_scaler)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred,average='weighted')
    result.append({
        'model':name,
        'accuracy':round(acc,4),
        'f1_score':round(f1,2)
    })
    print(result)
    
