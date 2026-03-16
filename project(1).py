import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('insurance.csv')
#eda
print(df.shape)
print(df.head()) 
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)
numneric_columns=['age','bmi','children','charges']
columns=['age','sex','bmi','smoker','children','chargers','region']
for col in numneric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col],kde=True,bins=20)
plt.show()
sns.countplot(x=df['sex'])
plt.show()
sns.countplot(x=df['smoker'])
plt.show()
for col in numneric_columns:
    plt.figure(figsize=(6,4))
    plt.boxplot(x=df[col])
plt.show()
for col in numneric_columns:
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()
# # #data cleaning and preprocessing
df_cleaned=df.copy()
df_cleaned.head()
print(df_cleaned)
df_cleaned.drop_duplicates(inplace=True)
df_cleaned.shape
print(df_cleaned)
df_cleaned.isnull().sum()
print(df_cleaned)
print(df_cleaned['sex'].value_counts())
df_cleaned['sex']=df_cleaned['sex'].map({"male":0,"female":1})
print(df_cleaned.head())
print(df_cleaned['smoker'].value_counts)
df_cleaned['smoker']=df_cleaned['smoker'].map({"yes":1,"no":0})
print(df_cleaned.head())
df_cleaned.rename(columns={
    'sex':'is_female',
    'smoker':'is_smoker'

},inplace=True)
print(df_cleaned)
print(df['region'].value_counts())
print(df)
df_cleaned=pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)
print(df_cleaned.head())
df_cleaned=df_cleaned.astype(int)
print(df_cleaned)
# #
df_cleaned['bmi_category']=pd.cut(
    df_cleaned['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['underweight','normal','over weight','obese']
)
print(df_cleaned)
df_cleaned=pd.get_dummies(df_cleaned,columns=['bmi_category'],drop_first=True)
print(df_cleaned.head())
df_cleaned=df_cleaned.astype(int)
print(df_cleaned)
from sklearn.preprocessing import StandardScaler
cols=['age','bmi','children']
scaler=StandardScaler()
df_cleaned[cols]=scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())