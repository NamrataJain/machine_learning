#Importing Libraries for data preparation
import pandas as pd
import numpy as np
#Read Necessary files
train = pd.read_csv("E:/ML/BlackFriday/train_black_friday.csv")
test = pd.read_csv("E:/ML/BlackFriday/test_black_friday.csv")
#Combined both Train and Test Data set to do preprocessing together # and set flag for both as well
train['Type'] = 'Train'
test['Type'] = 'Test'
fullData = pd.concat([train,test],axis=0)
print(fullData)
print(train['Type'])
#Identifying ID, Categorical
ID_col = ['User_ID','Product_ID']
flag_col= ['Type']
target_col = ["Purchase"]

cat_cols= ['Gender','Age','City_Category','Stay_In_Current_City_Years']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col)-set(flag_col))
print("cat_colscat_colscat_colscat_colscat_colscat_colscat_cols")
print(cat_cols)
print("num_colsnum_colsnum_colsnum_colsnum_colsnum_colsnum_cols")

print(num_cols)

# Combined numerical and Categorical variables
num_cat_cols = num_cols+cat_cols
#Create a new variable for each variable having missing value with VariableName_NA
# and flag missing value with 1 and other with 0
for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var+'_NA']=fullData[var].isnull()*1
#Impute numerical missing values with mean
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean())
#Impute categorical missing values with -9999
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)

#create label encoders for categorical features
from sklearn.preprocessing import LabelEncoder
for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype('str'))
#Normalize Data: Scale (normalize) the independent variables between 0 and 1. It will help us to converge comparatively faster.
features = list(set(list(fullData.columns))-set(ID_col)-set(target_col))
fullData[features] = fullData[features]/fullData[features].max()

print(fullData.head())