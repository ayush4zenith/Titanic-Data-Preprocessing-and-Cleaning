import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('titanic_dataset.csv')  #Load dataset

#Initial exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())

#Handling the missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)


#Encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

#Normalize/Standardize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

#Visualize outliers (example: Fare)
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.savefig('fare_boxplot.png')
plt.close()

df.to_csv('cleaned_titanic.csv', index=False) # Export cleaned dataset
print("Preprocessing Complete. Cleaned file saved as 'cleaned_titanic.csv'")

#Remove outliers using IQR method for 'Fare' and 'Age'
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

#Apply outlier removal
df = remove_outliers_iqr(df, 'Fare')
df = remove_outliers_iqr(df, 'Age')
