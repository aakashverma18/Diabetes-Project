##Diabetes Project
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.csv')
data.head(20)

data.columns
data.dtypes
#To check missing values
data.isnull().sum()

#Data imputation of o's in every feature.

data.shape # To know the size of data, last one is a target column.
#Target column [0,1] - Binary classification/ Supervised ML/

#EDA
#Correlation heatmap

data.corr() #[-1,1]; -1 = -vely correlated, +1 = +vely correlated

plt.figure(figsize = (15, 15))
ax = sns.heatmap(data.corr(), annot=True) #annotate to denote the values.
plt.savefig( 'correlation-coefficient.jpg')
plt.show()

#Heatmap shows all features are important.

data.describe() #to describe the mean, std, quantiles and more statistical terms.

sns.displot(data.BMI)
##Insulin -> right skewed distribution
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].median())
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())
data['Pregnancies'] = data['Pregnancies'].replace(0, data['Pregnancies'].median())
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].median())
data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
data['DiabetesPedigreeFunction'] = data['DiabetesPedigreeFunction'].replace(0, data['DiabetesPedigreeFunction'].median())
data['Age'] = data['Age'].replace(0, data['Age'].median())
data.head(20)

# divide the data into input features and target values
# x-> inpute features , y-> target values
x = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

## Outlier detection using Boxplot
fig, ax = plt.subplots(figsize=(15,15))
sns.boxplot(data = x, ax=ax)
plt.savefig('boxPlot.jpg')

## Droping outliers

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for col in cols:
    Q1 = x[col].quantile(0.25)
    Q3 = x[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    mask = (x[col] >= lower_bound) & (x[col] <= upper_bound)

x_outlier_detection = x[mask]
y_outlier_detection = y[mask]
y_outlier_detection.shape

#Standardization/ Normalization of values
# It converts the data into Std Normal Form, mean =0, stdev = 1.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_outlier_detection)

#Plotting them
fig, ax = plt.subplots(figsize=(15,15))
sns.boxplot(data = x_scaled, ax=ax)
plt.savefig('boxPlot.jpg')
x_scaled.shape

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x_scaled = pd.DataFrame(x_scaled, columns =cols)
x_scaled.describe()

x_scaled.shape

#Approach 2 of quantiles to remove the outliers
#Handling of imbalance data.
x_scaled.reset_index(drop=True, inplace=True)
y_outlier_detection.reset_index(drop=True, inplace=True)
q = x_scaled['Insulin'].quantile(.95)
mask = x_scaled['Insulin'] < q
dataNew = x_scaled[mask]
y_outlier_detection = y_outlier_detection[mask]

y_outlier_detection.shape

#Model Training and split the data into training and test.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataNew, y_outlier_detection, test_size=0.33, random_state =42)

#This is the issue of data imbalancing.
#We will use oversampling and undersampling, SMOTE= synthetic data most of the times to resolve it.

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

#Checking resampled distribution
print('\n Resampled class Distribution: ')
print(pd.Series(y_train_resampled).value_counts())

from sklearn.linear_model import LogisticRegression
classification = LogisticRegression()
classification.fit(x_train_resampled, y_train_resampled)

#To save the model and use it in future, we use pickle
import pickle
pickle.dump(classification, open('classification_model.pkl', 'wb'))

#Model Predictions

y_predictions = classification.predict(x_test)
print(y_predictions)

#Model Accuracy Check
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predictions)

from sklearn.metrics import classification_report #To get the confusion metrics
target_names = ['Non-Diabetic', 'Diabetic']
print(classification_report(y_test, y_predictions, target_names = target_names))

from sklearn.naive_bayes import GaussianNB
model_gaussian_naive_bayes  = GaussianNB()
model_gaussian_naive_bayes.fit(x_train_resampled, y_train_resampled)

y_predict_gaussian_naive_bayes = model_gaussian_naive_bayes.predict(x_test)
print(y_predict_gaussian_naive_bayes)

from sklearn.metrics import confusion_matrix
print('Confusion matrix')
print(confusion_matrix(y_test, y_predict_gaussian_naive_bayes))

accuracy_score(y_test, y_predict_gaussian_naive_bayes)

print('Classification Report')
print(classification_report(y_test, y_predict_gaussian_naive_bayes, target_names = target_names))

#Loading
classification_model = pickle.load(open('classification_model.pkl', 'rb'))
classification_model.predict(x_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

#To check the score on tests data and values.

rf.score(x_test, y_test)

y_predict_rf = rf.predict(x_test)
print(y_predict_rf)