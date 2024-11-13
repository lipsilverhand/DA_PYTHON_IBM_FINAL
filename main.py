import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import requests

# Task 1 : Import the dataset --------------------------------------------------------
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
output = './insurance.csv'
headers = ['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region', 'charges']

response = requests.get(url)
if response.status_code == 200:
    with open(output, "wb") as f:
        f.write(response.content)
        print("File downloaded successfully.")
else:
    print("Failed to download file.")

df = pd.read_csv(output, header=None)

#add headers
df.columns = headers

#Replace NaN to "?"
df.replace('?', np.nan, inplace = True)

#print first 10 rows of dataframe
#print(df.head(10))

# Task 2 : Data Wrangling ----------------------------------------------------------

# smoker is a categorical attribute, replace with most frequent entry
is_smoker = df['smoker'].value_counts().idxmax()
df['smoker'].replace(np.nan, is_smoker, inplace= True)

# age is a continuous variable, replace with mean age
mean_age = df['age'].astype('float').mean(axis=0)
df['age'].replace(np.nan, mean_age, inplace=True)

# Update data types
df[["age","smoker"]] = df[["age","smoker"]].astype("int")

print(df.info())

#Change to near 2 decimal places long in  "charges"
df[["charges"]] = np.round(df[["charges"]],2)
print(df.head())

#Task 3: Exploratory Data Analysis (EDA) ----------------------------------------------

#Implement the regression plot for charges with respect to bmi
sns.regplot(x='bmi', y='charges', data=df, line_kws={"color": "red"})
plt.ylim(0,)
plt.show()

#Implement the box plot for charges with respect to smoker.
sns.boxplot(x='smoker', y='charges', data=df)
plt.show()

#Print the correlation matrix for the dataset.
print(df.corr())

# Task 4 : Model Development -----------------------------------------------------------

#Fit a linear regression model that may be used to predict the charges value, just by using the smoker attribute of the dataset.
lm = LinearRegression()
X = df[['smoker']]
Y = df[['charges']] #(predict)
lm.fit(X,Y)
print(lm.score(X,Y)) #(R^2)

#Fit a linear regression model that may be used to predict the charges value, just by using all other attributes of the dataset.
Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
lm.fit(Z,Y)

#Create a training pipeline that uses StandardScaler(), PolynomialFeatures() and LinearRegression() to create a model that can predict the charges
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print(r2_score(Y,ypipe))

# Task 5 : Model Refinement (Evaluation)

#Split the data into training and testing subsets, assuming that 20% of the data will be reserved for testing
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)

#Initialize a Ridge regressor that used hyperparameter, alpha = 0.1
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test, yhat))

# Apply polynomial transformation to the training parameters with degree=2. Use this transformed feature set to fit the same regression model, as above, using the training subset. Print the score for the testing subset.
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))