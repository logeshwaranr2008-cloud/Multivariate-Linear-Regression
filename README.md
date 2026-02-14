# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
Step1

Import the necessary Python libraries for data processing and machine learning.

Step2

Read the data from a CSV file into a DataFrame for processing.

Step3

Separate the input (independent variables) and output (dependent variable) from the dataset.

Step4

Initialize a linear regression model from the machine learning library.

Step5

Fit the model using the input features and target to learn their relationship.

Step6
Retrieve and display the coefficients and intercept that the model has learned.

Step7
Use the trained model to predict the target value for a new set of input features.

Step8
Run the program.
## Program:
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

# load the boston dataset manually
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error
plt.style.use('fivethirtyeight')

# training data residuals
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# test data residuals
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

# zero error line
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()
```
## Output:

<img width="1240" height="725" alt="Screenshot 2026-02-14 085218" src="https://github.com/user-attachments/assets/c74fa72c-439d-4927-8b4e-c6d672643bd3" />

### Insert your output

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
