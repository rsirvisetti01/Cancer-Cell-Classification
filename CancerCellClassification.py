import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

# Normalize data so logistic regression is more efficient
scaler = StandardScaler()

# Scaled input data
X = scaler.fit_transform(data.data)

# Expected output data
y = data.target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Fitting the data to a logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Training prediction accuracy
y_train_pred = log_reg.predict(X_train)
print(f"Accuracy on training data: {accuracy_score(y_train, y_train_pred): .4f}")

# Testing prediction accuracy
y_test_pred = log_reg.predict(X_test)
print(f"Accuracy on testing data: {accuracy_score(y_test, y_test_pred): .4f}")