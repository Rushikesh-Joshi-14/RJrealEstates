# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report


iris = load_iris()
X = iris.data  
y = iris.target  


X = X[y != 2]
y = y[y != 2]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)  


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


new_data = [[5.0, 3.5, 1.5, 0.2]] 
prediction = model.predict(new_data)
print("Prediction for new data:", iris.target_names[prediction[0]])
