from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

print("Dataset Loaded Successfully")

X = iris.data
y = iris.target

print("Total Samples:", len(X))
print("Classes:", iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

print("Model Training Completed")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

sample = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(sample)

print("\nNew Flower Prediction:")
print("Predicted Flower:", iris.target_names[result[0]])